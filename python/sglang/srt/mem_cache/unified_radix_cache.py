from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class L3Segment:
    kind: str
    layer_id: int
    chunk_id: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class L3Entry:
    node_id: int
    file_path: str
    token_count: int
    page_count: int
    nbytes: int
    aligned_nbytes: int
    segments: List[L3Segment]
    created_at: float
    last_access_time: float
    node: TreeNode


@dataclass
class UnifiedRadixCacheStats:
    used_bytes: int = 0
    write_count: int = 0
    write_bytes: int = 0
    read_count: int = 0
    read_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    eviction_bytes: int = 0


class UnifiedRadixCache(RadixCache):
    """A Jetson-oriented two-tier radix cache: unified DRAM plus L3 SSD.

    This baseline intentionally avoids modeling CPU host memory as a separate
    cache tier. Evicted MHA KV segments are synchronously serialized from the
    active KV pool to raw SSD files, and restored synchronously through the
    existing CPU copy/load helpers.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        l3_dir: str,
        l3_budget_gb: float,
        l3_block_size: int,
        eviction_policy: str = "lru",
        offload_after_finish_min_tokens: int = 0,
        is_eagle: bool = False,
        tp_rank: int = 0,
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(self.kv_cache, MHATokenToKVPool):
            raise ValueError(
                "UnifiedRadixCache v1 only supports MHATokenToKVPool. "
                f"Got {type(self.kv_cache).__name__}."
            )
        if is_eagle:
            raise ValueError("UnifiedRadixCache v1 does not support EAGLE.")
        if l3_budget_gb <= 0:
            raise ValueError("--unified-radix-cache-l3-budget-gb must be > 0.")
        if l3_block_size <= 0:
            raise ValueError("--unified-radix-cache-l3-block-size must be > 0.")
        if offload_after_finish_min_tokens < 0:
            raise ValueError(
                "--unified-radix-cache-offload-after-finish-min-tokens must be >= 0."
            )

        self.l3_base_dir = Path(l3_dir).expanduser().resolve()
        self.l3_run_dir = self.l3_base_dir / (
            f"run-{int(time.time())}-{os.getpid()}-tp{tp_rank}"
        )
        self.l3_budget_bytes = int(l3_budget_gb * (1024**3))
        self.l3_block_size = l3_block_size
        self.offload_after_finish_min_tokens = offload_after_finish_min_tokens
        self.l3_entries: Dict[int, L3Entry] = {}
        self.stats = UnifiedRadixCacheStats()
        self.tp_rank = tp_rank

        self.l3_run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "UnifiedRadixCache enabled: l3_dir=%s, l3_run_dir=%s, "
            "l3_budget_bytes=%d, l3_budget_gb=%.3f, l3_block_size=%d, "
            "offload_after_finish_min_tokens=%d",
            self.l3_base_dir,
            self.l3_run_dir,
            self.l3_budget_bytes,
            l3_budget_gb,
            self.l3_block_size,
            self.offload_after_finish_min_tokens,
        )

        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=False,
            eviction_policy=eviction_policy,
            is_eagle=False,
        )

    def reset(self):
        self._clear_l3_entries()
        super().reset()

    def cache_finished_req(self, req, is_insert: bool = True):
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
        page_aligned_len = all_token_len // self.page_size * self.page_size
        page_aligned_token_ids = token_ids[:page_aligned_len]
        extra_key = req.extra_key

        super().cache_finished_req(req, is_insert=is_insert)

        if (
            is_insert
            and self.offload_after_finish_min_tokens > 0
            and page_aligned_len >= self.offload_after_finish_min_tokens
        ):
            self._offload_exact_prefix(page_aligned_token_ids, extra_key)

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key.token_ids = self.key_convert_fn(key.token_ids)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        if len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        value, last_node, partial_l3_miss = self._match_prefix_helper_l3(
            self.root_node, key
        )
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        l3_hit_length = 0
        last_l3_node = last_node
        device_node = last_node
        while device_node.evicted and self._has_l3_entry(device_node):
            entry = self._get_l3_entry(device_node)
            entry.last_access_time = time.monotonic()
            l3_hit_length += len(device_node.key)
            device_node = device_node.parent

        if l3_hit_length > 0:
            self.stats.hit_count += 1
            logger.info(
                "UnifiedRadixCache L3 hit: node_id=%s, token_count=%d, "
                "page_count=%d, used_bytes=%d, hits=%d, misses=%d",
                last_l3_node.id,
                l3_hit_length,
                self._page_count(l3_hit_length),
                self.stats.used_bytes,
                self.stats.hit_count,
                self.stats.miss_count,
            )
        else:
            self.stats.miss_count += 1
            logger.info(
                "UnifiedRadixCache L3 miss: last_node_id=%s, partial_l3_miss=%s, "
                "used_bytes=%d, hits=%d, misses=%d",
                last_node.id,
                partial_l3_miss,
                self.stats.used_bytes,
                self.stats.hit_count,
                self.stats.miss_count,
            )

        return MatchResult(
            device_indices=value,
            last_device_node=device_node,
            last_host_node=last_l3_node,
            host_hit_length=l3_hit_length,
        )

    def insert(self, key: RadixKey, value=None, chunked=False):
        if self.disable:
            return 0
        key.token_ids = self.key_convert_fn(key.token_ids)
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)
        if len(key) == 0:
            return 0

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)

            if prefix_len < len(child.key) and child.evicted:
                # Partial L3 hits are intentionally treated as misses in v1.
                # Drop the stale SSD subtree and insert recomputed KV below node.
                self._drop_subtree(child, reason="partial-l3-insert-miss")
                break

            node = child
            if prefix_len == len(node.key):
                if node.evicted:
                    node.value = value[:prefix_len]
                    self.evictable_size_ += len(node.value)
                else:
                    total_prefix_length += prefix_len
            else:
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node
                total_prefix_length += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._record_store_event(new_node)

        return total_prefix_length

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves_device()
        leaves.sort(key=lambda n: self.eviction_strategy.get_priority(n))

        num_evicted = 0
        while num_evicted < num_tokens and leaves:
            node = leaves.pop(0)
            if node == self.root_node or node.lock_ref > 0 or node.evicted:
                continue

            evicted = self._offload_node_to_l3(node, reason="dram-evict")
            num_evicted += evicted

            parent = node.parent
            if (
                parent is not None
                and parent != self.root_node
                and not parent.evicted
                and parent.lock_ref == 0
                and all(child.evicted for child in parent.children.values())
            ):
                leaves.append(parent)
                leaves.sort(key=lambda n: self.eviction_strategy.get_priority(n))

    def init_load_back(
        self,
        last_host_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        if host_hit_length <= 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_host_node,
            )

        start_time = time.perf_counter()
        last_hit_node = last_host_node
        nodes_to_load = []
        node = last_host_node
        while node.evicted and self._has_l3_entry(node):
            nodes_to_load.insert(0, node)
            node = node.parent
        ancestor_node = node

        total_tokens = sum(len(n.key) for n in nodes_to_load)
        if total_tokens == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                ancestor_node,
            )
        if mem_quota is not None and total_tokens > mem_quota:
            logger.info(
                "UnifiedRadixCache L3 restore skipped: token_count=%d exceeds mem_quota=%d",
                total_tokens,
                mem_quota,
            )
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                ancestor_node,
            )

        delta = self.inc_lock_ref(ancestor_node)
        device_indices = self.token_to_kv_pool_allocator.alloc(total_tokens)
        if device_indices is None:
            self.evict(total_tokens)
            device_indices = self.token_to_kv_pool_allocator.alloc(total_tokens)
        self.dec_lock_ref(ancestor_node)
        if device_indices is None:
            logger.warning(
                "UnifiedRadixCache L3 restore failed: insufficient DRAM token slots "
                "for token_count=%d",
                total_tokens,
            )
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                ancestor_node,
            )

        _ = delta  # kept to mirror HiRadixCache load-back accounting shape
        offset = 0
        restored_bytes = 0
        restored_nodes = []
        for l3_node in nodes_to_load:
            token_count = len(l3_node.key)
            dst_indices = device_indices[offset : offset + token_count]
            entry = self._get_l3_entry(l3_node)
            try:
                kv_cpu = self._read_l3_entry(entry)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                logger.warning(
                    "UnifiedRadixCache L3 read failed; dropping stale entry and "
                    "falling back to recompute: node_id=%s, error=%s",
                    l3_node.id,
                    exc,
                )
                for restored_node, restored_token_count in restored_nodes:
                    restored_node.value = None
                    self.evictable_size_ -= restored_token_count
                self.token_to_kv_pool_allocator.free(device_indices)
                self._drop_subtree(l3_node, reason="stale-l3-read-failure")
                return (
                    torch.empty((0,), dtype=torch.int64, device=self.device),
                    ancestor_node,
                )
            self.token_to_kv_pool_allocator.load_cpu_copy(kv_cpu, dst_indices)
            l3_node.value = dst_indices
            self.evictable_size_ += token_count
            restored_nodes.append((l3_node, token_count))
            restored_bytes += entry.nbytes
            offset += token_count

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.stats.read_count += 1
        self.stats.read_bytes += restored_bytes
        logger.info(
            "UnifiedRadixCache L3 read: node_id=%s, token_count=%d, page_count=%d, "
            "read_bytes=%d, total_read_count=%d, total_read_bytes=%d",
            last_hit_node.id,
            total_tokens,
            self._page_count(total_tokens),
            restored_bytes,
            self.stats.read_count,
            self.stats.read_bytes,
        )
        logger.info(
            "UnifiedRadixCache L3 restore: node_id=%s, token_count=%d, "
            "page_count=%d, latency_ms=%.3f, used_bytes=%d",
            last_hit_node.id,
            total_tokens,
            self._page_count(total_tokens),
            latency_ms,
            self.stats.used_bytes,
        )
        return device_indices, last_hit_node

    def ready_to_load_host_cache(self) -> int:
        return -1

    def check_hicache_events(self):
        return None

    def clear_storage_backend(self) -> bool:
        self._clear_l3_entries()
        return True

    def _match_prefix_helper_l3(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []
        partial_l3_miss = False

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                if child.evicted:
                    partial_l3_miss = self._has_l3_entry(child)
                    break
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break

            if not child.evicted:
                value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        return value, node, partial_l3_miss

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        if child.evicted:
            raise RuntimeError(
                "UnifiedRadixCache v1 does not split evicted L3 nodes; "
                "partial L3 matches must fall back to recompute."
            )
        self._delete_l3_entry(child)
        return super()._split_node(key, child, split_len)

    def _offload_exact_prefix(self, token_ids: List[int], extra_key: Optional[str]):
        if not token_ids:
            return
        result = self.match_prefix(RadixKey(token_ids=token_ids, extra_key=extra_key))
        node = result.last_device_node
        if node == self.root_node or node.evicted:
            return
        path = []
        while node != self.root_node:
            path.append(node)
            node = node.parent
        for path_node in path:
            if not path_node.evicted:
                self._offload_node_to_l3(path_node, reason="finish-trigger")

    def _offload_node_to_l3(self, node: TreeNode, reason: str) -> int:
        if node.evicted or node.value is None:
            return 0
        if node.lock_ref > 0:
            logger.info(
                "UnifiedRadixCache L3 write skipped: node_id=%s, reason=%s, lock_ref=%d",
                node.id,
                reason,
                node.lock_ref,
            )
            return 0

        token_count = len(node.value)
        entry = self._write_l3_entry(node, reason)
        if entry is None:
            self._drop_subtree(node, reason="l3-write-failure")
            logger.warning(
                "UnifiedRadixCache L3 write failed; evicted DRAM without L3 backup: "
                "node_id=%s, token_count=%d, reason=%s",
                node.id,
                token_count,
                reason,
            )
            return token_count

        self.token_to_kv_pool_allocator.free(node.value)
        node.value = None
        self.evictable_size_ -= token_count
        self._record_remove_event(node)
        return token_count

    def _write_l3_entry(self, node: TreeNode, reason: str) -> Optional[L3Entry]:
        if node.value is None:
            return None
        kv_cpu = self.token_to_kv_pool_allocator.get_cpu_copy(node.value)
        segments, raw_nbytes, aligned_nbytes = self._build_segments(kv_cpu)
        if raw_nbytes > self.l3_budget_bytes:
            logger.warning(
                "UnifiedRadixCache L3 write skipped: entry_bytes=%d exceeds "
                "l3_budget_bytes=%d, node_id=%s",
                raw_nbytes,
                self.l3_budget_bytes,
                node.id,
            )
            return None

        self._ensure_l3_budget(raw_nbytes, protected_node_id=node.id)
        self._delete_l3_entry(node)

        file_path = self.l3_run_dir / f"node-{node.id}.bin"
        try:
            with open(file_path, "wb", buffering=0) as f:
                for segment in segments:
                    tensor = self._get_cpu_segment_tensor(kv_cpu, segment)
                    f.seek(segment.offset)
                    tensor.contiguous().view(torch.uint8).numpy().tofile(f)
                f.truncate(aligned_nbytes)
                os.fsync(f.fileno())
        except OSError as exc:
            logger.error(
                "UnifiedRadixCache L3 write failed: node_id=%s, file=%s, error=%s",
                node.id,
                file_path,
                exc,
            )
            return None

        now = time.monotonic()
        entry = L3Entry(
            node_id=node.id,
            file_path=str(file_path),
            token_count=len(node.key),
            page_count=self._page_count(len(node.key)),
            nbytes=raw_nbytes,
            aligned_nbytes=aligned_nbytes,
            segments=segments,
            created_at=now,
            last_access_time=now,
            node=node,
        )
        node.l3_entry = entry
        self.l3_entries[node.id] = entry
        self.stats.used_bytes += raw_nbytes
        self.stats.write_count += 1
        self.stats.write_bytes += raw_nbytes
        logger.info(
            "UnifiedRadixCache L3 write: node_id=%s, reason=%s, token_count=%d, "
            "page_count=%d, write_bytes=%d, aligned_bytes=%d, used_bytes=%d, "
            "budget_bytes=%d, total_write_count=%d, total_write_bytes=%d",
            node.id,
            reason,
            entry.token_count,
            entry.page_count,
            entry.nbytes,
            entry.aligned_nbytes,
            self.stats.used_bytes,
            self.l3_budget_bytes,
            self.stats.write_count,
            self.stats.write_bytes,
        )
        return entry

    def _build_segments(self, kv_cpu: List[List[Any]]):
        segments = []
        offset = 0
        raw_nbytes = 0
        for layer_id, chunks in enumerate(kv_cpu):
            for chunk_id, pair in enumerate(chunks):
                k_cpu, v_cpu = pair
                for kind, tensor in (("k", k_cpu), ("v", v_cpu)):
                    aligned_offset = self._align_up(offset)
                    nbytes = tensor.numel() * tensor.element_size()
                    segments.append(
                        L3Segment(
                            kind=kind,
                            layer_id=layer_id,
                            chunk_id=chunk_id,
                            offset=aligned_offset,
                            nbytes=nbytes,
                            shape=tuple(tensor.shape),
                            dtype=str(tensor.dtype).replace("torch.", ""),
                        )
                    )
                    offset = aligned_offset + nbytes
                    raw_nbytes += nbytes
        return segments, raw_nbytes, self._align_up(offset)

    def _read_l3_entry(self, entry: L3Entry):
        layers: List[List[Any]] = [[] for _ in range(self.kv_cache.layer_num)]
        with open(entry.file_path, "rb", buffering=0) as f:
            by_layer_chunk: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
            for segment in entry.segments:
                dtype = self._torch_dtype(segment.dtype)
                tensor = torch.empty(segment.shape, dtype=dtype, device="cpu")
                f.seek(segment.offset)
                buf = memoryview(tensor.view(torch.uint8).numpy())
                read_bytes = f.readinto(buf)
                if read_bytes != segment.nbytes:
                    raise RuntimeError(
                        f"Short L3 read for node {entry.node_id}: "
                        f"expected {segment.nbytes}, got {read_bytes}"
                    )
                key = (segment.layer_id, segment.chunk_id)
                by_layer_chunk.setdefault(key, {})[segment.kind] = tensor

        for (layer_id, chunk_id), pair in sorted(by_layer_chunk.items()):
            if "k" not in pair or "v" not in pair:
                raise RuntimeError(
                    f"Incomplete L3 segment pair: node_id={entry.node_id}, "
                    f"layer_id={layer_id}, chunk_id={chunk_id}"
                )
            while len(layers[layer_id]) <= chunk_id:
                layers[layer_id].append(None)
            layers[layer_id][chunk_id] = [pair["k"], pair["v"]]
        return layers

    def _get_cpu_segment_tensor(self, kv_cpu, segment: L3Segment):
        pair = kv_cpu[segment.layer_id][segment.chunk_id]
        return pair[0] if segment.kind == "k" else pair[1]

    def _ensure_l3_budget(self, incoming_bytes: int, protected_node_id: int):
        while (
            self.stats.used_bytes + incoming_bytes > self.l3_budget_bytes
            and self.l3_entries
        ):
            candidates = [
                entry
                for node_id, entry in self.l3_entries.items()
                if node_id != protected_node_id
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda entry: entry.last_access_time)
            self._evict_l3_entry(victim, reason="budget")

    def _evict_l3_entry(self, entry: L3Entry, reason: str):
        node = entry.node
        evicted_bytes = self._delete_l3_entry(node)
        self.stats.eviction_count += 1
        self.stats.eviction_bytes += evicted_bytes
        if node.value is None and node != self.root_node:
            self._drop_subtree(node, reason=f"l3-{reason}-eviction")
        logger.info(
            "UnifiedRadixCache L3 eviction: node_id=%s, reason=%s, "
            "evicted_bytes=%d, used_bytes=%d, total_evictions=%d, "
            "total_eviction_bytes=%d",
            entry.node_id,
            reason,
            evicted_bytes,
            self.stats.used_bytes,
            self.stats.eviction_count,
            self.stats.eviction_bytes,
        )

    def _delete_l3_entry(self, node: TreeNode) -> int:
        entry = self._get_l3_entry(node)
        if entry is None:
            return 0
        self.l3_entries.pop(entry.node_id, None)
        node.l3_entry = None
        try:
            os.remove(entry.file_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "UnifiedRadixCache failed to remove L3 file: node_id=%s, file=%s, error=%s",
                entry.node_id,
                entry.file_path,
                exc,
            )
        self.stats.used_bytes = max(0, self.stats.used_bytes - entry.nbytes)
        return entry.nbytes

    def _drop_subtree(self, node: TreeNode, reason: str):
        for child in list(node.children.values()):
            self._drop_subtree(child, reason=reason)
        self._delete_l3_entry(node)
        if node.value is not None:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
            else:
                self.protected_size_ -= len(node.value)
            self.token_to_kv_pool_allocator.free(node.value)
            node.value = None
        if node.parent is not None:
            for key, child in list(node.parent.children.items()):
                if child is node:
                    del node.parent.children[key]
                    break
        logger.info(
            "UnifiedRadixCache dropped radix subtree: node_id=%s, reason=%s",
            node.id,
            reason,
        )

    def _clear_l3_entries(self):
        if hasattr(self, "l3_entries"):
            self.l3_entries.clear()
        if hasattr(self, "stats"):
            self.stats.used_bytes = 0
        if hasattr(self, "l3_run_dir"):
            shutil.rmtree(self.l3_run_dir, ignore_errors=True)
            self.l3_run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "UnifiedRadixCache L3 cache directory reset: l3_run_dir=%s",
                self.l3_run_dir,
            )

    def _collect_leaves_device(self):
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if cur_node == self.root_node:
                stack.extend(cur_node.children.values())
                continue
            if cur_node.evicted:
                continue
            if len(cur_node.children) == 0 or all(
                child.evicted for child in cur_node.children.values()
            ):
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())
        return ret_list

    def _has_l3_entry(self, node: TreeNode) -> bool:
        return self._get_l3_entry(node) is not None

    def _get_l3_entry(self, node: TreeNode) -> Optional[L3Entry]:
        return getattr(node, "l3_entry", None)

    def _align_up(self, value: int) -> int:
        return ((value + self.l3_block_size - 1) // self.l3_block_size) * (
            self.l3_block_size
        )

    def _page_count(self, token_count: int) -> int:
        return (token_count + self.page_size - 1) // self.page_size

    @staticmethod
    def _torch_dtype(dtype_name: str):
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise RuntimeError(f"Unsupported L3 tensor dtype: {dtype_name}") from exc
