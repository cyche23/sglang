from __future__ import annotations

import heapq
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.unified_radix_cache_io import (
    AlignedPageBuffer,
    DirectPageArena,
    JsonlProfiler,
    PageSlot,
    PageTransferBuffer,
)

logger = logging.getLogger(__name__)


@dataclass
class L3Entry:
    node_id: int
    token_count: int
    page_count: int
    nbytes: int
    aligned_nbytes: int
    slots: List[PageSlot]
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
    async_submitted: int = 0
    async_completed: int = 0
    async_failed: int = 0
    async_stale: int = 0
    async_backpressure_skipped: int = 0
    async_pending: int = 0
    async_queue_latency_ms: float = 0.0
    async_snapshot_latency_ms: float = 0.0
    async_write_latency_ms: float = 0.0


@dataclass
class L3WriteOperation:
    sequence_id: int
    generation: int
    node_id: int
    node: TreeNode
    key_token_ids: tuple[int, ...]
    key_extra: Optional[str]
    value: torch.Tensor
    token_count: int
    reason: str
    slots: List[PageSlot]
    submitted_at: float
    activation_event: threading.Event
    active: bool = False
    canceled: bool = False


@dataclass
class L3WriteResult:
    operation: L3WriteOperation
    entry: Optional[L3Entry]
    error: Optional[str]
    started_at: float
    completed_at: float
    snapshot_latency_ms: float
    write_latency_ms: float


@dataclass
class L3RestoreNode:
    node: TreeNode
    entry: L3Entry
    dst_indices: torch.Tensor


@dataclass
class L3RestoreOperation:
    sequence_id: int
    generation: int
    terminal_node: TreeNode
    source_nodes: List[tuple[TreeNode, L3Entry]]
    token_count: int
    submitted_at: float
    waiter_rids: set[str] = field(default_factory=set)
    device_indices: Optional[torch.Tensor] = None
    restore_nodes: List[L3RestoreNode] = field(default_factory=list)
    state: str = "allocating"
    completion_lock_held: bool = False

    @property
    def path_key(self) -> tuple[int, ...]:
        return tuple(node.id for node, _entry in self.source_nodes)


@dataclass
class L3RestoreResult:
    operation: L3RestoreOperation
    error: Optional[str]
    started_at: float
    completed_at: float
    read_latency_ms: float
    refill_latency_ms: float


class _AsyncL3WriteBackend:
    """Single-worker write backend. Cache state is committed by the scheduler thread."""

    def __init__(
        self,
        execute: Callable[[L3WriteOperation], L3WriteResult],
        thread_name: str,
    ):
        self._execute = execute
        self._task_queue: queue.Queue[Optional[L3WriteOperation]] = queue.Queue()
        self._start_queue: queue.SimpleQueue[L3WriteOperation] = queue.SimpleQueue()
        # Results must never stall the I/O worker. The scheduler owns cache-state
        # mutation and drains this unbounded handoff on its normal event checks.
        self._result_queue: queue.SimpleQueue[L3WriteResult] = queue.SimpleQueue()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._accepting = True
        self._current_operation: Optional[L3WriteOperation] = None
        self._thread = threading.Thread(
            target=self._worker_loop, name=thread_name, daemon=True
        )
        self._thread.start()

    def submit(self, operation: L3WriteOperation) -> bool:
        with self._state_lock:
            if not self._accepting:
                return False
            self._task_queue.put_nowait(operation)
            return True

    def get_result(
        self, block: bool = False, timeout: Optional[float] = None
    ) -> Optional[L3WriteResult]:
        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_start_request(self) -> Optional[L3WriteOperation]:
        try:
            return self._start_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self) -> tuple[List[L3WriteOperation], List[L3WriteResult]]:
        with self._state_lock:
            self._accepting = False
            self._stop_event.set()
            if self._current_operation is not None:
                self._current_operation.canceled = True
                self._current_operation.activation_event.set()

        canceled = []
        while True:
            try:
                operation = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if operation is not None:
                canceled.append(operation)
        self._task_queue.put_nowait(None)

        results = []
        while self._thread.is_alive():
            result = self.get_result(block=True, timeout=0.05)
            if result is not None:
                results.append(result)
            self._thread.join(timeout=0.05)
        while True:
            result = self.get_result(block=False)
            if result is None:
                break
            results.append(result)
        return canceled, results

    def _worker_loop(self):
        while True:
            try:
                operation = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            if operation is None:
                return

            self._current_operation = operation
            self._start_queue.put(operation)
            while not operation.activation_event.wait(timeout=0.05):
                if self._stop_event.is_set():
                    return
            if operation.canceled or self._stop_event.is_set():
                self._current_operation = None
                if self._stop_event.is_set():
                    return
                now = time.monotonic()
                self._result_queue.put(
                    L3WriteResult(
                        operation=operation,
                        entry=None,
                        error="stale before snapshot activation",
                        started_at=now,
                        completed_at=now,
                        snapshot_latency_ms=0.0,
                        write_latency_ms=0.0,
                    )
                )
                continue

            result = self._execute(operation)
            self._result_queue.put(result)
            self._current_operation = None
            if self._stop_event.is_set():
                return


class _AsyncL3RestoreBackend:
    """Bounded restore workers; radix-tree mutation remains on the scheduler."""

    def __init__(
        self,
        execute: Callable[[L3RestoreOperation, int], L3RestoreResult],
        thread_name: str,
        worker_count: int,
    ):
        self._execute = execute
        self._task_queue: queue.Queue[Optional[L3RestoreOperation]] = queue.Queue()
        self._result_queue: queue.SimpleQueue[L3RestoreResult] = queue.SimpleQueue()
        self._state_lock = threading.Lock()
        self._accepting = True
        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"{thread_name}-{worker_id}",
                daemon=True,
            )
            for worker_id in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, operation: L3RestoreOperation) -> bool:
        with self._state_lock:
            if not self._accepting:
                return False
            self._task_queue.put_nowait(operation)
            return True

    def get_result(self) -> Optional[L3RestoreResult]:
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self) -> tuple[List[L3RestoreOperation], List[L3RestoreResult]]:
        with self._state_lock:
            self._accepting = False

        canceled = []
        while True:
            try:
                operation = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if operation is not None:
                canceled.append(operation)
        for _ in self._threads:
            self._task_queue.put_nowait(None)
        for thread in self._threads:
            thread.join()

        results = []
        while True:
            result = self.get_result()
            if result is None:
                break
            results.append(result)
        return canceled, results

    def _worker_loop(self, worker_id: int):
        while True:
            operation = self._task_queue.get()
            if operation is None:
                return
            self._result_queue.put(self._execute(operation, worker_id))


class UnifiedRadixCache(RadixCache):
    """A Jetson-oriented two-tier radix cache: unified DRAM plus L3 SSD.

    This baseline intentionally avoids modeling CPU host memory as a separate
    cache tier. Non-chunked MHA KV insertions are written through to SSD by a
    single-worker asynchronous backend. Writes are best effort under queue
    backpressure. Restores and partial L3 node splits remain synchronous.
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
        max_pending_writes: int = 8,
        debug: bool = False,
        profile_path: Optional[str] = None,
        tp_cache_group: Optional[torch.distributed.ProcessGroup] = None,
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
        if max_pending_writes <= 0:
            raise ValueError(
                "--unified-radix-cache-max-pending-writes must be greater than 0."
            )

        self.l3_base_dir = Path(l3_dir).expanduser().resolve()
        self.l3_run_dir = self.l3_base_dir / (
            f"run-{int(time.time())}-{os.getpid()}-tp{tp_rank}"
        )
        self.l3_budget_bytes = int(l3_budget_gb * (1024**3))
        self.l3_block_size = l3_block_size
        self.l3_entries: Dict[int, L3Entry] = {}
        self.stats = UnifiedRadixCacheStats()
        self.tp_rank = tp_rank
        self.tp_cache_group = tp_cache_group
        self.tp_world_size = (
            torch.distributed.get_world_size(group=tp_cache_group)
            if tp_cache_group is not None
            else 1
        )
        self.max_pending_writes = max_pending_writes
        self.debug = debug
        self.profiler = JsonlProfiler(profile_path)
        self._profile_throttle: Dict[tuple, int] = {}
        self._write_buffers = [
            PageTransferBuffer(self.kv_cache, page_size, l3_block_size)
            for _ in range(2)
        ]
        # SSD records stay page-granular, while refill kernels operate on a
        # small group of pages to amortize one launch per model layer. Two
        # group buffers preserve read/refill pipelining.
        self._restore_worker_count = 2
        self._restore_batch_pages = 8
        pin_restore_memory = torch.device(self.kv_cache.device).type == "cuda"
        self._restore_read_buffers = [
            AlignedPageBuffer(
                layer_num=self.kv_cache.layer_num,
                page_size=page_size,
                head_num=self.kv_cache.head_num,
                head_dim=self.kv_cache.head_dim,
                dtype=self.kv_cache.store_dtype,
                alignment=l3_block_size,
                pin_memory=pin_restore_memory,
            )
            for _ in range(self._restore_worker_count)
        ]
        self._restore_buffers = [
            [
                PageTransferBuffer(
                    self.kv_cache,
                    page_size * self._restore_batch_pages,
                    l3_block_size,
                    stream_priority=-1,
                )
                for _ in range(2)
            ]
            for _ in range(self._restore_worker_count)
        ]
        self.l3_page_data_bytes = self._write_buffers[0].data_bytes
        self.l3_page_record_bytes = self._write_buffers[0].record_bytes
        self._arena: Optional[DirectPageArena] = None
        self._write_generation = 0
        self._write_sequence = 0
        self._ongoing_writes: Dict[int, L3WriteOperation] = {}
        self._local_write_result: Optional[L3WriteResult] = None
        self._async_backend: Optional[_AsyncL3WriteBackend] = None
        self._restore_generation = 0
        self._restore_sequence = 0
        self._restore_backend: Optional[_AsyncL3RestoreBackend] = None
        self._restore_by_path: Dict[tuple[int, ...], L3RestoreOperation] = {}
        self._restore_by_rid: Dict[str, L3RestoreOperation] = {}
        self._completed_restore_rids: Dict[str, Optional[L3RestoreOperation]] = {}
        self._restoring_node_refs: Dict[int, int] = {}
        self._restore_demand = threading.Event()

        self.l3_run_dir.mkdir(parents=True, exist_ok=True)
        self._log_info(
            "UnifiedRadixCache enabled: l3_dir=%s, l3_run_dir=%s, "
            "l3_budget_bytes=%d, l3_budget_gb=%.3f, l3_block_size=%d, "
            "write_policy=async-write-through, max_pending_writes=%d",
            self.l3_base_dir,
            self.l3_run_dir,
            self.l3_budget_bytes,
            l3_budget_gb,
            self.l3_block_size,
            self.max_pending_writes,
        )

        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=False,
            eviction_policy=eviction_policy,
            is_eagle=False,
        )
        self.root_node.async_write_ref = 0
        self._start_async_backend()
        self._start_restore_backend()

    def profile_event(self, stage: str, **fields):
        if stage in {"scheduler_tick", "prefix_match", "restore_allocation_wait"}:
            now_ns = time.monotonic_ns()
            if stage == "scheduler_tick":
                key = (
                    stage,
                    fields.get("running_reqs"),
                    fields.get("waiting_reqs"),
                )
            elif stage == "prefix_match":
                key = (
                    stage,
                    fields.get("rid"),
                    fields.get("device_hit_tokens"),
                    fields.get("l3_hit_tokens"),
                )
            else:
                key = (stage, fields.get("operation_id"))
            if now_ns - self._profile_throttle.get(key, 0) < 100_000_000:
                return
            self._profile_throttle[key] = now_ns
        self.profiler.record(stage, **fields)

    def _log_info(self, msg, *args, **kwargs):
        if self.debug:
            logger.info(msg, *args, **kwargs)

    def reset(self):
        self._stop_restore_backend()
        self._stop_async_backend()
        self._close_arena()
        self._clear_l3_entries(drop_evicted=False)
        super().reset()
        self.root_node.async_write_ref = 0
        self._restore_demand.clear()
        self._open_arena()
        self._start_async_backend()
        self._start_restore_backend()

    def _open_arena(self):
        if self._arena is not None:
            return
        self._arena = DirectPageArena(
            self.l3_run_dir / "pages.arena",
            budget_bytes=self.l3_budget_bytes,
            record_bytes=self.l3_page_record_bytes,
            alignment=self.l3_block_size,
        )

    def _close_arena(self):
        if self._arena is None:
            return
        self._arena.close()
        self._arena = None

    def _start_async_backend(self):
        if self._async_backend is not None:
            return
        self._async_backend = _AsyncL3WriteBackend(
            execute=self._execute_async_write,
            thread_name=f"unified-l3-write-tp{self.tp_rank}",
        )

    def _start_restore_backend(self):
        if self._restore_backend is not None:
            return
        self._restore_backend = _AsyncL3RestoreBackend(
            execute=self._execute_async_restore,
            thread_name=f"unified-l3-restore-tp{self.tp_rank}",
            worker_count=self._restore_worker_count,
        )

    def _release_restore_operation(self, operation: L3RestoreOperation):
        for node, _entry in operation.source_nodes:
            count = self._restoring_node_refs.get(node.id, 0)
            if count <= 1:
                self._restoring_node_refs.pop(node.id, None)
            else:
                self._restoring_node_refs[node.id] = count - 1
        self._restore_by_path.pop(operation.path_key, None)
        for rid in list(operation.waiter_rids):
            if self._restore_by_rid.get(rid) is operation:
                self._restore_by_rid.pop(rid, None)

    def _free_restore_device_indices(self, operation: L3RestoreOperation):
        if operation.device_indices is None:
            return
        self.protected_size_ -= len(operation.device_indices)
        self.token_to_kv_pool_allocator.free(operation.device_indices)
        operation.device_indices = None

    def _stop_restore_backend(self):
        self._restore_generation += 1
        backend = self._restore_backend
        if backend is not None:
            backend.shutdown()
        for operation in list(self._restore_by_path.values()):
            self._free_restore_device_indices(operation)
            self._release_restore_operation(operation)
        completed_operations = {
            id(operation): operation
            for operation in self._completed_restore_rids.values()
            if operation is not None and operation.completion_lock_held
        }
        for operation in completed_operations.values():
            if self._is_node_attached(operation.terminal_node):
                self.dec_lock_ref(operation.terminal_node)
            operation.completion_lock_held = False
        self._restore_by_path.clear()
        self._restore_by_rid.clear()
        self._completed_restore_rids.clear()
        self._restoring_node_refs.clear()
        self._restore_backend = None

    def _stop_async_backend(self):
        self._write_generation += 1
        backend = self._async_backend
        results = []
        if self._local_write_result is not None:
            results.append(self._local_write_result)
            self._local_write_result = None
        if backend is not None:
            _, backend_results = backend.shutdown()
            results.extend(backend_results)

        for operation in list(self._ongoing_writes.values()):
            if self._arena is not None:
                self._arena.release(operation.slots)
            if operation.active and self._is_node_attached(operation.node):
                self._dec_async_write_ref(operation.node)

        self._ongoing_writes.clear()
        self.stats.async_pending = 0
        self._async_backend = None

    def _execute_async_write(self, operation: L3WriteOperation) -> L3WriteResult:
        started_at = time.monotonic()
        self.profile_event(
            "write_worker_start",
            operation_id=operation.sequence_id,
            node_id=operation.node_id,
            token_count=operation.token_count,
            queue_ms=max(0.0, (started_at - operation.submitted_at) * 1000),
        )
        entry = None
        error = None
        snapshot_latency_ms = 0.0
        write_latency_ms = 0.0
        try:
            worker_device = torch.device(self.device)
            if worker_device.type == "cuda":
                torch.cuda.set_device(
                    worker_device.index
                    if worker_device.index is not None
                    else torch.cuda.current_device()
                )
            entry, snapshot_latency_ms, write_latency_ms = (
                self._write_l3_entry_from_device(operation)
            )
            self.profile_event(
                "write_snapshot_complete",
                operation_id=operation.sequence_id,
                node_id=operation.node_id,
                latency_ms=snapshot_latency_ms,
            )
            self.profile_event(
                "write_io_complete",
                operation_id=operation.sequence_id,
                node_id=operation.node_id,
                latency_ms=write_latency_ms,
                success=entry is not None,
            )
            if entry is None:
                error = (
                    "preempted by latency-critical L3 restore"
                    if self._restore_demand.is_set()
                    else "snapshot serialization or SSD write failed"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "UnifiedRadixCache async L3 worker failed: node_id=%s, sequence_id=%s",
                operation.node_id,
                operation.sequence_id,
            )
        return L3WriteResult(
            operation=operation,
            entry=entry,
            error=error,
            started_at=started_at,
            completed_at=time.monotonic(),
            snapshot_latency_ms=snapshot_latency_ms,
            write_latency_ms=write_latency_ms,
        )

    def _execute_async_restore(
        self, operation: L3RestoreOperation, worker_id: int
    ) -> L3RestoreResult:
        started_at = time.monotonic()
        read_latency_ms = 0.0
        refill_latency_ms = 0.0
        error = None
        try:
            worker_device = torch.device(self.device)
            if worker_device.type == "cuda":
                torch.cuda.set_device(
                    worker_device.index
                    if worker_device.index is not None
                    else torch.cuda.current_device()
                )
            request_id = next(iter(operation.waiter_rids), None)
            for restore_node in operation.restore_nodes:
                read_ms, refill_ms = self._restore_l3_entry(
                    restore_node.entry,
                    restore_node.dst_indices,
                    request_id,
                    worker_id,
                )
                read_latency_ms += read_ms
                refill_latency_ms += refill_ms
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "UnifiedRadixCache async L3 restore failed: node_id=%s, "
                "sequence_id=%s",
                operation.terminal_node.id,
                operation.sequence_id,
            )
        return L3RestoreResult(
            operation=operation,
            error=error,
            started_at=started_at,
            completed_at=time.monotonic(),
            read_latency_ms=read_latency_ms,
            refill_latency_ms=refill_latency_ms,
        )

    def _schedule_async_restore(self, last_host_node: TreeNode, rid: str) -> bool:
        if self._restore_backend is None:
            return False
        if rid in self._restore_by_rid:
            return True

        source_nodes = []
        node = last_host_node
        while node.evicted and self._has_l3_entry(node):
            source_nodes.insert(0, (node, self._get_l3_entry(node)))
            node = node.parent
        if not source_nodes:
            return False

        path_key = tuple(source_node.id for source_node, _entry in source_nodes)
        existing = self._restore_by_path.get(path_key)
        if existing is not None:
            self._restore_demand.set()
            existing.waiter_rids.add(rid)
            self._restore_by_rid[rid] = existing
            self.profile_event(
                "restore_coalesced",
                rid=rid,
                operation_id=existing.sequence_id,
                node_id=last_host_node.id,
                waiter_count=len(existing.waiter_rids),
            )
            return True

        # Device KV allocated for a restore is protected until one waiter can
        # actually enter PrefillAdder. Bound unrelated paths to the worker
        # count so completed-but-not-staged restores cannot fill the pool;
        # later requests are rematched and same-path requests coalesce above.
        completed_protected = {
            id(operation)
            for operation in self._completed_restore_rids.values()
            if operation is not None and operation.completion_lock_held
        }
        if (
            len(self._restore_by_path) + len(completed_protected)
            >= self._restore_worker_count
        ):
            return False

        self._restore_sequence += 1
        operation = L3RestoreOperation(
            sequence_id=self._restore_sequence,
            generation=self._restore_generation,
            terminal_node=last_host_node,
            source_nodes=source_nodes,
            token_count=sum(len(source_node.key) for source_node, _ in source_nodes),
            submitted_at=time.monotonic(),
            waiter_rids={rid},
        )
        self._restore_by_path[path_key] = operation
        self._restore_by_rid[rid] = operation
        self._restore_demand.set()
        for source_node, _entry in source_nodes:
            self._restoring_node_refs[source_node.id] = (
                self._restoring_node_refs.get(source_node.id, 0) + 1
            )
        self.profile_event(
            "restore_scheduled",
            rid=rid,
            operation_id=operation.sequence_id,
            node_id=last_host_node.id,
            token_count=operation.token_count,
        )
        return True

    def _restore_operation_is_current(
        self, operation: L3RestoreOperation, require_evicted: bool = True
    ) -> bool:
        if (
            operation.generation != self._restore_generation
            or self._restore_by_path.get(operation.path_key) is not operation
        ):
            return False
        for node, entry in operation.source_nodes:
            if (
                not self._is_node_attached(node)
                or self._get_l3_entry(node) is not entry
                or (require_evicted and not node.evicted)
                or self._arena is None
                or any(not self._arena.is_current(slot) for slot in entry.slots)
            ):
                return False
        return True

    def _try_start_pending_restore(self) -> bool:
        if self._restore_backend is None:
            return False
        if (
            sum(op.state == "submitted" for op in self._restore_by_path.values())
            >= self._restore_worker_count
        ):
            return False
        operation = next(
            (op for op in self._restore_by_path.values() if op.state == "allocating"),
            None,
        )
        if operation is None:
            return False
        if not self._restore_operation_is_current(operation):
            self._finish_restore_without_commit(operation, "stale before allocation")
            return True

        ancestor_node = operation.source_nodes[0][0].parent
        self.inc_lock_ref(ancestor_node)
        device_indices = self.token_to_kv_pool_allocator.alloc(operation.token_count)
        if device_indices is None:
            self.evict(operation.token_count)
            device_indices = self.token_to_kv_pool_allocator.alloc(
                operation.token_count
            )
        self.dec_lock_ref(ancestor_node)
        if device_indices is None:
            self.profile_event(
                "restore_allocation_wait",
                operation_id=operation.sequence_id,
                node_id=operation.terminal_node.id,
                token_count=operation.token_count,
                available_tokens=self.token_to_kv_pool_allocator.available_size(),
                evictable_tokens=self.evictable_size_,
                protected_tokens=self.protected_size_,
                completed_waiters=len(self._completed_restore_rids),
            )
            return False

        offset = 0
        restore_nodes = []
        for node, entry in operation.source_nodes:
            token_count = len(node.key)
            restore_nodes.append(
                L3RestoreNode(
                    node=node,
                    entry=entry,
                    dst_indices=device_indices[offset : offset + token_count],
                )
            )
            offset += token_count
        operation.device_indices = device_indices
        self.protected_size_ += len(device_indices)
        operation.restore_nodes = restore_nodes
        operation.state = "submitted"
        if not self._restore_backend.submit(operation):
            operation.state = "allocating"
            operation.restore_nodes = []
            self._free_restore_device_indices(operation)
            return False

        self.profile_event(
            "restore_start",
            operation_id=operation.sequence_id,
            node_id=operation.terminal_node.id,
            token_count=operation.token_count,
            waiter_count=len(operation.waiter_rids),
            queue_ms=max(0.0, (time.monotonic() - operation.submitted_at) * 1000),
        )
        return True

    def _finish_restore_without_commit(self, operation: L3RestoreOperation, error: str):
        waiters = set(operation.waiter_rids)
        self._free_restore_device_indices(operation)
        self._release_restore_operation(operation)
        for rid in waiters:
            self._completed_restore_rids[rid] = None
        self.profile_event(
            "restore_discarded",
            operation_id=operation.sequence_id,
            node_id=operation.terminal_node.id,
            error=error,
        )

    def _drain_restore_results(self):
        if self._restore_backend is None:
            return
        while True:
            result = self._restore_backend.get_result()
            if result is None:
                return
            operation = result.operation
            current = self._restore_operation_is_current(operation)
            if result.error is not None or not current:
                self._finish_restore_without_commit(
                    operation, result.error or "stale restore result"
                )
                if result.error is not None and operation.source_nodes:
                    first_node = operation.source_nodes[0][0]
                    if self._is_node_attached(first_node):
                        self._drop_subtree(first_node, reason="stale-l3-read-failure")
                continue

            offset = 0
            restored_bytes = 0
            for restore_node in operation.restore_nodes:
                token_count = len(restore_node.node.key)
                restore_node.node.value = restore_node.dst_indices
                restore_node.entry.last_access_time = time.monotonic()
                self.evictable_size_ += token_count
                restored_bytes += restore_node.entry.nbytes
                offset += token_count

            waiters = set(operation.waiter_rids)
            self.protected_size_ -= operation.token_count
            self.inc_lock_ref(operation.terminal_node)
            operation.completion_lock_held = True
            operation.state = "committed"
            self._release_restore_operation(operation)
            for rid in waiters:
                self._completed_restore_rids[rid] = operation
            self.stats.read_count += 1
            self.stats.read_bytes += restored_bytes
            self.profile_event(
                "restore_complete",
                operation_id=operation.sequence_id,
                node_id=operation.terminal_node.id,
                token_count=operation.token_count,
                read_bytes=restored_bytes,
                read_ms=result.read_latency_ms,
                refill_ms=result.refill_latency_ms,
                latency_ms=(result.completed_at - operation.submitted_at) * 1000,
                waiter_count=len(waiters),
            )

    def _acknowledge_completed_restore(self, rid: str) -> bool:
        if rid in self._completed_restore_rids:
            operation = self._completed_restore_rids.pop(rid)
            if operation is not None:
                operation.waiter_rids.discard(rid)
                if operation.completion_lock_held:
                    self.dec_lock_ref(operation.terminal_node)
                    operation.completion_lock_held = False
                self.profile_event(
                    "restore_acknowledged",
                    rid=rid,
                    operation_id=operation.sequence_id,
                    remaining_waiters=len(operation.waiter_rids),
                )
            return True
        return False

    def check_restore_progress(self, rid: str) -> bool:
        if self._acknowledge_completed_restore(rid):
            return True
        return rid not in self._restore_by_rid

    def release_aborted_request(self, rid: str):
        if self._acknowledge_completed_restore(rid):
            return
        operation = self._restore_by_rid.pop(rid, None)
        if operation is None:
            return
        operation.waiter_rids.discard(rid)
        if operation.waiter_rids or operation.state == "submitted":
            return
        self._finish_restore_without_commit(operation, "all waiters aborted")

    def _get_async_write_ref(self, node: TreeNode) -> int:
        return getattr(node, "async_write_ref", 0)

    def _inc_async_write_ref(self, node: TreeNode):
        self.inc_lock_ref(node)
        current = node
        while current != self.root_node:
            current.async_write_ref = self._get_async_write_ref(current) + 1
            current = current.parent

    def _dec_async_write_ref(self, node: TreeNode):
        current = node
        while current != self.root_node:
            async_ref = self._get_async_write_ref(current)
            if async_ref <= 0:
                raise RuntimeError(
                    f"Invalid async write ref on node {current.id}: {async_ref}"
                )
            current.async_write_ref = async_ref - 1
            current = current.parent
        self.dec_lock_ref(node)

    def _has_external_lock(self, node: TreeNode) -> bool:
        return node.lock_ref > self._get_async_write_ref(node)

    def _is_node_attached(self, node: TreeNode) -> bool:
        current = node
        seen = set()
        while current != self.root_node:
            if current is None or current.id in seen or current.parent is None:
                return False
            seen.add(current.id)
            if not any(child is current for child in current.parent.children.values()):
                return False
            current = current.parent
        return True

    def _has_resident_descendant(self, node: TreeNode) -> bool:
        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            if not child.evicted:
                return True
            # Traverse through L3-only nodes as a defensive check for mixed-tier
            # holes left by an older implementation.
            stack.extend(child.children.values())
        return False

    def _submit_async_write(
        self,
        node: TreeNode,
        reason: str,
        allow_external_lock: bool = False,
    ) -> bool:
        if self._async_backend is None or node.value is None or node.evicted:
            return False
        if node.id in self._ongoing_writes:
            return False
        if not allow_external_lock and self._has_external_lock(node):
            self._log_info(
                "UnifiedRadixCache async L3 write skipped: node_id=%s, "
                "reason=%s, external_lock_ref=%d",
                node.id,
                reason,
                node.lock_ref - self._get_async_write_ref(node),
            )
            return False
        # max_pending_writes excludes the one active/result operation.
        if len(self._ongoing_writes) >= self.max_pending_writes + 1:
            self.stats.async_backpressure_skipped += 1
            self._log_info(
                "UnifiedRadixCache async L3 write skipped by backpressure: "
                "node_id=%s, reason=%s, outstanding=%d, max_pending=%d",
                node.id,
                reason,
                len(self._ongoing_writes),
                self.max_pending_writes,
            )
            return False

        if len(node.value) % self.page_size:
            self._log_info(
                "UnifiedRadixCache async L3 write skipped: node_id=%s, "
                "reason=%s, token_count=%d is not page aligned",
                node.id,
                reason,
                len(node.value),
            )
            return False
        slots = self._reserve_l3_slots(
            len(node.value) // self.page_size, protected_node_id=node.id
        )
        if slots is None:
            self.stats.async_backpressure_skipped += 1
            self.profile_event(
                "write_skipped_no_l3_slots",
                node_id=node.id,
                token_count=len(node.value),
            )
            return False

        self._write_sequence += 1
        operation = L3WriteOperation(
            sequence_id=self._write_sequence,
            generation=self._write_generation,
            node_id=node.id,
            node=node,
            key_token_ids=tuple(node.key.token_ids),
            key_extra=node.key.extra_key,
            value=node.value,
            token_count=len(node.value),
            reason=reason,
            slots=slots,
            submitted_at=time.monotonic(),
            activation_event=threading.Event(),
        )
        self._ongoing_writes[node.id] = operation
        if not self._async_backend.submit(operation):
            self._ongoing_writes.pop(node.id, None)
            self._arena.release(slots)
            return False

        self.stats.async_submitted += 1
        self.stats.async_pending = len(self._ongoing_writes)
        self._log_info(
            "UnifiedRadixCache async L3 write submitted: node_id=%s, "
            "sequence_id=%d, reason=%s, token_count=%d, outstanding=%d",
            node.id,
            operation.sequence_id,
            reason,
            operation.token_count,
            len(self._ongoing_writes),
        )
        self.profile_event(
            "write_submitted",
            operation_id=operation.sequence_id,
            node_id=node.id,
            token_count=operation.token_count,
            pending_writes=len(self._ongoing_writes),
        )
        return True

    def _activate_next_async_write(self) -> bool:
        """Let the worker snapshot one descriptor without locking queued nodes.

        Descriptors may wait for seconds behind SSD I/O. Holding a radix-cache
        lock for that entire queueing interval prevents DRAM eviction and turns
        best-effort write-through into request-path backpressure. The worker
        therefore asks the scheduler for activation only immediately before it
        snapshots KV. The scheduler revalidates the node and acquires the lock;
        stale descriptors are canceled without touching freed device storage.
        """
        if self._async_backend is None:
            return False
        operation = self._async_backend.get_start_request()
        if operation is None:
            return False

        node = operation.node
        current = (
            operation.generation == self._write_generation
            and self._ongoing_writes.get(operation.node_id) is operation
            and self._is_node_attached(node)
            and node.id == operation.node_id
            and tuple(node.key.token_ids) == operation.key_token_ids
            and node.key.extra_key == operation.key_extra
            and node.value is operation.value
            and not node.evicted
            and not self._has_l3_entry(node)
        )
        if current:
            self._inc_async_write_ref(node)
            operation.active = True
            self.profile_event(
                "write_activated",
                operation_id=operation.sequence_id,
                node_id=operation.node_id,
                queue_ms=max(0.0, (time.monotonic() - operation.submitted_at) * 1000),
            )
        else:
            operation.canceled = True
            self.profile_event(
                "write_activation_stale",
                operation_id=operation.sequence_id,
                node_id=operation.node_id,
            )
        operation.activation_event.set()
        return True

    def _tp_min(self, value: int) -> int:
        if self.tp_world_size <= 1:
            return value
        value_tensor = torch.tensor(value, dtype=torch.int, device="cpu")
        torch.distributed.all_reduce(
            value_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_cache_group,
        )
        return int(value_tensor.item())

    def _is_async_result_current(self, result: L3WriteResult) -> bool:
        operation = result.operation
        node = operation.node
        return (
            operation.active
            and operation.generation == self._write_generation
            and self._ongoing_writes.get(operation.node_id) is operation
            and self._is_node_attached(node)
            and node.id == operation.node_id
            and tuple(node.key.token_ids) == operation.key_token_ids
            and node.key.extra_key == operation.key_extra
            and node.value is operation.value
            and not self._has_l3_entry(node)
        )

    def _process_one_async_result(self, block: bool = False) -> bool:
        if self._async_backend is None or not self._ongoing_writes:
            return False
        if self._local_write_result is None:
            self._local_write_result = self._async_backend.get_result(
                block=block, timeout=0.05 if block else None
            )

        all_ready = self._tp_min(1 if self._local_write_result is not None else 0)
        if not all_ready:
            return False

        result = self._local_write_result
        operation = result.operation
        node = operation.node
        local_io_success = result.entry is not None and result.error is None
        all_io_success = self._tp_min(1 if local_io_success else 0)
        local_current = local_io_success and self._is_async_result_current(result)
        all_current = self._tp_min(1 if local_current else 0)

        committed = False
        if all_io_success and all_current:
            self._register_l3_entry(node, result.entry, operation.reason)
            committed = True

        if not committed and self._arena is not None:
            self._arena.release(operation.slots)

        self._ongoing_writes.pop(operation.node_id, None)
        if operation.active and self._is_node_attached(node):
            self._dec_async_write_ref(node)

        if committed:
            self.stats.async_completed += 1
        elif all_io_success and not all_current:
            self.stats.async_stale += 1
        else:
            self.stats.async_failed += 1

        self.stats.async_pending = len(self._ongoing_writes)
        self.stats.async_queue_latency_ms += max(
            0.0, (result.started_at - operation.submitted_at) * 1000
        )
        self.stats.async_snapshot_latency_ms += result.snapshot_latency_ms
        self.stats.async_write_latency_ms += result.write_latency_ms
        self._log_info(
            "UnifiedRadixCache async L3 write finished: node_id=%s, "
            "sequence_id=%d, committed=%s, stale=%s, error=%s, "
            "outstanding=%d, snapshot_ms=%.3f, write_ms=%.3f",
            operation.node_id,
            operation.sequence_id,
            committed,
            bool(all_io_success and not all_current),
            result.error,
            len(self._ongoing_writes),
            result.snapshot_latency_ms,
            result.write_latency_ms,
        )

        self._local_write_result = None
        self.profile_event(
            "write_committed" if committed else "write_discarded",
            operation_id=operation.sequence_id,
            node_id=operation.node_id,
            pending_writes=len(self._ongoing_writes),
            stale=bool(all_io_success and not all_current),
            error=result.error,
        )
        return True

    def _drain_async_results(self, block: bool = False):
        while True:
            if not self._process_one_async_result(block=block):
                break
            if block:
                break

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        match_start = time.perf_counter()
        request_id = kwargs.get("rid")
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

        value, last_node, l3_miss_reason = self._match_prefix_helper_l3(
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
            if request_id is not None:
                self._schedule_async_restore(last_l3_node, request_id)
            self._log_info(
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
            self._log_info(
                "UnifiedRadixCache L3 miss: last_node_id=%s, reason=%s, "
                "used_bytes=%d, hits=%d, misses=%d",
                last_node.id,
                l3_miss_reason,
                self.stats.used_bytes,
                self.stats.hit_count,
                self.stats.miss_count,
            )

        result = MatchResult(
            device_indices=value,
            last_device_node=device_node,
            last_host_node=last_l3_node,
            host_hit_length=l3_hit_length,
        )
        self.profile_event(
            "prefix_match",
            rid=request_id,
            device_hit_tokens=len(value),
            l3_hit_tokens=l3_hit_length,
            latency_ms=(time.perf_counter() - match_start) * 1000,
        )
        return result

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
        insert_path: List[TreeNode] = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)

            if prefix_len < len(child.key) and child.evicted:
                split_node, split_status = self._split_evicted_l3_node(
                    child, prefix_len, reason="insert"
                )
                if split_node is None:
                    if split_status in ("read-failure", "missing-entry"):
                        break
                    self._free_uninserted_value(value)
                    self._log_info(
                        "UnifiedRadixCache insert skipped after partial L3 split "
                        "failure: node_id=%s, status=%s, preserved_subtree=True",
                        child.id,
                        split_status,
                    )
                    if not chunked:
                        self._submit_insert_path(insert_path)
                    return total_prefix_length
                child = split_node

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
            insert_path.append(node)

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
            insert_path.append(new_node)

        if not chunked:
            self._submit_insert_path(insert_path)

        return total_prefix_length

    def _submit_insert_path(self, insert_path: List[TreeNode]) -> None:
        # The path is collected by insert in root-to-leaf order. Reuse it to
        # avoid a second radix-tree traversal after the insertion completes.
        for node in insert_path:
            if (
                node.parent is not None
                and node.parent.children.get(self.get_child_key_fn(node.key)) is node
                and not node.evicted
                and not self._has_l3_entry(node)
            ):
                self._submit_async_write(
                    node,
                    reason="insert-trigger",
                    allow_external_lock=True,
                )

    def evict(self, num_tokens: int):
        if self.disable:
            return

        # Make already completed insert-trigger writes visible, but never wait
        # for or submit L3 I/O from the memory-pressure path.
        self._drain_async_results(block=False)

        leaves = self._collect_leaves_device()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node.id, node)
            for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, _node_id, node = heapq.heappop(eviction_heap)
            if node == self.root_node or node.lock_ref > 0 or node.evicted:
                continue

            parent = node.parent
            if self._has_l3_entry(node):
                num_evicted += self._release_dram_copy(node, reason="memory-pressure")
            else:
                token_count = len(node.value)
                self._record_remove_event(node)
                self._drop_subtree(node, reason="unbacked-memory-pressure")
                num_evicted += token_count

            if (
                parent is not None
                and parent != self.root_node
                and not parent.evicted
                and parent.lock_ref == 0
                and all(child.evicted for child in parent.children.values())
            ):
                heapq.heappush(
                    eviction_heap,
                    (
                        self.eviction_strategy.get_priority(parent),
                        parent.id,
                        parent,
                    ),
                )

        self._log_info(
            "UnifiedRadixCache pressure eviction finished: policy=write-through, "
            "requested_tokens=%d, freed_tokens=%d",
            num_tokens,
            num_evicted,
        )

    def init_load_back(
        self,
        last_host_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        if request_id is not None and request_id in self._restore_by_rid:
            raise RuntimeError(
                "UnifiedRadixCache restore is still pending; the scheduler must "
                "wait for check_restore_progress() before staging the request."
            )
        if host_hit_length <= 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_host_node,
            )

        start_time = time.perf_counter()
        self.profile_event(
            "restore_start",
            rid=request_id,
            node_id=last_host_node.id,
            host_hit_tokens=host_hit_length,
        )
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
            self._log_info(
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

        self.profile_event(
            "restore_allocated",
            rid=request_id,
            node_id=last_host_node.id,
            token_count=total_tokens,
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
                read_ms, refill_ms = self._restore_l3_entry(
                    entry, dst_indices, request_id, worker_id=0
                )
                self.profile_event(
                    "restore_read_complete",
                    rid=request_id,
                    node_id=l3_node.id,
                    token_count=token_count,
                    read_bytes=entry.nbytes,
                    latency_ms=read_ms,
                )
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
            self.profile_event(
                "restore_refill_complete",
                rid=request_id,
                node_id=l3_node.id,
                token_count=token_count,
                latency_ms=refill_ms,
            )
            l3_node.value = dst_indices
            self.evictable_size_ += token_count
            restored_nodes.append((l3_node, token_count))
            restored_bytes += entry.nbytes
            offset += token_count

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.profile_event(
            "restore_complete",
            rid=request_id,
            node_id=last_hit_node.id,
            token_count=total_tokens,
            read_bytes=restored_bytes,
            latency_ms=latency_ms,
        )
        self.stats.read_count += 1
        self.stats.read_bytes += restored_bytes
        self._log_info(
            "UnifiedRadixCache L3 read: node_id=%s, token_count=%d, page_count=%d, "
            "read_bytes=%d, total_read_count=%d, total_read_bytes=%d",
            last_hit_node.id,
            total_tokens,
            self._page_count(total_tokens),
            restored_bytes,
            self.stats.read_count,
            self.stats.read_bytes,
        )
        self._log_info(
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

    def check_hicache_events(self, allow_restore_allocation: bool = True):
        self._drain_restore_results()
        # A chunked prefill owns an implicit reservation: its next chunk must
        # be admitted after the previous chunk is cached and released. Do not
        # let a newly allocated restore consume that reservation between two
        # chunks. In-flight restores are still drained above.
        if allow_restore_allocation:
            self._try_start_pending_restore()
        # Restore work has strict priority: writes are best effort, while a
        # restore directly gates requests waiting for their first token.
        if not self._restore_by_path and not self._restore_demand.is_set():
            self._activate_next_async_write()
        self._drain_async_results(block=False)
        if not self._restore_by_path and not any(
            operation is not None and operation.completion_lock_held
            for operation in self._completed_restore_rids.values()
        ):
            self._restore_demand.clear()
        return None

    def clear_storage_backend(self) -> bool:
        self._stop_restore_backend()
        self._stop_async_backend()
        self._close_arena()
        self._clear_l3_entries(drop_evicted=True)
        self._restore_demand.clear()
        self._open_arena()
        self._start_async_backend()
        self._start_restore_backend()
        return True

    def _match_prefix_helper_l3(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []
        l3_miss_reason = "no-l3-prefix"

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                if child.evicted:
                    split_node, split_status = self._split_evicted_l3_node(
                        child, prefix_len, reason="match"
                    )
                    if split_node is not None:
                        node = split_node
                        l3_miss_reason = "partial-l3-split"
                    else:
                        l3_miss_reason = f"partial-l3-{split_status}"
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

        return value, node, l3_miss_reason

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        if child.evicted:
            split_node, split_status = self._split_evicted_l3_node(
                child, split_len, reason="direct"
            )
            if split_node is None:
                raise RuntimeError(
                    "UnifiedRadixCache failed to split evicted L3 node: "
                    f"node_id={child.id}, status={split_status}"
                )
            return split_node
        self._delete_l3_entry(child)
        new_node = super()._split_node(key, child, split_len)
        new_node.async_write_ref = self._get_async_write_ref(child)
        return new_node

    def _split_evicted_l3_node(
        self, child: TreeNode, split_len: int, reason: str
    ) -> tuple[Optional[TreeNode], str]:
        if not child.evicted:
            return child, "not-evicted"
        if split_len <= 0 or split_len >= len(child.key):
            return None, "invalid-split"
        if child.id in self._restoring_node_refs:
            return None, "restore-inflight"

        old_entry = self._get_l3_entry(child)
        if old_entry is None:
            logger.warning(
                "UnifiedRadixCache partial L3 split found evicted node without "
                "L3 entry; dropping unrecoverable subtree: node_id=%s, reason=%s",
                child.id,
                reason,
            )
            self._drop_subtree(child, reason="missing-l3-entry")
            return None, "missing-entry"
        if split_len % self.page_size:
            return None, "unaligned-split"
        split_pages = split_len // self.page_size
        if split_pages <= 0 or split_pages >= len(old_entry.slots):
            return None, "invalid-page-split"

        old_key = child.key
        parent = child.parent
        new_node = TreeNode()
        new_node.parent = parent
        new_node.lock_ref = child.lock_ref
        new_node.async_write_ref = self._get_async_write_ref(child)
        new_node.key = old_key[:split_len]
        new_node.value = None

        now = time.monotonic()
        prefix_slots = old_entry.slots[:split_pages]
        tail_slots = old_entry.slots[split_pages:]
        prefix_entry = L3Entry(
            node_id=new_node.id,
            token_count=split_len,
            page_count=len(prefix_slots),
            nbytes=len(prefix_slots) * self.l3_page_data_bytes,
            aligned_nbytes=len(prefix_slots) * self.l3_page_record_bytes,
            slots=prefix_slots,
            created_at=old_entry.created_at,
            last_access_time=now,
            node=new_node,
        )
        tail_entry = L3Entry(
            node_id=child.id,
            token_count=len(child.key) - split_len,
            page_count=len(tail_slots),
            nbytes=len(tail_slots) * self.l3_page_data_bytes,
            aligned_nbytes=len(tail_slots) * self.l3_page_record_bytes,
            slots=tail_slots,
            created_at=old_entry.created_at,
            last_access_time=now,
            node=child,
        )

        self._record_remove_event(child)
        self.l3_entries.pop(old_entry.node_id, None)
        child.l3_entry = None
        self.stats.used_bytes = max(0, self.stats.used_bytes - old_entry.nbytes)

        new_node.children = {self.get_child_key_fn(old_key[split_len:]): child}
        child.parent = new_node
        child.key = old_key[split_len:]
        child.value = None
        parent.children[self.get_child_key_fn(old_key)] = new_node

        self._register_l3_entry(
            new_node, prefix_entry, reason="partial-l3-split", count_write=False
        )
        self._register_l3_entry(
            child, tail_entry, reason="partial-l3-split", count_write=False
        )
        self._record_store_event(new_node)
        self._record_store_event(child)

        self._log_info(
            "UnifiedRadixCache partial-l3-split: old_node_id=%s, "
            "prefix_node_id=%s, tail_node_id=%s, split_tokens=%d, "
            "tail_tokens=%d, child_count=%d, reason=%s, used_bytes=%d",
            old_entry.node_id,
            new_node.id,
            child.id,
            split_len,
            len(child.key),
            len(child.children),
            reason,
            self.stats.used_bytes,
        )
        return new_node, "success"

    def _release_dram_copy(self, node: TreeNode, reason: str) -> int:
        if node.value is None:
            return 0
        if node.lock_ref != 0:
            self._log_info(
                "UnifiedRadixCache DRAM release skipped: node_id=%s, "
                "reason=%s, lock_ref=%d",
                node.id,
                reason,
                node.lock_ref,
            )
            return 0
        if self._has_resident_descendant(node):
            logger.warning(
                "UnifiedRadixCache DRAM release skipped: node_id=%s, "
                "reason=%s, resident_descendant=True",
                node.id,
                reason,
            )
            return 0
        token_count = len(node.value)
        self.token_to_kv_pool_allocator.free(node.value)
        node.value = None
        self.evictable_size_ -= token_count
        self._record_remove_event(node)
        self._log_info(
            "UnifiedRadixCache DRAM copy released: node_id=%s, reason=%s, "
            "token_count=%d",
            node.id,
            reason,
            token_count,
        )
        return token_count

    def _write_l3_entry_from_device(
        self, operation: L3WriteOperation
    ) -> tuple[Optional[L3Entry], float, float]:
        if self._arena is None:
            raise RuntimeError("L3 page arena is not open")
        if operation.token_count % self.page_size:
            raise RuntimeError(
                f"L3 write is not page aligned: tokens={operation.token_count}"
            )
        if len(operation.slots) != operation.token_count // self.page_size:
            raise RuntimeError("L3 slot count does not match the write operation")

        snapshot_ms = 0.0
        write_ms = 0.0
        inflight = []
        next_page = 0

        def launch(page_index: int, buffer_index: int):
            token_start = page_index * self.page_size
            event = self._write_buffers[buffer_index].snapshot(
                operation.value[token_start : token_start + self.page_size]
            )
            return [page_index, buffer_index, event, time.perf_counter()]

        def preempt_for_restore():
            # A second snapshot may already be using the same device indices.
            # Drain it before the scheduler releases the operation's node lock.
            for _page, _buffer, pending_event, _started in inflight:
                pending_event.synchronize()
            return None, snapshot_ms, write_ms

        while next_page < min(2, len(operation.slots)):
            if self._restore_demand.is_set():
                return preempt_for_restore()
            inflight.append(launch(next_page, next_page))
            next_page += 1

        while inflight:
            page_index, buffer_index, event, snapshot_start = inflight.pop(0)
            event.synchronize()
            snapshot_ms += (time.perf_counter() - snapshot_start) * 1000

            if self._restore_demand.is_set():
                return preempt_for_restore()

            write_start = time.perf_counter()
            self._arena.write(
                operation.slots[page_index], self._write_buffers[buffer_index]
            )
            write_ms += (time.perf_counter() - write_start) * 1000

            if next_page < len(operation.slots):
                inflight.append(launch(next_page, buffer_index))
                next_page += 1

        now = time.monotonic()
        page_count = len(operation.slots)
        return (
            L3Entry(
                node_id=operation.node_id,
                token_count=operation.token_count,
                page_count=page_count,
                nbytes=page_count * self.l3_page_data_bytes,
                aligned_nbytes=page_count * self.l3_page_record_bytes,
                slots=list(operation.slots),
                created_at=now,
                last_access_time=now,
                node=operation.node,
            ),
            snapshot_ms,
            write_ms,
        )

    def _restore_l3_entry(
        self,
        entry: L3Entry,
        dst_indices: torch.Tensor,
        request_id: Optional[str],
        worker_id: int,
    ) -> tuple[float, float]:
        if self._arena is None:
            raise RuntimeError("L3 page arena is not open")
        if len(dst_indices) != entry.token_count:
            raise RuntimeError(
                f"L3 restore size mismatch: tokens={entry.token_count}, "
                f"indices={len(dst_indices)}"
            )

        read_ms = 0.0
        refill_ms = 0.0
        restore_buffers = self._restore_buffers[worker_id]
        read_buffer = self._restore_read_buffers[worker_id]
        refill_events = [None, None]
        refill_started = [0.0, 0.0]
        try:
            for group_index, page_start in enumerate(
                range(0, len(entry.slots), self._restore_batch_pages)
            ):
                buffer_index = group_index % 2
                previous_event = refill_events[buffer_index]
                if previous_event is not None:
                    previous_event.synchronize()
                    refill_ms += (
                        time.perf_counter() - refill_started[buffer_index]
                    ) * 1000

                group_slots = entry.slots[
                    page_start : page_start + self._restore_batch_pages
                ]
                group_buffer = restore_buffers[buffer_index]
                for page_offset, slot in enumerate(group_slots):
                    read_start = time.perf_counter()
                    self._arena.read(slot, read_buffer)
                    token_offset = page_offset * self.page_size
                    group_buffer.tensor[
                        :, token_offset : token_offset + self.page_size
                    ].copy_(read_buffer.tensor)
                    page_read_ms = (time.perf_counter() - read_start) * 1000
                    read_ms += page_read_ms
                    self.profile_event(
                        "restore_page_read",
                        rid=request_id,
                        node_id=entry.node_id,
                        page_index=page_start + page_offset,
                        slot=slot.index,
                        latency_ms=page_read_ms,
                    )

                token_start = page_start * self.page_size
                token_count = len(group_slots) * self.page_size
                refill_started[buffer_index] = time.perf_counter()
                refill_events[buffer_index] = group_buffer.refill(
                    dst_indices[token_start : token_start + token_count]
                )

            for buffer_index, event in enumerate(refill_events):
                if event is not None:
                    event.synchronize()
                    refill_ms += (
                        time.perf_counter() - refill_started[buffer_index]
                    ) * 1000
        except Exception:
            for event in refill_events:
                if event is not None:
                    event.synchronize()
            raise
        return read_ms, refill_ms

    def _register_l3_entry(
        self,
        node: TreeNode,
        entry: L3Entry,
        reason: str,
        count_write: bool = True,
    ):
        entry.node_id = node.id
        entry.node = node
        node.l3_entry = entry
        self.l3_entries[node.id] = entry
        self.stats.used_bytes += entry.nbytes
        if count_write:
            self.stats.write_count += 1
            self.stats.write_bytes += entry.nbytes
        self._log_info(
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

    def _free_uninserted_value(self, value):
        if (
            value is not None
            and isinstance(value, torch.Tensor)
            and value.numel() > 0
            and self.token_to_kv_pool_allocator is not None
        ):
            self.token_to_kv_pool_allocator.free(value)

    def _reserve_l3_slots(
        self, page_count: int, protected_node_id: int
    ) -> Optional[List[PageSlot]]:
        if self._arena is None:
            return None
        slots = self._arena.reserve(page_count)
        while slots is None and self.l3_entries:
            candidates = [
                entry
                for node_id, entry in self.l3_entries.items()
                if node_id != protected_node_id
                and node_id not in self._restoring_node_refs
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda entry: entry.last_access_time)
            self._evict_l3_entry(victim, reason="budget")
            slots = self._arena.reserve(page_count)
        return slots

    def _evict_l3_entry(self, entry: L3Entry, reason: str):
        node = entry.node
        evicted_bytes = self._delete_l3_entry(node)
        self.stats.eviction_count += 1
        self.stats.eviction_bytes += evicted_bytes
        if node.value is None and node != self.root_node:
            self._drop_subtree(node, reason=f"l3-{reason}-eviction")
        self._log_info(
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
        if self._arena is not None:
            self._arena.release(entry.slots)
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
        self._log_info(
            "UnifiedRadixCache dropped radix subtree: node_id=%s, reason=%s",
            node.id,
            reason,
        )

    def _clear_l3_entries(self, drop_evicted: bool = False):
        if hasattr(self, "root_node"):
            stack = list(self.root_node.children.values())
            while stack:
                node = stack.pop()
                if drop_evicted and node.evicted:
                    self._drop_subtree(node, reason="clear-l3-storage")
                    continue
                stack.extend(list(node.children.values()))
                node.l3_entry = None
        if hasattr(self, "l3_entries"):
            if self._arena is not None:
                for entry in self.l3_entries.values():
                    current_slots = [
                        slot for slot in entry.slots if self._arena.is_current(slot)
                    ]
                    if current_slots:
                        self._arena.release(current_slots)
            self.l3_entries.clear()
        if hasattr(self, "stats"):
            self.stats.used_bytes = 0
        if hasattr(self, "l3_run_dir"):
            shutil.rmtree(self.l3_run_dir, ignore_errors=True)
            self.l3_run_dir.mkdir(parents=True, exist_ok=True)
            self._log_info(
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

    def _page_count(self, token_count: int) -> int:
        return (token_count + self.page_size - 1) // self.page_size
