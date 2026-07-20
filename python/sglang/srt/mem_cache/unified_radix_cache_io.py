from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class PageSlot:
    index: int
    generation: int


class AlignedPageBuffer:
    """Pinned, block-aligned storage buffer with a typed page-first KV view."""

    def __init__(
        self,
        *,
        layer_num: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        alignment: int,
        pin_memory: bool,
    ):
        self.shape = (2, page_size, layer_num, head_num, head_dim)
        self.dtype = dtype
        self.data_bytes = (
            2 * page_size * layer_num * head_num * head_dim * dtype.itemsize
        )
        self.record_bytes = align_up(self.data_bytes, alignment)

        allocation = torch.empty(
            self.record_bytes + alignment,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        offset = (-allocation.data_ptr()) % alignment
        self._allocation = allocation
        self.raw = allocation[offset : offset + self.record_bytes]
        if self.raw.data_ptr() % alignment:
            raise RuntimeError("failed to allocate an aligned page buffer")
        self.raw.zero_()
        self.tensor = self.raw[: self.data_bytes].view(dtype).view(self.shape)

    def as_buffer(self) -> memoryview:
        return memoryview(self.raw.numpy())


class DirectPageArena:
    """Single-file fixed-slot O_DIRECT arena.

    The arena owns only physical slot allocation. Radix/LRU policy stays in
    UnifiedRadixCache, which makes stale completions easy to reject using slot
    generations without coupling filesystem I/O to tree mutations.
    """

    def __init__(
        self,
        path: Path,
        *,
        budget_bytes: int,
        record_bytes: int,
        alignment: int,
    ):
        if record_bytes <= 0 or record_bytes % alignment:
            raise ValueError("record_bytes must be a positive alignment multiple")
        self.path = path
        self.record_bytes = record_bytes
        self.alignment = alignment
        self.capacity = budget_bytes // record_bytes
        if self.capacity <= 0:
            raise ValueError(
                f"L3 budget {budget_bytes} cannot hold one {record_bytes}-byte page"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        direct_flag = getattr(os, "O_DIRECT", None)
        if direct_flag is None:
            raise RuntimeError("O_DIRECT is unavailable on this platform")
        self._fd = os.open(
            path,
            os.O_CREAT | os.O_RDWR | os.O_TRUNC | direct_flag,
            0o600,
        )
        os.ftruncate(self._fd, self.capacity * record_bytes)
        self._lock = threading.Lock()
        self._generations = [0] * self.capacity
        self._allocated = [False] * self.capacity
        self._free = list(range(self.capacity - 1, -1, -1))
        self._closed = False

    @property
    def used_slots(self) -> int:
        with self._lock:
            return self.capacity - len(self._free)

    def reserve(self, count: int) -> Optional[list[PageSlot]]:
        if count <= 0:
            return []
        with self._lock:
            if count > len(self._free):
                return None
            refs = []
            for _ in range(count):
                index = self._free.pop()
                self._generations[index] += 1
                self._allocated[index] = True
                refs.append(PageSlot(index, self._generations[index]))
            return refs

    def release(self, slots: Iterable[PageSlot]) -> None:
        with self._lock:
            for slot in slots:
                self._validate_locked(slot)
                self._allocated[slot.index] = False
                self._free.append(slot.index)

    def is_current(self, slot: PageSlot) -> bool:
        with self._lock:
            return (
                0 <= slot.index < self.capacity
                and self._allocated[slot.index]
                and self._generations[slot.index] == slot.generation
            )

    def write(self, slot: PageSlot, buffer: AlignedPageBuffer) -> None:
        self._validate_buffer(buffer)
        with self._lock:
            self._validate_locked(slot)
        written = os.pwritev(
            self._fd,
            [buffer.as_buffer()],
            slot.index * self.record_bytes,
        )
        if written != self.record_bytes:
            raise RuntimeError(
                f"short O_DIRECT write for slot {slot.index}: "
                f"expected {self.record_bytes}, got {written}"
            )

    def read(self, slot: PageSlot, buffer: AlignedPageBuffer) -> None:
        self._validate_buffer(buffer)
        with self._lock:
            self._validate_locked(slot)
        read_bytes = os.preadv(
            self._fd,
            [buffer.as_buffer()],
            slot.index * self.record_bytes,
        )
        if read_bytes != self.record_bytes:
            raise RuntimeError(
                f"short O_DIRECT read for slot {slot.index}: "
                f"expected {self.record_bytes}, got {read_bytes}"
            )

    def close(self, unlink: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)
        if unlink:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _validate_buffer(self, buffer: AlignedPageBuffer) -> None:
        if buffer.record_bytes != self.record_bytes:
            raise ValueError(
                f"buffer record size {buffer.record_bytes} != {self.record_bytes}"
            )
        if buffer.raw.data_ptr() % self.alignment:
            raise ValueError("buffer address is not O_DIRECT aligned")

    def _validate_locked(self, slot: PageSlot) -> None:
        if not 0 <= slot.index < self.capacity:
            raise RuntimeError(f"invalid L3 slot index {slot.index}")
        if not self._allocated[slot.index]:
            raise RuntimeError(f"L3 slot {slot.index} is not allocated")
        generation = self._generations[slot.index]
        if generation != slot.generation:
            raise RuntimeError(
                f"stale L3 slot {slot.index}: expected generation {generation}, "
                f"got {slot.generation}"
            )


class _ImmediateEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class PageTransferBuffer(AlignedPageBuffer):
    """One page-first staging buffer and its dedicated CUDA stream."""

    def __init__(
        self, kv_cache, page_size: int, alignment: int, stream_priority: int = 0
    ):
        self.kv_cache = kv_cache
        self.page_size = page_size
        self.device = torch.device(kv_cache.device)
        super().__init__(
            layer_num=kv_cache.layer_num,
            page_size=page_size,
            head_num=kv_cache.head_num,
            head_dim=kv_cache.head_dim,
            dtype=kv_cache.store_dtype,
            alignment=alignment,
            pin_memory=self.device.type == "cuda",
        )
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(
                device=self.device, priority=stream_priority
            )
            self.staging_indices = torch.arange(
                page_size, dtype=torch.int64, device=self.device
            )
        else:
            self.stream = None
            self.staging_indices = torch.arange(page_size, dtype=torch.int64)

    def snapshot(self, device_indices: torch.Tensor):
        if len(device_indices) != self.page_size:
            raise ValueError("snapshot requires exactly one full page")
        if self.device.type != "cuda":
            for layer_id in range(self.kv_cache.layer_num):
                self.tensor[0, :, layer_id].copy_(
                    self.kv_cache.k_buffer[layer_id][device_indices]
                )
                self.tensor[1, :, layer_id].copy_(
                    self.kv_cache.v_buffer[layer_id][device_indices]
                )
            return _ImmediateEvent()

        from sgl_kernel.kvcacheio import transfer_kv_all_layer_lf_pf

        event = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            indices = device_indices.to(self.device, non_blocking=True)
            transfer_kv_all_layer_lf_pf(
                src_k_layers=self.kv_cache.k_data_ptrs,
                dst_k=self.tensor[0],
                src_v_layers=self.kv_cache.v_data_ptrs,
                dst_v=self.tensor[1],
                src_indices=indices,
                dst_indices=self.staging_indices,
                item_size=self.kv_cache.head_num
                * self.kv_cache.head_dim
                * self.kv_cache.store_dtype.itemsize,
                dst_layout_dim=self.kv_cache.head_num
                * self.kv_cache.head_dim
                * self.kv_cache.store_dtype.itemsize
                * self.kv_cache.layer_num,
                num_layers=self.kv_cache.layer_num,
            )
            event.record(self.stream)
            if indices.is_cuda:
                indices.record_stream(self.stream)
        return event

    def refill(self, device_indices: torch.Tensor):
        token_count = len(device_indices)
        if token_count <= 0 or token_count > self.page_size:
            raise ValueError(
                "refill requires between one token and the staging-buffer size"
            )
        if self.device.type != "cuda":
            for layer_id in range(self.kv_cache.layer_num):
                self.kv_cache.k_buffer[layer_id][device_indices] = self.tensor[
                    0, :token_count, layer_id
                ]
                self.kv_cache.v_buffer[layer_id][device_indices] = self.tensor[
                    1, :token_count, layer_id
                ]
            return _ImmediateEvent()

        from sgl_kernel.kvcacheio import transfer_kv_per_layer_pf_lf

        event = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            indices = device_indices.to(self.device, non_blocking=True)
            staging_indices = self.staging_indices[:token_count]
            for layer_id in range(self.kv_cache.layer_num):
                transfer_kv_per_layer_pf_lf(
                    src_k=self.tensor[0],
                    dst_k=self.kv_cache.k_buffer[layer_id],
                    src_v=self.tensor[1],
                    dst_v=self.kv_cache.v_buffer[layer_id],
                    src_indices=staging_indices,
                    dst_indices=indices,
                    layer_id=layer_id,
                    item_size=self.kv_cache.head_num
                    * self.kv_cache.head_dim
                    * self.kv_cache.store_dtype.itemsize,
                    src_layout_dim=self.kv_cache.head_num
                    * self.kv_cache.head_dim
                    * self.kv_cache.store_dtype.itemsize
                    * self.kv_cache.layer_num,
                )
            event.record(self.stream)
            if indices.is_cuda:
                indices.record_stream(self.stream)
        return event


class JsonlProfiler:
    """Best-effort structured event sink for UnifiedRadixCache diagnostics.

    Events are serialized by a daemon thread so cache and scheduler paths never
    perform file I/O. A full queue drops events instead of delaying inference.
    """

    _STOP = object()

    def __init__(self, path: Optional[str], max_events: int = 8192):
        self.path = Path(path).expanduser().resolve() if path else None
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self.dropped_events = 0

        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=max_events)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="unified-radix-profile-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._queue is not None and not self._closed

    def record(self, stage: str, **fields: Any) -> None:
        event_queue = self._queue
        if event_queue is None or self._closed:
            return

        event = {
            "schema_version": "unified_radix_cache.profile.v1",
            "monotonic_ns": time.monotonic_ns(),
            "stage": stage,
            **fields,
        }
        try:
            event_queue.put_nowait(event)
        except queue.Full:
            self.dropped_events += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        event_queue = self._queue
        thread = self._thread
        if event_queue is None or thread is None:
            return
        event_queue.put(self._STOP)
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warning(
                "UnifiedRadixCache profile writer did not stop cleanly: path=%s",
                self.path,
            )

    def _writer_loop(self) -> None:
        assert self.path is not None
        assert self._queue is not None
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as output:
                while True:
                    event = self._queue.get()
                    if event is self._STOP:
                        return
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
        except OSError:
            logger.exception(
                "UnifiedRadixCache profile writer failed: path=%s", self.path
            )
