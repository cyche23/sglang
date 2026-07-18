"""
Unit tests for UnifiedRadixCache L3 radix split behavior.

These tests use a minimal MHA-compatible KV pool and allocator so the cache
exercises real L3 file serialization without requiring a model server.
"""

import os
import tempfile
import threading
import time
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class FakeMHAKVPool(MHATokenToKVPool):
    def __init__(self, layer_num=2, cpu_offloading_chunk_size=4):
        self.layer_num = layer_num
        self.cpu_offloading_chunk_size = cpu_offloading_chunk_size


class FakeAllocator:
    def __init__(self, size=128, layer_num=2, chunk_size=4):
        self.device = torch.device("cpu")
        self._kvcache = FakeMHAKVPool(
            layer_num=layer_num, cpu_offloading_chunk_size=chunk_size
        )
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.k_buffers = [
            torch.zeros((size + 1, 1, 1), dtype=torch.float32) for _ in range(layer_num)
        ]
        self.v_buffers = [
            torch.zeros((size + 1, 1, 1), dtype=torch.float32) for _ in range(layer_num)
        ]
        self.freed = []

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if need_size > len(self.free_pages):
            return None
        indices = self.free_pages[:need_size].clone()
        self.free_pages = self.free_pages[need_size:]
        return indices

    def free(self, free_index: torch.Tensor):
        if free_index is not None and free_index.numel() > 0:
            self.freed.append(free_index.clone())

    def fill(self, indices: torch.Tensor, base: int):
        positions = torch.arange(len(indices), dtype=torch.float32)
        for layer_id in range(self._kvcache.layer_num):
            self.k_buffers[layer_id][indices, 0, 0] = base + layer_id * 100 + positions
            self.v_buffers[layer_id][indices, 0, 0] = (
                base + layer_id * 100 + 1000 + positions
            )

    def get_cpu_copy(self, indices: torch.Tensor):
        kv_cpu = []
        chunk_size = self._kvcache.cpu_offloading_chunk_size
        for layer_id in range(self._kvcache.layer_num):
            layer_chunks = []
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                layer_chunks.append(
                    [
                        self.k_buffers[layer_id][chunk_indices].clone(),
                        self.v_buffers[layer_id][chunk_indices].clone(),
                    ]
                )
            kv_cpu.append(layer_chunks)
        return kv_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices: torch.Tensor):
        chunk_size = self._kvcache.cpu_offloading_chunk_size
        for layer_id in range(self._kvcache.layer_num):
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                chunk_id = start // chunk_size
                k_cpu, v_cpu = kv_cache_cpu[layer_id][chunk_id]
                assert k_cpu.shape[0] == len(chunk_indices)
                assert v_cpu.shape[0] == len(chunk_indices)
                self.k_buffers[layer_id][chunk_indices] = k_cpu
                self.v_buffers[layer_id][chunk_indices] = v_cpu


class TestUnifiedRadixCache(unittest.TestCase):
    def setUp(self):
        TreeNode.counter = 0
        self.tmpdir = tempfile.TemporaryDirectory()
        self.allocator = FakeAllocator()
        self.cache = UnifiedRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            l3_dir=self.tmpdir.name,
            l3_budget_gb=0.01,
            l3_block_size=64,
            write_backend="sync",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _only_child(self, node):
        self.assertEqual(len(node.children), 1)
        return next(iter(node.children.values()))

    def _insert_with_filled_indices(self, tokens, base):
        indices = self.allocator.alloc(len(tokens))
        self.allocator.fill(indices, base)
        self.cache.insert(RadixKey(tokens), indices)
        return indices

    def _prepare_evicted_parent_with_child(self):
        self._insert_with_filled_indices([1, 2, 3, 4, 5, 6], base=10)
        self._insert_with_filled_indices([1, 2, 3, 4, 5, 6, 7, 8], base=100)

        parent = self._only_child(self.cache.root_node)
        child = self._only_child(parent)
        self.cache._offload_node_to_l3(child, reason="unit-test")
        self.cache._offload_node_to_l3(parent, reason="unit-test")
        return parent, child

    def test_match_splits_evicted_l3_node_and_preserves_tail_children(self):
        parent, child = self._prepare_evicted_parent_with_child()
        old_parent_id = parent.id
        old_parent_entry = parent.l3_entry
        old_file_path = old_parent_entry.file_path

        result = self.cache.match_prefix(RadixKey([1, 2, 3, 9]))

        self.assertEqual(result.host_hit_length, 3)
        self.assertEqual(len(result.device_indices), 0)
        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)

        self.assertEqual(prefix.key.token_ids, [1, 2, 3])
        self.assertEqual(tail.key.token_ids, [4, 5, 6])
        self.assertEqual(tail.id, old_parent_id)
        self.assertIs(child.parent, tail)
        self.assertEqual(child.key.token_ids, [7, 8])

        self.assertTrue(prefix.evicted)
        self.assertTrue(tail.evicted)
        self.assertTrue(child.evicted)
        self.assertIs(prefix.l3_entry.node, prefix)
        self.assertIs(tail.l3_entry.node, tail)
        self.assertIs(child.l3_entry.node, child)
        self.assertIsNot(self.cache.l3_entries[old_parent_id], old_parent_entry)
        self.assertFalse(os.path.exists(old_file_path))
        self.assertEqual(
            self.cache.stats.used_bytes,
            sum(entry.nbytes for entry in self.cache.l3_entries.values()),
        )

    def test_old_tail_hits_and_restores_after_partial_split(self):
        self._prepare_evicted_parent_with_child()
        self.cache.match_prefix(RadixKey([1, 2, 3, 9]))

        result = self.cache.match_prefix(RadixKey([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(result.host_hit_length, 8)

        loaded_indices, last_node = self.cache.init_load_back(
            result.last_host_node, result.host_hit_length
        )

        self.assertEqual(len(loaded_indices), 8)
        self.assertIs(last_node, result.last_host_node)
        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)
        child = self._only_child(tail)
        self.assertFalse(prefix.evicted)
        self.assertFalse(tail.evicted)
        self.assertFalse(child.evicted)
        self.assertEqual(len(prefix.value), 3)
        self.assertEqual(len(tail.value), 3)
        self.assertEqual(len(child.value), 2)

    def test_insert_splits_evicted_l3_node_and_adds_new_tail(self):
        self._prepare_evicted_parent_with_child()

        new_indices = self.allocator.alloc(5)
        self.allocator.fill(new_indices, base=500)
        self.cache.insert(RadixKey([1, 2, 3, 9, 10]), new_indices)

        prefix = self._only_child(self.cache.root_node)
        self.assertEqual(prefix.key.token_ids, [1, 2, 3])
        self.assertFalse(prefix.evicted)
        self.assertIsNotNone(prefix.l3_entry)

        child_keys = {
            tuple(child.key.token_ids): child for child in prefix.children.values()
        }
        self.assertIn((4, 5, 6), child_keys)
        self.assertIn((9, 10), child_keys)

        old_tail = child_keys[(4, 5, 6)]
        new_tail = child_keys[(9, 10)]
        self.assertTrue(old_tail.evicted)
        self.assertIsNotNone(old_tail.l3_entry)
        self.assertFalse(new_tail.evicted)
        self.assertIs(self._only_child(old_tail).parent, old_tail)

        old_result = self.cache.match_prefix(RadixKey([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertEqual(len(old_result.device_indices), 3)
        self.assertEqual(old_result.host_hit_length, 5)

    def test_non_evicted_split_still_uses_native_radix_behavior(self):
        self._insert_with_filled_indices([1, 2, 3, 4], base=700)

        result = self.cache.match_prefix(RadixKey([1, 2]))

        self.assertEqual(len(result.device_indices), 2)
        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)
        self.assertEqual(prefix.key.token_ids, [1, 2])
        self.assertEqual(tail.key.token_ids, [3, 4])
        self.assertFalse(prefix.evicted)
        self.assertFalse(tail.evicted)
        self.assertEqual(len(self.cache.l3_entries), 0)


class TestUnifiedRadixCacheAsync(unittest.TestCase):
    def setUp(self):
        TreeNode.counter = 0
        self.tmpdir = tempfile.TemporaryDirectory()
        self.allocator = FakeAllocator()
        self.cache = UnifiedRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            l3_dir=self.tmpdir.name,
            l3_budget_gb=0.01,
            l3_block_size=64,
            write_backend="async",
            max_pending_writes=1,
        )

    def tearDown(self):
        self.cache._stop_async_backend()
        self.tmpdir.cleanup()

    def _insert(self, tokens, base=10):
        indices = self.allocator.alloc(len(tokens))
        self.allocator.fill(indices, base)
        self.cache.insert(RadixKey(tokens), indices)
        return next(
            node
            for node in self.cache.root_node.children.values()
            if node.key.token_ids[0] == tokens[0]
        )

    def _wait_for_async(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while self.cache._ongoing_writes and time.monotonic() < deadline:
            self.cache.check_hicache_events()
            time.sleep(0.005)
        self.cache.check_hicache_events()
        self.assertFalse(self.cache._ongoing_writes)

    def _install_blocking_writer(self):
        started = threading.Event()
        release = threading.Event()
        original = self.cache._write_l3_entry_from_cpu

        def blocking_writer(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(timeout=3.0))
            return original(*args, **kwargs)

        self.cache._write_l3_entry_from_cpu = blocking_writer
        return started, release

    def test_finish_trigger_submission_is_non_blocking(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        # Production allocators may expose device as a string rather than torch.device.
        self.cache.device = "cpu"

        begin = time.perf_counter()
        self.assertEqual(
            self.cache._offload_node_to_l3(node, reason="finish-trigger"), 0
        )
        submit_elapsed = time.perf_counter() - begin

        self.assertLess(submit_elapsed, 0.1)
        self.assertTrue(started.wait(timeout=1.0))
        self.assertIsNotNone(node.value)
        self.assertEqual(node.lock_ref, 1)
        self.assertEqual(node.async_write_ref, 1)

        release.set()
        self._wait_for_async()
        self.assertTrue(node.evicted)
        self.assertIsNotNone(node.l3_entry)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.async_write_ref, 0)
        self.assertEqual(self.cache.stats.async_completed, 1)

    def test_completion_keeps_dram_while_request_holds_lock(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        self.assertTrue(started.wait(timeout=1.0))

        self.cache.inc_lock_ref(node)
        release.set()
        self._wait_for_async()

        self.assertIsNotNone(node.value)
        self.assertIsNotNone(node.l3_entry)
        self.assertEqual(node.lock_ref, 1)
        self.cache.dec_lock_ref(node)
        self.cache._offload_exact_prefix([1, 2, 3, 4], extra_key=None)
        self.assertTrue(node.evicted)

    def test_backpressure_skips_without_locking_extra_node(self):
        first = self._insert([1, 2], base=10)
        second = self._insert([3, 4], base=20)
        third = self._insert([5, 6], base=30)
        started, release = self._install_blocking_writer()

        self.cache._offload_node_to_l3(first, reason="finish-trigger")
        self.assertTrue(started.wait(timeout=1.0))
        self.cache._offload_node_to_l3(second, reason="finish-trigger")
        self.cache._offload_node_to_l3(third, reason="finish-trigger")

        self.assertEqual(len(self.cache._ongoing_writes), 2)
        self.assertEqual(self.cache.stats.async_backpressure_skipped, 1)
        self.assertEqual(third.lock_ref, 0)
        self.assertEqual(
            third.async_write_ref if hasattr(third, "async_write_ref") else 0, 0
        )

        release.set()
        self._wait_for_async()
        self.assertIsNotNone(third.value)

    def test_split_during_write_discards_stale_result(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        self.assertTrue(started.wait(timeout=1.0))

        self.cache.match_prefix(RadixKey([1, 2]))
        release.set()
        self._wait_for_async()

        prefix = next(iter(self.cache.root_node.children.values()))
        tail = next(iter(prefix.children.values()))
        self.assertEqual(prefix.key.token_ids, [1, 2])
        self.assertEqual(tail.key.token_ids, [3, 4])
        self.assertIsNotNone(prefix.value)
        self.assertIsNotNone(tail.value)
        self.assertIsNone(prefix.l3_entry if hasattr(prefix, "l3_entry") else None)
        self.assertIsNone(tail.l3_entry if hasattr(tail, "l3_entry") else None)
        self.assertEqual(self.cache.stats.async_stale, 1)

    def test_worker_failure_drops_unlocked_cache_node(self):
        node = self._insert([1, 2, 3, 4])

        def fail_snapshot(_indices):
            raise RuntimeError("injected snapshot failure")

        self.allocator.get_cpu_copy = fail_snapshot
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        self._wait_for_async()

        self.assertNotIn(node, self.cache.root_node.children.values())
        self.assertEqual(self.cache.stats.async_failed, 1)
        self.assertEqual(node.lock_ref, 0)

    def test_pressure_evict_waits_for_write_and_releases_tokens(self):
        node = self._insert([1, 2, 3, 4])
        self.cache.evict(4)

        self.assertTrue(node.evicted)
        self.assertIsNotNone(node.l3_entry)
        self.assertGreaterEqual(self.cache.stats.async_released_tokens, 4)

    def test_clear_removes_evicted_l3_subtree_and_restarts_worker(self):
        node = self._insert([1, 2, 3, 4])
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        self._wait_for_async()
        self.assertTrue(node.evicted)

        self.assertTrue(self.cache.clear_storage_backend())

        self.assertEqual(len(self.cache.root_node.children), 0)
        self.assertEqual(len(self.cache.l3_entries), 0)
        self.assertIsNotNone(self.cache._async_backend)

    def test_tp_slow_rank_delays_local_commit(self):
        node = self._insert([1, 2, 3, 4])
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        deadline = time.monotonic() + 2.0
        while (
            self.cache._async_backend._result_queue.empty()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        original_tp_min = self.cache._tp_min
        self.cache._tp_min = lambda _value: 0
        self.cache.check_hicache_events()
        self.assertIsNotNone(node.value)
        self.assertIn(node.id, self.cache._ongoing_writes)

        self.cache._tp_min = original_tp_min
        self._wait_for_async()
        self.assertTrue(node.evicted)
        self.assertIsNotNone(node.l3_entry)

    def test_tp_peer_failure_rolls_back_local_success(self):
        node = self._insert([1, 2, 3, 4])
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        deadline = time.monotonic() + 2.0
        while (
            self.cache._async_backend._result_queue.empty()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        call_count = 0

        def peer_fails_io(value):
            nonlocal call_count
            call_count += 1
            return 0 if call_count == 2 else value

        self.cache._tp_min = peer_fails_io
        self.cache.check_hicache_events()

        self.assertNotIn(node, self.cache.root_node.children.values())
        self.assertEqual(self.cache.stats.async_failed, 1)
        self.assertFalse(os.path.exists(self.cache.l3_run_dir / f"node-{node.id}.bin"))

    def test_reset_waits_for_active_worker_and_restarts_cleanly(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        self.cache._offload_node_to_l3(node, reason="finish-trigger")
        self.assertTrue(started.wait(timeout=1.0))
        timer = threading.Timer(0.05, release.set)
        timer.start()

        self.cache.reset()
        timer.join()

        self.assertEqual(len(self.cache.root_node.children), 0)
        self.assertFalse(self.cache._ongoing_writes)
        self.assertEqual(self.cache.stats.async_pending, 0)
        self.assertIsNotNone(self.cache._async_backend)
        self.assertEqual(list(self.cache.l3_run_dir.glob("*.tmp")), [])

    def test_async_budget_evicts_old_l3_entry_on_commit(self):
        first = self._insert([1, 2, 3, 4], base=10)
        self.cache._offload_node_to_l3(first, reason="finish-trigger")
        self._wait_for_async()
        first_entry_bytes = first.l3_entry.nbytes
        self.cache.l3_budget_bytes = first_entry_bytes

        second = self._insert([5, 6, 7, 8], base=20)
        self.cache._offload_node_to_l3(second, reason="finish-trigger")
        self._wait_for_async()

        self.assertNotIn(first, self.cache.root_node.children.values())
        self.assertIsNotNone(second.l3_entry)
        self.assertLessEqual(self.cache.stats.used_bytes, self.cache.l3_budget_bytes)


if __name__ == "__main__":
    unittest.main()
