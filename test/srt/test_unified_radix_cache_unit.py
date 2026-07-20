"""Unit tests for UnifiedRadixCache async write-through behavior."""

import argparse
import dataclasses
import heapq
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs


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

    def available_size(self):
        return len(self.free_pages)

    def free(self, free_index: torch.Tensor):
        if free_index is not None and free_index.numel() > 0:
            self.freed.append(free_index.clone())
            self.free_pages = torch.unique(
                torch.cat((self.free_pages, free_index.to(dtype=torch.int64)))
            )

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


class FakeReqToTokenPool:
    def __init__(self, size=4, max_context_len=128):
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int64, device="cpu"
        )
        self.freed = []

    def free(self, index):
        self.freed.append(index)


class TestUnifiedRadixCache(unittest.TestCase):
    def setUp(self):
        TreeNode.counter = 0
        self.tmpdir = tempfile.TemporaryDirectory()
        self.allocator = FakeAllocator()
        self.req_pool = FakeReqToTokenPool()
        self.cache = self._new_cache()

    def tearDown(self):
        self.cache._stop_async_backend()
        self.tmpdir.cleanup()

    def _new_cache(self, page_size=1, max_pending_writes=8):
        return UnifiedRadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=page_size,
            l3_dir=self.tmpdir.name,
            l3_budget_gb=0.01,
            l3_block_size=64,
            max_pending_writes=max_pending_writes,
        )

    def _only_child(self, node):
        self.assertEqual(len(node.children), 1)
        return next(iter(node.children.values()))

    def _insert(self, tokens, base=10, chunked=True):
        indices = self.allocator.alloc(len(tokens))
        self.allocator.fill(indices, base)
        self.cache.insert(RadixKey(tokens), indices, chunked=chunked)
        return next(
            node
            for node in self.cache.root_node.children.values()
            if node.key.token_ids[0] == tokens[0]
        )

    def _finish_req(self, tokens, *, is_insert=True, slot=0, base=10):
        indices = self.allocator.alloc(len(tokens))
        self.allocator.fill(indices, base)
        self.req_pool.req_to_token[slot, : len(tokens)] = indices
        req = SimpleNamespace(
            origin_input_ids=list(tokens),
            output_ids=[],
            extra_key=None,
            req_pool_idx=slot,
            prefix_indices=torch.empty(0, dtype=torch.int64),
            last_node=self.cache.root_node,
        )
        self.cache.cache_finished_req(req, is_insert=is_insert)
        return req

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

    def _assert_dram_residency_is_prefix_closed(self):
        stack = [(self.cache.root_node, False)]
        while stack:
            node, has_evicted_ancestor = stack.pop()
            if node != self.cache.root_node and not node.evicted:
                self.assertFalse(has_evicted_ancestor)
            next_has_evicted_ancestor = has_evicted_ancestor or (
                node != self.cache.root_node and node.evicted
            )
            stack.extend(
                (child, next_has_evicted_ancestor) for child in node.children.values()
            )

    def _prepare_evicted_parent_with_child(self):
        self._insert([1, 2, 3, 4, 5, 6], base=10)
        self._insert([1, 2, 3, 4, 5, 6, 7, 8], base=100)
        parent = self._only_child(self.cache.root_node)
        child = self._only_child(parent)
        self._insert(list(range(1, 9)), base=200, chunked=False)
        self._wait_for_async()
        self.cache.evict(8)
        self.assertTrue(parent.evicted)
        self.assertTrue(child.evicted)
        return parent, child

    def test_cache_finished_req_always_submits_page_aligned_insert(self):
        self.assertNotIn("cache_finished_req", UnifiedRadixCache.__dict__)
        self._finish_req([1, 2, 3, 4])
        self.assertEqual(self.cache.stats.async_submitted, 1)
        self._wait_for_async()

        node = self._only_child(self.cache.root_node)
        self.assertIsNotNone(node.l3_entry)
        self.assertEqual(node.l3_entry.token_count, 4)
        self.assertFalse(node.evicted)

    def test_cache_finished_req_skips_non_insert(self):
        self._finish_req([1, 2, 3, 4], is_insert=False)
        self.assertEqual(self.cache.stats.async_submitted, 0)
        self.assertEqual(len(self.cache.root_node.children), 0)

    def test_cache_finished_req_backs_up_only_page_aligned_tokens(self):
        self.cache._stop_async_backend()
        self.cache = self._new_cache(page_size=2)
        self._finish_req([1, 2, 3])
        self._wait_for_async()

        node = self._only_child(self.cache.root_node)
        self.assertEqual(node.key.token_ids, [1, 2])
        self.assertEqual(node.l3_entry.token_count, 2)

    def test_chunked_insert_skips_async_backup(self):
        self._insert([1, 2, 3, 4], chunked=True)

        self.assertEqual(self.cache.stats.async_submitted, 0)
        self.assertFalse(self.cache._ongoing_writes)

    def test_non_chunked_insert_submits_without_second_tree_traversal(self):
        indices = self.allocator.alloc(4)
        self.allocator.fill(indices, 10)

        with (
            mock.patch.object(
                self.cache,
                "match_prefix",
                side_effect=AssertionError("insert must not match the prefix twice"),
            ),
            mock.patch.object(
                self.cache,
                "_is_node_attached",
                side_effect=AssertionError("insert must not retraverse parent paths"),
            ),
        ):
            self.cache.insert(RadixKey([1, 2, 3, 4]), indices, chunked=False)

        self.assertEqual(self.cache.stats.async_submitted, 1)
        operation = next(iter(self.cache._ongoing_writes.values()))
        self.assertEqual(operation.reason, "insert-trigger")
        self._wait_for_async()

    def test_insert_trigger_allows_current_request_lock(self):
        node = self._insert([1, 2], chunked=True)
        self.cache.inc_lock_ref(node)
        indices = self.allocator.alloc(2)
        self.allocator.fill(indices, 20)

        prefix_len = self.cache.insert(
            RadixKey([1, 2]), indices, chunked=False
        )

        self.assertEqual(prefix_len, 2)
        self.assertIn(node.id, self.cache._ongoing_writes)
        self.assertEqual(node.lock_ref, 2)
        self.allocator.free(indices[:prefix_len])
        self.cache.dec_lock_ref(node)
        self._wait_for_async()
        self.assertEqual(node.lock_ref, 0)

    def test_insert_submission_is_non_blocking_and_retains_dram(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()

        begin = time.perf_counter()
        self.assertTrue(self.cache._submit_async_write(node, "insert-trigger"))
        self.assertLess(time.perf_counter() - begin, 0.1)
        self.assertTrue(started.wait(timeout=1.0))
        self.assertIsNotNone(node.value)

        release.set()
        self._wait_for_async()
        self.assertIsNotNone(node.value)
        self.assertIsNotNone(node.l3_entry)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(self.cache.stats.async_completed, 1)

    def test_insert_path_submits_root_to_leaf_under_backpressure(self):
        self.cache._stop_async_backend()
        self.cache = self._new_cache(max_pending_writes=1)
        self._insert([1, 2], base=10)
        self._insert([1, 2, 3, 4], base=20)
        self._insert([1, 2, 3, 4, 5, 6], base=30)
        parent = self._only_child(self.cache.root_node)
        middle = self._only_child(parent)
        leaf = self._only_child(middle)
        started, release = self._install_blocking_writer()

        self._insert([1, 2, 3, 4, 5, 6], base=40, chunked=False)
        self.assertTrue(started.wait(timeout=1.0))
        self.assertEqual(list(self.cache._ongoing_writes), [parent.id, middle.id])
        self.assertNotIn(leaf.id, self.cache._ongoing_writes)
        self.assertEqual(self.cache.stats.async_backpressure_skipped, 1)

        release.set()
        self._wait_for_async()
        self.assertIsNotNone(parent.l3_entry)
        self.assertIsNotNone(middle.l3_entry)
        self.assertIsNone(getattr(leaf, "l3_entry", None))

    def test_insert_failure_retains_unlocked_node(self):
        node = self._insert([1, 2, 3, 4])

        def fail_snapshot(_indices):
            raise RuntimeError("injected snapshot failure")

        self.allocator.get_cpu_copy = fail_snapshot
        self.assertTrue(self.cache._submit_async_write(node, "insert-trigger"))
        self._wait_for_async()
        self.assertIn(node, self.cache.root_node.children.values())
        self.assertIsNotNone(node.value)
        self.assertIsNone(getattr(node, "l3_entry", None))
        self.assertEqual(self.cache.stats.async_failed, 1)

    def test_partial_insert_failure_still_submits_valid_ancestors(self):
        self._insert([1, 2], base=10)
        self._insert([1, 2, 3, 4], base=20)
        parent = self._only_child(self.cache.root_node)
        child = self._only_child(parent)
        self.assertTrue(self.cache._submit_async_write(child, "insert-trigger"))
        self._wait_for_async()
        self.cache.evict(2)
        self.assertTrue(child.evicted)

        indices = self.allocator.alloc(4)
        self.allocator.fill(indices, 30)
        with mock.patch.object(
            self.cache,
            "_split_evicted_l3_node",
            return_value=(None, "write-failure"),
        ):
            self.cache.insert(RadixKey([1, 2, 3, 9]), indices, chunked=False)

        self.assertIn(parent.id, self.cache._ongoing_writes)
        self._wait_for_async()
        self.assertIsNotNone(parent.l3_entry)

    def test_split_during_write_discards_stale_result_and_retains_dram(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        self.cache._submit_async_write(node, "insert-trigger")
        self.assertTrue(started.wait(timeout=1.0))

        self.cache.match_prefix(RadixKey([1, 2]))
        release.set()
        self._wait_for_async()

        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)
        self.assertIsNotNone(prefix.value)
        self.assertIsNotNone(tail.value)
        self.assertIsNone(getattr(prefix, "l3_entry", None))
        self.assertIsNone(getattr(tail, "l3_entry", None))
        self.assertEqual(self.cache.stats.async_stale, 1)

    def test_evict_backed_leaf_releases_without_rewriting(self):
        node = self._insert([1, 2, 3, 4])
        self.cache._submit_async_write(node, "insert-trigger")
        self._wait_for_async()
        available_before = self.allocator.available_size()
        writes_before = self.cache.stats.write_count
        submissions_before = self.cache.stats.async_submitted

        self.cache.evict(4)

        self.assertTrue(node.evicted)
        self.assertIsNotNone(node.l3_entry)
        self.assertEqual(self.allocator.available_size(), available_before + 4)
        self.assertEqual(self.cache.stats.write_count, writes_before)
        self.assertEqual(self.cache.stats.async_submitted, submissions_before)

    def test_evict_unbacked_leaf_drops_without_writing(self):
        node = self._insert([1, 2, 3, 4])
        available_before = self.allocator.available_size()

        self.cache.evict(4)

        self.assertNotIn(node, self.cache.root_node.children.values())
        self.assertEqual(self.allocator.available_size(), available_before + 4)
        self.assertEqual(self.cache.stats.async_submitted, 0)
        self.assertEqual(self.cache.stats.write_count, 0)

    def test_evict_unbacked_parent_removes_unreachable_l3_descendant(self):
        self._insert([1, 2], base=10)
        self._insert([1, 2, 3, 4], base=20)
        parent = self._only_child(self.cache.root_node)
        child = self._only_child(parent)
        self.cache._submit_async_write(child, "insert-trigger")
        self._wait_for_async()
        child_file_path = child.l3_entry.file_path

        self.cache.evict(2)
        self.assertTrue(child.evicted)
        self.assertIsNotNone(child.l3_entry)
        self.cache.evict(2)

        self.assertEqual(len(self.cache.root_node.children), 0)
        self.assertEqual(len(self.cache.l3_entries), 0)
        self.assertFalse(os.path.exists(child_file_path))

    def test_evict_uses_heap_and_promotes_parent(self):
        self._insert([1, 2], base=10)
        self._insert([1, 2, 3, 4], base=20)
        original_heapify = heapq.heapify
        original_heappop = heapq.heappop
        original_heappush = heapq.heappush

        with (
            mock.patch(
                "sglang.srt.mem_cache.unified_radix_cache.heapq.heapify",
                side_effect=original_heapify,
            ) as heapify_mock,
            mock.patch(
                "sglang.srt.mem_cache.unified_radix_cache.heapq.heappop",
                side_effect=original_heappop,
            ) as heappop_mock,
            mock.patch(
                "sglang.srt.mem_cache.unified_radix_cache.heapq.heappush",
                side_effect=original_heappush,
            ) as heappush_mock,
        ):
            self.cache.evict(4)

        heapify_mock.assert_called_once()
        self.assertEqual(heappop_mock.call_count, 2)
        heappush_mock.assert_called_once()
        self.assertEqual(len(self.cache.root_node.children), 0)

    def test_evict_does_not_wait_for_or_submit_writes(self):
        pending = self._insert([1, 2, 3, 4], base=10)
        victim = self._insert([5, 6, 7, 8], base=20)
        started, release = self._install_blocking_writer()
        self.cache._submit_async_write(pending, "insert-trigger")
        self.assertTrue(started.wait(timeout=1.0))
        submitted_before = self.cache.stats.async_submitted

        begin = time.perf_counter()
        self.cache.evict(4)
        elapsed = time.perf_counter() - begin

        self.assertLess(elapsed, 0.1)
        self.assertIn(pending, self.cache.root_node.children.values())
        self.assertNotIn(victim, self.cache.root_node.children.values())
        self.assertEqual(self.cache.stats.async_submitted, submitted_before)
        release.set()
        self._wait_for_async()

    def test_release_guard_rejects_parent_with_resident_descendants(self):
        self._insert([1, 2], base=10)
        self._insert([1, 2, 3, 4], base=20)
        parent = self._only_child(self.cache.root_node)
        available_before = self.allocator.available_size()

        released = self.cache._release_dram_copy(parent, reason="unit-test")

        self.assertEqual(released, 0)
        self.assertFalse(parent.evicted)
        self.assertEqual(self.allocator.available_size(), available_before)

    def test_match_splits_evicted_l3_node_and_preserves_tail_children(self):
        parent, child = self._prepare_evicted_parent_with_child()
        old_parent_id = parent.id
        old_parent_entry = parent.l3_entry
        old_file_path = old_parent_entry.file_path

        result = self.cache.match_prefix(RadixKey([1, 2, 3, 9]))

        self.assertEqual(result.host_hit_length, 3)
        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)
        self.assertEqual(prefix.key.token_ids, [1, 2, 3])
        self.assertEqual(tail.key.token_ids, [4, 5, 6])
        self.assertEqual(tail.id, old_parent_id)
        self.assertIs(child.parent, tail)
        self.assertTrue(prefix.evicted)
        self.assertTrue(tail.evicted)
        self.assertTrue(child.evicted)
        self.assertFalse(os.path.exists(old_file_path))
        self.assertIsNot(self.cache.l3_entries[old_parent_id], old_parent_entry)

    def test_partial_split_tail_restores(self):
        self._prepare_evicted_parent_with_child()
        self.cache.match_prefix(RadixKey([1, 2, 3, 9]))
        result = self.cache.match_prefix(RadixKey(list(range(1, 9))))

        loaded_indices, _ = self.cache.init_load_back(
            result.last_host_node, result.host_hit_length
        )

        self.assertEqual(len(loaded_indices), 8)
        prefix = self._only_child(self.cache.root_node)
        tail = self._only_child(prefix)
        child = self._only_child(tail)
        self.assertFalse(prefix.evicted)
        self.assertFalse(tail.evicted)
        self.assertFalse(child.evicted)
        self._assert_dram_residency_is_prefix_closed()

    def test_clear_removes_l3_only_subtree_and_restarts_worker(self):
        node = self._insert([1, 2, 3, 4])
        self.cache._submit_async_write(node, "insert-trigger")
        self._wait_for_async()
        self.cache.evict(4)

        self.assertTrue(self.cache.clear_storage_backend())

        self.assertEqual(len(self.cache.root_node.children), 0)
        self.assertEqual(len(self.cache.l3_entries), 0)
        self.assertIsNotNone(self.cache._async_backend)

    def test_reset_waits_for_active_worker_and_restarts_cleanly(self):
        node = self._insert([1, 2, 3, 4])
        started, release = self._install_blocking_writer()
        self.cache._submit_async_write(node, "insert-trigger")
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

    def test_l3_budget_evicts_old_resident_backup(self):
        first = self._insert([1, 2, 3, 4], base=10)
        self.cache._submit_async_write(first, "insert-trigger")
        self._wait_for_async()
        self.cache.l3_budget_bytes = first.l3_entry.nbytes

        second = self._insert([5, 6, 7, 8], base=20)
        self.cache._submit_async_write(second, "insert-trigger")
        self._wait_for_async()

        self.assertIsNotNone(first.value)
        self.assertIsNone(getattr(first, "l3_entry", None))
        self.assertIsNotNone(second.l3_entry)
        self.assertLessEqual(self.cache.stats.used_bytes, self.cache.l3_budget_bytes)

    def test_removed_cli_options_are_absent(self):
        field_names = {field.name for field in dataclasses.fields(ServerArgs)}
        self.assertNotIn(
            "unified_radix_cache_offload_after_finish_min_tokens", field_names
        )
        self.assertNotIn("unified_radix_cache_write_backend", field_names)

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn(
            "--unified-radix-cache-offload-after-finish-min-tokens",
            option_strings,
        )
        self.assertNotIn("--unified-radix-cache-write-backend", option_strings)


if __name__ == "__main__":
    unittest.main()
