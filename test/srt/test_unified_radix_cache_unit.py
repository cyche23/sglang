"""
Unit tests for UnifiedRadixCache L3 radix split behavior.

These tests use a minimal MHA-compatible KV pool and allocator so the cache
exercises real L3 file serialization without requiring a model server.
"""

import tempfile
import unittest
import os

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
            torch.zeros((size + 1, 1, 1), dtype=torch.float32)
            for _ in range(layer_num)
        ]
        self.v_buffers = [
            torch.zeros((size + 1, 1, 1), dtype=torch.float32)
            for _ in range(layer_num)
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

        old_result = self.cache.match_prefix(
            RadixKey([1, 2, 3, 4, 5, 6, 7, 8])
        )
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


if __name__ == "__main__":
    unittest.main()
