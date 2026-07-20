import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_radix_cache_io import (
    AlignedPageBuffer,
    DirectPageArena,
    PageTransferBuffer,
)


class TestDirectPageArena(unittest.TestCase):
    def setUp(self):
        if not hasattr(os, "O_DIRECT"):
            self.skipTest("O_DIRECT is unavailable")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.buffer = AlignedPageBuffer(
            layer_num=2,
            page_size=4,
            head_num=1,
            head_dim=2,
            dtype=torch.float32,
            alignment=4096,
            pin_memory=False,
        )
        self.arena = DirectPageArena(
            Path(self.tmpdir.name) / "arena.bin",
            budget_bytes=self.buffer.record_bytes * 3,
            record_bytes=self.buffer.record_bytes,
            alignment=4096,
        )

    def tearDown(self):
        if hasattr(self, "arena"):
            self.arena.close()
        self.tmpdir.cleanup()

    def test_round_trip_and_sparse_capacity(self):
        slots = self.arena.reserve(2)
        self.assertEqual(len(slots), 2)
        self.buffer.tensor.copy_(
            torch.arange(self.buffer.tensor.numel(), dtype=torch.float32).view(
                self.buffer.shape
            )
        )
        expected = self.buffer.tensor.clone()

        self.arena.write(slots[0], self.buffer)
        self.buffer.raw.zero_()
        self.arena.read(slots[0], self.buffer)

        self.assertTrue(torch.equal(self.buffer.tensor, expected))
        self.assertEqual(self.arena.path.stat().st_size, self.buffer.record_bytes * 3)
        self.assertEqual(self.arena.used_slots, 2)

    def test_generation_rejects_stale_slot(self):
        old_slot = self.arena.reserve(1)[0]
        self.arena.release([old_slot])
        new_slot = self.arena.reserve(1)[0]

        self.assertEqual(old_slot.index, new_slot.index)
        self.assertNotEqual(old_slot.generation, new_slot.generation)
        self.assertFalse(self.arena.is_current(old_slot))
        with self.assertRaisesRegex(RuntimeError, "stale L3 slot"):
            self.arena.write(old_slot, self.buffer)

    def test_reserve_is_all_or_nothing(self):
        self.assertIsNone(self.arena.reserve(4))
        self.assertEqual(self.arena.used_slots, 0)


class TestPageTransferBuffer(unittest.TestCase):
    def test_cpu_snapshot_and_refill(self):
        page_size = 4
        k_buffer = [
            torch.arange(24, dtype=torch.float32).view(12, 1, 2),
            torch.arange(100, 124, dtype=torch.float32).view(12, 1, 2),
        ]
        v_buffer = [tensor + 1000 for tensor in k_buffer]
        pool = SimpleNamespace(
            device=torch.device("cpu"),
            layer_num=2,
            head_num=1,
            head_dim=2,
            store_dtype=torch.float32,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
        )
        transfer = PageTransferBuffer(pool, page_size, 4096)
        source = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
        destination = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        expected_k = [tensor[source].clone() for tensor in k_buffer]
        expected_v = [tensor[source].clone() for tensor in v_buffer]

        self.assertTrue(transfer.snapshot(source).query())
        for tensor in k_buffer + v_buffer:
            tensor[destination] = -1
        self.assertTrue(transfer.refill(destination).query())

        for layer_id in range(2):
            self.assertTrue(
                torch.equal(k_buffer[layer_id][destination], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(v_buffer[layer_id][destination], expected_v[layer_id])
            )

    def test_cpu_refill_accepts_a_partial_staging_buffer(self):
        pool = SimpleNamespace(
            device=torch.device("cpu"),
            layer_num=1,
            head_num=1,
            head_dim=1,
            store_dtype=torch.float32,
            k_buffer=[torch.zeros((8, 1, 1), dtype=torch.float32)],
            v_buffer=[torch.zeros((8, 1, 1), dtype=torch.float32)],
        )
        transfer = PageTransferBuffer(pool, page_size=4, alignment=4096)
        transfer.tensor[0, :, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 99.0])
        transfer.tensor[1, :, 0, 0, 0] = torch.tensor([4.0, 5.0, 6.0, 99.0])

        destination = torch.tensor([1, 3, 5], dtype=torch.int64)
        self.assertTrue(transfer.refill(destination).query())

        self.assertEqual(pool.k_buffer[0][destination, 0, 0].tolist(), [1, 2, 3])
        self.assertEqual(pool.v_buffer[0][destination, 0, 0].tolist(), [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
