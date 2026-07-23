"""Scheduler-side regressions for synchronous UnifiedRadixCache restore."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


class _Allocator:
    def __init__(self, available=128):
        self.available = available

    def available_size(self):
        return self.available


class _Cache:
    async_restore_prefetch = False

    def __init__(self, *, restore_succeeds=True):
        self.restore_succeeds = restore_succeeds
        self.anchor = object()
        self.terminal = object()
        self.events = []

    def evictable_size(self):
        return 0

    def inc_lock_ref(self, node):
        self.events.append(("lock", node))

    def dec_lock_ref(self, node):
        self.events.append(("unlock", node))

    def init_load_back(
        self, _last_host_node, host_hit_length, request_id=None, **_kwargs
    ):
        self.events.append(("restore", request_id))
        if not self.restore_succeeds:
            return torch.empty(0, dtype=torch.int64), self.anchor
        return torch.arange(
            10, 10 + host_hit_length, dtype=torch.int64
        ), self.terminal


def _request(cache, *, rid="req", host_hit_length=2):
    return SimpleNamespace(
        rid=rid,
        fill_ids=[1, 2, 3, 4, 5, 6],
        prefix_indices=torch.tensor([1, 2], dtype=torch.int64),
        last_node=cache.anchor,
        last_host_node=cache.terminal,
        host_hit_length=host_hit_length,
        extend_input_len=4,
        last_matched_prefix_len=2,
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=2, ignore_eos=False),
        swa_uuid_for_lock=None,
    )


def _adder(cache, *, available=128, rem_input_tokens=128, rem_chunk_tokens=None):
    return PrefillAdder(
        page_size=1,
        tree_cache=cache,
        token_to_kv_pool_allocator=_Allocator(available),
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=rem_input_tokens,
        rem_chunk_tokens=rem_chunk_tokens,
    )


def test_sync_restore_happens_after_admission_and_transfers_lock():
    cache = _Cache()
    req = _request(cache)
    adder = _adder(cache)

    result = adder.add_one_req(req, False, None)

    assert result == AddReqResult.CONTINUE
    assert adder.can_run_list == [req]
    assert req.prefix_indices.tolist() == [1, 2, 10, 11]
    assert req.extend_input_len == 2
    assert req.host_hit_length == 0
    assert cache.events == [
        ("lock", cache.anchor),
        ("restore", "req"),
        ("lock", cache.terminal),
        ("unlock", cache.anchor),
    ]


def test_failed_sync_restore_falls_back_to_recompute():
    cache = _Cache(restore_succeeds=False)
    req = _request(cache)
    adder = _adder(cache)

    result = adder.add_one_req(req, False, None)

    assert result == AddReqResult.CONTINUE
    assert adder.can_run_list == [req]
    assert req.prefix_indices.tolist() == [1, 2]
    assert req.extend_input_len == 4
    assert req.host_hit_length == 0
    assert req.last_node is cache.anchor
    assert req.last_host_node is cache.anchor
    assert cache.events.count(("restore", "req")) == 1


def test_failed_restore_rechecks_recompute_budget_before_batching():
    cache = _Cache(restore_succeeds=False)
    req = _request(cache)
    adder = _adder(cache, rem_input_tokens=3)
    adder.can_run_list.append(SimpleNamespace(rid="already-admitted"))

    result = adder.add_one_req(req, False, None)

    assert result == AddReqResult.OTHER
    assert [candidate.rid for candidate in adder.can_run_list] == [
        "already-admitted"
    ]
    assert cache.events.count(("restore", "req")) == 1
    assert not any(
        event == ("lock", cache.terminal) for event in cache.events
    )


def test_rejected_candidate_does_not_start_restore():
    cache = _Cache()
    req = _request(cache)
    adder = _adder(cache, rem_input_tokens=2)
    adder.can_run_list.append(SimpleNamespace(rid="already-admitted"))

    result = adder.add_one_req(req, False, None)

    assert result == AddReqResult.OTHER
    assert not any(event[0] == "restore" for event in cache.events)


def test_zero_chunk_budget_does_not_start_restore():
    cache = _Cache()
    req = _request(cache)
    adder = _adder(cache, rem_chunk_tokens=0)

    result = adder.add_one_req(req, False, None)

    assert result == AddReqResult.OTHER
    assert not any(event[0] == "restore" for event in cache.events)


def test_multiple_sync_restores_follow_prefilladder_order():
    cache = _Cache()
    first = _request(cache, rid="first")
    second = _request(cache, rid="second")
    adder = _adder(cache)

    first_result = adder.add_one_req(first, False, None)
    second_result = adder.add_one_req(second, False, None)

    assert first_result == AddReqResult.CONTINUE
    assert second_result == AddReqResult.CONTINUE
    assert [req.rid for req in adder.can_run_list] == ["first", "second"]
    assert [event for event in cache.events if event[0] == "restore"] == [
        ("restore", "first"),
        ("restore", "second"),
    ]
