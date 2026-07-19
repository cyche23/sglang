# UnifiedRadixCache Validation Report

> Historical results below document earlier baseline revisions. The current
> implementation is async-only best-effort write-through: every page-aligned
> finished request attempts an L3 backup, while memory-pressure eviction never
> submits or waits for L3 writes. The removed finish-threshold and write-backend
> CLI flags must not be used in new validation commands.

Date: 2026-07-09

Branch:

```bash
git branch --show-current
# local-v0.5.4-baseline
```

Runtime container:

```bash
docker exec sglang-dev-v054 bash -lc 'cd /codes/sglang && python3 -c "import sglang; print(sglang.__file__)"'
# /codes/sglang/python/sglang/__init__.py
```

Target model:

```bash
/models/Qwen3-1.7B/origin
```

## Async Write-Through Validation

Date: 2026-07-19

- `test/srt/test_unified_radix_cache_unit.py`: `18 passed` in the Jetson
  container. Coverage includes automatic finish backup, root-to-leaf
  backpressure, non-blocking eviction, unbacked deletion, partial split/restore,
  L3 budget eviction, reset/clear, and removed CLI options.
- All changed-file pre-commit hooks passed, including AST, isort, ruff, Black,
  and codespell.
- A page-size-64, 40960-token, five-trace high-pressure smoke processed 60
  request records before intentional interruption: 59 were HTTP 200/status
  `ok`; the one interrupted in-flight request is not a server failure.
- Server logs recorded 129 finish-trigger L3 writes, 41 pressure evictions, and
  30 L3 restores. They recorded zero `reason=dram-evict` writes and no prefill
  OOM, traceback, or scheduler exception.

The original 20-instance trace with production arrival timing was not rerun in
full during this change; the high-pressure smoke above is not a replacement for
that long-running acceptance test.

## Static Checks

```bash
PYTHONPYCACHEPREFIX=/tmp/sglang-pycache-check python3 -m py_compile \
  python/sglang/srt/mem_cache/unified_radix_cache.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/managers/scheduler.py \
  docs/unifiedradixcache_baseline_dev/unified_radix_cache_demo.py
```

Result: passed on host and inside `sglang-dev-v054`.

The host environment cannot run `python3 -m sglang.launch_server --help` because
it is missing Python runtime dependencies such as `tqdm`; the same command works
inside the container and shows the UnifiedRadixCache flags.

## Default Server Smoke

Command:

```bash
python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 8000 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 2048 \
  --enable-cache-report
```

Request:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello baseline smoke test.","sampling_params":{"temperature":0.0,"max_new_tokens":8}}'
```

Observed result: HTTP 200, non-empty text, `cached_tokens=0`.

## UnifiedRadixCache Restore Smoke

Server command:

```bash
python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 8000 \
  --max-total-tokens 4096 \
  --max-prefill-tokens 4096 \
  --enable-cache-report \
  --enable-unified-radix-cache \
  --unified-radix-cache-l3-dir /tmp/sglang-unified-radix-l3-smoke \
  --unified-radix-cache-l3-budget-gb 1.0 \
  --unified-radix-cache-l3-block-size 4096
```

Demo command:

```bash
python3 docs/unifiedradixcache_baseline_dev/unified_radix_cache_demo.py \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt-len 256 \
  --repeat 2 \
  --max-new-tokens 8 \
  --output /tmp/unified_radix_cache_demo.json
```

Observed key logs:

```text
UnifiedRadixCache enabled: l3_dir=/tmp/sglang-unified-radix-l3-smoke, l3_budget_bytes=1073741824
UnifiedRadixCache L3 write: node_id=8, token_count=2048, write_bytes=234881024
UnifiedRadixCache L3 hit: node_id=8, token_count=2048, page_count=2048
UnifiedRadixCache L3 read: node_id=8, read_bytes=234881024
UnifiedRadixCache L3 restore: node_id=8, token_count=2048, latency_ms=151.523
UnifiedRadixCache L3 write: node_id=8, reason=finish-trigger, token_count=2048
```

Finish-trigger now creates a backup-only L3 entry. It intentionally retains the
DRAM copy, so an immediate repeat can remain a DRAM hit. To validate `L3 hit`,
`L3 read`, and `L3 restore`, first apply enough memory pressure for leaf-based
eviction to log `DRAM copy released` for the backed-up prefix.

Observed demo summary:

```json
{
  "prompt_len_words_requested": 256,
  "recompute_latency_s": 1.497013098996831,
  "restore_latency_s": 1.1122739760030527,
  "restore_recompute_ratio": 0.7429954866449751,
  "recompute_prompt_tokens": 2731,
  "restore_prompt_tokens": 2475,
  "restored_cached_tokens": 2048,
  "output_nonempty": true
}
```

## Historical Async Write-back Extension Validation

Date: 2026-07-17

Static and unit checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/sglang-pycache-async python3 -m py_compile \
  python/sglang/srt/mem_cache/unified_radix_cache.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/managers/scheduler.py \
  test/srt/test_unified_radix_cache_unit.py

pre-commit run isort --files [CHANGED_PYTHON_FILES]
pre-commit run ruff --files [CHANGED_SGLANG_PYTHON_FILES]
pre-commit run black-jupyter --files [CHANGED_PYTHON_FILES]

docker exec sglang-dev-v054 bash -lc \
  'cd /codes/sglang && PYTHONPATH=python \
   python3 test/srt/test_unified_radix_cache_unit.py'
```

Result: static checks and formatting passed; all 15 UnifiedRadixCache unit
tests passed. The async tests cover non-blocking submission, active-request
locking, queue backpressure, stale split results, worker failure, pressure
eviction, budget eviction, reset/clear, and simulated TP readiness/failure.

The existing server on port 8000 was left untouched. A separate async smoke
server was launched on port 8001 with Qwen3-1.7B, a 2048-token pool and
`--unified-radix-cache-max-pending-writes 8`. The demo used:

```bash
python3 docs/unifiedradixcache_baseline_dev/unified_radix_cache_demo.py \
  --host 127.0.0.1 \
  --port 8001 \
  --prompt-len 64 \
  --repeat 4 \
  --max-new-tokens 8 \
  --output /tmp/unified_radix_cache_async_smoke.json
```

Observed async timeline:

```text
async L3 write submitted: node_id=2, token_count=640
HTTP POST /generate 200 OK
L3 write: node_id=2, write_bytes=73400320
async L3 write finished: node_id=2, committed=True,
  freed_tokens=640, snapshot_ms=110.285, write_ms=200.142

async L3 write submitted: node_id=3, token_count=576
HTTP POST /generate 200 OK
async L3 write finished: node_id=3, committed=True,
  snapshot_ms=17.413, write_ms=144.218
L3 hit: node_id=3, token_count=576
L3 read: node_id=3, read_bytes=66060288
L3 restore: node_id=3, token_count=576, latency_ms=46.739
```

The response preceding the corresponding write completion provides the runtime
non-blocking evidence. Two later requests restored 576 cached tokens from L3;
the best observed demo restore/recompute latency ratio was approximately 0.674.
The temporary port-8001 server was stopped after validation. No signal or
configuration change was sent to the pre-existing port-8000 server.

## L3 Budget Eviction Smoke

Server command is the same as above except:

```bash
--unified-radix-cache-l3-dir /tmp/sglang-unified-radix-l3-evict
--unified-radix-cache-l3-budget-gb 0.25
```

Observed key logs:

```text
UnifiedRadixCache L3 eviction: node_id=7, reason=budget, evicted_bytes=802816
UnifiedRadixCache L3 eviction: node_id=6, reason=budget, evicted_bytes=78331904
UnifiedRadixCache L3 eviction: node_id=10, reason=budget, evicted_bytes=802816
UnifiedRadixCache L3 eviction: node_id=9, reason=budget, evicted_bytes=48971776
UnifiedRadixCache L3 hit: node_id=8, token_count=2048
UnifiedRadixCache L3 read: node_id=8, read_bytes=234881024
UnifiedRadixCache L3 restore: node_id=8, latency_ms=179.448
```

Observed demo summary:

```json
{
  "prompt_len_words_requested": 256,
  "recompute_latency_s": 2.341211699997075,
  "restore_latency_s": 0.7076593629899435,
  "restore_recompute_ratio": 0.30226201372170985,
  "restored_cached_tokens": 2048,
  "l3_read_bytes": 234881024,
  "output_nonempty": true
}
```
