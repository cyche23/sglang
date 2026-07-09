# UnifiedRadixCache Validation Report

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
  --unified-radix-cache-l3-block-size 4096 \
  --unified-radix-cache-offload-after-finish-min-tokens 128
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
