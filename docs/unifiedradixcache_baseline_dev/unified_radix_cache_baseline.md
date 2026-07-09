# UnifiedRadixCache Baseline

This baseline is an experimental Jetson-oriented KV cache path for SGLang
v0.5.4. It treats GPU/CPU unified memory as one DRAM tier and adds a synchronous
L3 SSD tier behind the radix cache.

## Scope

- Default behavior is unchanged. The feature is enabled only with
  `--enable-unified-radix-cache`.
- v1 supports only `MHATokenToKVPool`. Non-MHA, MLA, NSA, SWA, Mamba, and EAGLE
  paths fail fast.
- L3 data is process-local. The server creates a run-specific subdirectory below
  `--unified-radix-cache-l3-dir` and keeps metadata in memory only.
- L3 files are raw per-entry files. Metadata tracks node id, token/page count,
  dtype, shape, byte offsets, and aligned file size.
- L3 I/O is synchronous. There is no prefetch, async write-back, Mooncake, HF3FS,
  NIXL, or remote KV backend in this baseline.

## Reproduce

Check the branch:

```bash
git branch --show-current
```

Start the server inside the Jetson SGLang container:

```bash
cd /codes/sglang

python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-cache-report \
  --enable-unified-radix-cache \
  --unified-radix-cache-l3-dir /tmp/sglang-unified-radix-l3 \
  --unified-radix-cache-l3-budget-gb 1.0 \
  --unified-radix-cache-l3-block-size 4096 \
  --unified-radix-cache-offload-after-finish-min-tokens 512
```

Run the demo client in another shell in the same container:

```bash
cd /codes/sglang

python3 docs/unifiedradixcache_baseline_dev/unified_radix_cache_demo.py \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt-len 1024 \
  --repeat 2 \
  --output /tmp/unified_radix_cache_demo.json
```

Check server logs:

```bash
grep -E "UnifiedRadixCache|L3 write|L3 hit|L3 read|L3 restore|L3 eviction" [SERVER_LOG]
```

Check client-side latency summary:

```bash
cat /tmp/unified_radix_cache_demo.json
```

## Expected Signals

The server log should show:

- `UnifiedRadixCache enabled`
- `L3 write`
- `L3 hit`
- `L3 read`
- `L3 restore`
- `L3 eviction` when the configured L3 budget is exceeded

The demo JSON reports recompute latency, restore latency, restore/recompute
ratio, requested prompt length, prompt token counts returned by the server,
cached token counts, and output non-empty checks. The server logs are the
authoritative source for L3 byte counters and restore latency.
