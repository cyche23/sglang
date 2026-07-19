# UnifiedRadixCache Baseline

This baseline is an experimental Jetson-oriented KV cache path for SGLang
v0.5.4. It treats GPU/CPU unified memory as one DRAM tier and adds an L3 SSD
tier behind the radix cache. Finished KV is written through to L3 by a bounded,
single-worker asynchronous backend.

## Scope

- Default behavior is unchanged. The feature is enabled only with
  `--enable-unified-radix-cache`.
- v1 supports only `MHATokenToKVPool`. Non-MHA, MLA, NSA, SWA, Mamba, and EAGLE
  paths fail fast.
- L3 data is process-local. The server creates a run-specific subdirectory below
  `--unified-radix-cache-l3-dir` and keeps metadata in memory only.
- L3 files are raw per-entry files. Metadata tracks node id, token/page count,
  dtype, shape, byte offsets, and aligned file size.
- Every non-empty page-aligned finished request attempts DRAM-to-SSD
  write-through. The background worker performs the CPU snapshot and raw-file
  write, then the scheduler thread commits radix metadata. Successful writes
  retain the DRAM copy.
- L3 restore and partial-node split I/O remain synchronous. There is no prefetch,
  Mooncake, HF3FS, NIXL, or remote KV backend in this baseline.
- Async writes are protected by radix reference locks. The queue is bounded and
  a full finish-trigger queue skips the write without blocking the request.
- DRAM residency is prefix-closed: an L3-only node cannot have a DRAM-resident
  descendant. Finish-trigger submits ancestors before suffix nodes. Pressure
  eviction releases backed device leaves and drops unbacked leaves without
  submitting or waiting for L3 I/O.

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
  --unified-radix-cache-max-pending-writes 8
```

`--unified-radix-cache-max-pending-writes` value counts queued operations and
excludes the single active or completed operation. There is no synchronous
write backend or finish-token threshold.

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
- `async L3 write submitted`
- `async L3 write finished`
- `L3 write`
- `DRAM copy released` and `pressure eviction finished` under memory pressure
- `L3 hit`, `L3 read`, and `L3 restore` after a backed-up prefix has been
  released by memory-pressure eviction
- `L3 eviction` when the configured L3 budget is exceeded

The demo JSON reports recompute latency, restore latency, restore/recompute
ratio, requested prompt length, prompt token counts returned by the server,
cached token counts, and output non-empty checks. The server logs are the
authoritative source for L3 byte counters and restore latency.

## Async baseline limitations

- The existing `MHATokenToKVPool.get_cpu_copy()` performs CUDA synchronization.
  Moving it to a worker removes scheduler-thread blocking, but it does not
  guarantee that GPU execution is completely stall-free.
- L3 budget accounting excludes the one temporary file currently being written;
  the budget is enforced before that file is committed as a cache entry.
- Memory-pressure eviction never waits for pending writes. Pending nodes are
  lock-protected and skipped, so `evict()` may release fewer tokens than
  requested when all candidates have an active finish-trigger write.
- Backpressure, write failure, or L3 budget eviction can leave a node without a
  backup. Memory pressure drops such an unlocked leaf and later requests
  recompute it.
