# UnifiedRadixCache L3 baseline performance repair

Date: 2026-07-20

## Outcome

The repaired L3 backend passes the lightweight acceptance gate on Jetson AGX
Orin with Qwen3-1.7B. Results below are medians of three isolated server runs;
profiling was disabled for the production measurements and enabled separately
for bottleneck attribution.

| Scenario | Mode | TTFT p50 | TTFT p95 | Throughput | Duration |
| --- | --- | ---: | ---: | ---: | ---: |
| Restore/reuse | Native radix | 5.121 s | 5.715 s | 2.778 req/s | 17.276 s |
| Restore/reuse | Repaired L3 | 2.731 s | 4.876 s | 4.029 req/s | 11.913 s |
| Unique/no reuse | Native radix | 6.942 s | 7.779 s | 2.135 req/s | 22.484 s |
| Unique/no reuse | Repaired L3 | 6.966 s | 7.515 s | 2.115 req/s | 22.699 s |

Relative to native radix, the L3 restore workload improves TTFT p50 by 46.7%,
TTFT p95 by 14.7%, and request throughput by 45.0%. On the unique workload,
throughput regresses by 0.95% and TTFT p50 by 0.35%, both within the 3% guard.

The frozen acceptance gates were therefore met:

- restore p50 and p95 improve by at least 10%;
- restore throughput improves by at least 5%;
- unique/no-reuse throughput regression is no more than 3%.

All 288 formal requests per mode completed successfully: three runs times 48
restore requests plus 48 unique requests.

## Supplied long-trace baseline

The supplied 964-request runs establish the original problem even though L3
increased the serving-log cache-hit ratio from 49.83% to 84.77%:

| Metric | Native | Original L3 baseline | Change |
| --- | ---: | ---: | ---: |
| TTFT mean | 12.052 s | 20.223 s | +67.8% |
| TTFT p50 | 5.824 s | 9.637 s | +65.5% |
| TTFT p95 | 34.613 s | 75.263 s | +117.4% |
| Measured duration | 2070.8 s | 2384.8 s | +15.2% |
| Target output throughput | 59.32 tok/s | 51.51 tok/s | -13.2% |

Sources:

- `/home/jetson/codes/experiment/outputs/sglang_arr0.02_seed42_bsl_02`
- `/home/jetson/codes/experiment/outputs/sglang_arr0.02_seed42_origin`

The full trace takes roughly 35--40 minutes per run, so it was analyzed but was
not rerun after every change. The deterministic benchmark added by this repair
was used for iteration and the three-run acceptance comparison.

## Bottleneck findings

### Write path

The original write implementation coupled best-effort L3 work to inference:

1. queued descriptors retained radix-node locks while waiting for the I/O
   worker;
2. the completion/ACK queue had capacity one, so a slow scheduler poll could
   block the worker;
3. per-node buffered files added file lifecycle overhead and page-cache
   interference;
4. a running write continued to compete with a latency-critical restore.

In the initial unique/no-reuse diagnostic, queued writes waited about 3.5 s on
average and spent about 1.17 s in I/O, producing 0.420 req/s and 46.1 s p95
TTFT. Releasing queued-node locks and making writes best effort restored the
same workload to approximately native performance before restore optimization
began.

### Restore path

The old restore performed a complete read followed by a separate refill, both
on the request path. Page-at-a-time refill also launched work once per model
layer for every cache page. Under concurrency, restore allocation, completion
locks, and chunked-prefill capacity could interact in three harmful ways:

- several completed restores could protect the entire device KV pool and leave
  all requests waiting;
- acknowledging a restore during policy rematching could evict it before the
  request actually entered `PrefillAdder`, causing repeated restores;
- a new restore allocation between two prefill chunks could consume the
  chunked request's implicit reservation and create an invalid zero-token
  forward.

After repair, two diagnostic runs recorded median restore-operation latency of
487--610 ms. Median direct-read time was 191--211 ms and median grouped refill
time was 181--194 ms. Exact-path waiter coalescing avoided duplicate reads, and
active writes were dropped at a page boundary whenever restore demand appeared.

## Implemented design

### Page-granular direct-I/O arena

- A single sparse, fixed-slot arena replaces per-node files.
- Slots are one radix-cache page and use generation numbers to reject stale
  asynchronous completions.
- `O_DIRECT` plus 4096-byte-aligned pinned buffers removes Linux page-cache
  double buffering.
- L3 metadata stores ordered page-slot references, so partial radix-node splits
  split metadata without rewriting SSD data.
- No `fsync` or persistence protocol is used; this remains an ephemeral cache.

### Non-blocking best-effort writes

- The scheduler submits descriptors without taking a long-lived node lock.
- The worker requests activation immediately before snapshot; the scheduler
  revalidates the descriptor and only then locks the node.
- Stale or backpressured descriptors are discarded without blocking inference.
- Results use an unbounded `SimpleQueue`, eliminating the one-entry ACK stall.
- An active write checks restore demand at page boundaries, drains any in-flight
  CUDA snapshot safely, and gives up its best-effort write.

### Asynchronous pipelined restore

- Request admission starts restore before the request reaches `PrefillAdder`.
- Two bounded restore workers use independent buffers and high-priority CUDA
  streams.
- Each worker has double-buffered 8-page refill groups. SSD records remain
  page-granular; pages are packed into a larger staging buffer to amortize
  per-layer refill kernel launches.
- Exact radix paths coalesce waiters into one restore operation.
- At most two unrelated protected restores are admitted. This permits useful
  SSD/GPU overlap without allowing completed restores to occupy the full 8192
  token device pool used by the benchmark.
- A completion lock is released only when a waiter reaches real admission, not
  during schedule-policy rematching.
- Restore allocation is paused while a chunked prefill is between chunks.
- Abort, stale-result, failed-read, reset, and storage-clear paths release
  allocations, locks, and slot references.

### Profiling checkpoints

`--unified-radix-cache-profile-path PATH` enables a bounded, best-effort JSONL
sink. File writes occur on a daemon thread; a full profiler queue drops events
instead of delaying inference. Important stages include:

- `scheduler_tick`, `prefix_match`, and `invalid_prefix_match`;
- `write_submitted`, `write_activated`, `write_worker_start`,
  `write_snapshot_complete`, `write_io_complete`, and commit/discard;
- `restore_scheduled`, `restore_coalesced`, `restore_start`,
  `restore_allocation_wait`, `restore_page_read`, `restore_complete`, and ACK.

High-frequency state events are throttled to one record per 100 ms for an
unchanged key/state.

## Benchmark and reproduction

The benchmark is `benchmark/unified_radix_cache/bench_unified_radix_cache.py`.
Its restore scenario populates 12 reusable prefixes, applies cache pressure,
then sends 48 requests at concurrency 16. The unique scenario sends 48
non-reusable requests at the same concurrency.

Server configuration used for both modes:

```bash
python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 18000 \
  --max-total-tokens 8192 \
  --page-size 64
```

L3 adds:

```bash
--enable-unified-radix-cache \
--unified-radix-cache-l3-dir /tmp/sglang-unified-radix-final \
--unified-radix-cache-l3-budget-gb 4 \
--unified-radix-cache-l3-block-size 4096 \
--unified-radix-cache-max-pending-writes 100
```

Run either scenario with:

```bash
PYTHONPATH=python python3 benchmark/unified_radix_cache/bench_unified_radix_cache.py \
  --base-url http://127.0.0.1:18000 \
  --scenario restore \
  --run-label example \
  --output /tmp/unified-radix-restore.json
```

Formal artifacts are under:

- `/home/jetson/codes/experiment/outputs/unified_radix_perf/final_native_{1,2,3}`
- `/home/jetson/codes/experiment/outputs/unified_radix_perf/final_l3_prod_{1,2,3}`
- diagnostic profiles:
  `/home/jetson/codes/experiment/outputs/unified_radix_perf/final_l3_{1,2}`

## Verification

- `test/srt/test_unified_radix_cache_unit.py`: 25 passed.
- `test/srt/test_unified_radix_cache_io.py`: 5 passed.
- Python compilation of all touched runtime, benchmark, and test modules passed
  inside `sglang-dev-v054`.
- Repository pre-commit hooks passed for every touched file.
- A stress smoke with 5.4K--6.0K-token prompts completed 24/24 requests after
  the chunked-prefill reservation fix.

## Scope and limitations

- UnifiedRadixCache now fails fast unless `tp_size == pp_size == dp_size == 1`.
- Only the generic MHA KV pool is supported. MLA, SWA/hybrid, Mamba, and EAGLE
  remain unsupported rather than silently running an unsafe path.
- `O_DIRECT` and a filesystem/device that supports aligned direct I/O are
  required.
- Restore staging consumes model-dependent pinned memory. For Qwen3-1.7B with
  page size 64, the two restore pipelines use roughly 250 MiB, in addition to
  two page-sized write buffers.
- L3 writes are intentionally lossy under pressure. This protects latency but
  may reduce future hit rate on a sustained unique workload.
- The repaired code passed the deterministic acceptance workload. A fresh
  multi-hour/three-run execution of the supplied 964-request trace remains a
  recommended follow-up before declaring results portable to every trace mix.
