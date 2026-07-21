# UnifiedRadixCache 高并发 Prefill OOM 根因分析

日期：2026-07-18
调查版本：`35d0ac8408b5b09aa57a291afa1bbbeea1367545`（`impl async offloading baseline.`）
初始调查范围：只读复现、日志与代码调查。本文末尾另行记录基于该结论实施的修复与验证结果。

## 结论

这不是 GPU 物理显存不足，也不是 FlashInfer、chunked prefill、sync/async 写盘速度本身导致的 OOM。

核心原因是 UnifiedRadixCache 打破了原生 RadixCache 的树形驻留不变量：它允许一个内部节点已经从 DRAM eviction 到 L3（`node.value is None`），但该节点下面仍有 DRAM-resident 的子孙节点。随后 `_collect_leaves_device()` 和继承自 RadixCache 的 `total_size()` 遇到 evicted 节点会直接剪掉整棵子树，因此这些子孙持有的 KV page 仍然实际占用 DRAM，却无法被 eviction 遍历和释放。

结果形成如下闭环：

1. finish-trigger 沿已完成请求的整条 prefix path 做 offload，可能 offload 共享内部节点，但保留其他分支的 DRAM 子孙。
2. `evictable_size_` 仍把这些未锁定、实际 resident 的子孙计为可淘汰，因此调度器把它们纳入 admission budget。
3. 高并发下 free pages 下降，prefill 分配前调用 `tree_cache.evict(...)`。
4. eviction traversal 在 evicted ancestor 处剪枝，无法找到并释放被遮蔽的 resident descendants。
5. allocator 的实际 free pages 不足，`alloc_extend()` 返回 `None`，抛出 `Prefill out of memory`；与此同时错误信息仍显示大量 `evictable_size`。

sync 与 async 共用同一 tier topology、finish-trigger 和 device-leaf collector，所以两种写回方式都会失败。async 只改变 L3 写入与 DRAM copy 释放的时序，没有修复树的不变量。

## 实验结果

### 已有 sync 失败实验

日志：`/home/jetson/codes/experiment/outputs/sglang_server.log`

- 09:53:25 申请 2048 tokens 时 OOM。
- 报告 `available_size=1728`、`evictable_size=23296`，账面总可用 25024。
- eviction 在 OOM 前仅成功写出并释放 7 个小节点，共 960 tokens，未达到 paged prefill 本轮约 2112 tokens 的预留目标。
- `pretty_print()` 最后的 `#tokens` 为 15680。该值来自会在 evicted ancestor 处剪枝的 `total_size()`，不是完整 DRAM resident 总量。
- 因而至少有 `23296 - 15680 = 7616` 个账面 evictable tokens 位于 traversal 看不到的 DRAM 子树中；考虑 visible protected tokens 后，实际被遮蔽数量只会更高。

### 当前 async 版本复现

server 日志：`/home/jetson/codes/experiment/outputs/sglang_server_async_investigation.log`
loader 日志：`/home/jetson/codes/experiment/outputs/trace_loader_async_investigation.log`
请求日志：`/home/jetson/codes/experiment/outputs/sglang_syswait_arr0.02_seed42_async_investigation/request_log.jsonl`

使用用户给出的参数与 trace，默认 `write_backend=async`：

- 10:07:23 在 prefill 申请 2048 tokens 时稳定复现 OOM。
- allocator 实际只剩 `available_size=256`，但 `evictable_size=27968`，账面显示可用 28224。
- `pretty_print()` 的 device-visible `#tokens` 仅 12224。
- 至少 `27968 - 12224 = 15744` 个 evictable tokens 被 evicted ancestors 遮蔽，无法被 `_collect_leaves_device()` 找到。
- OOM 前 async writer 已正常提交和完成大量写入，没有 SSD write error；所以不是异步 worker 失败或 L3 budget 用尽。
- server OOM 后 loader 仍继续重试，之后对照 server 又使用了相同 8000 端口，因此该请求日志在故障点之后混入了连接异常和对照 server 的成功响应；根因证据应以 server 日志的 10:07:23 故障边界为准。

### 关闭 UnifiedRadixCache 的有界对照

server 日志：`/home/jetson/codes/experiment/outputs/sglang_server_no_unified_control.log`
请求日志：`/home/jetson/codes/experiment/outputs/sglang_syswait_arr0.02_seed42_no_unified_control/request_log.jsonl`

保持模型、40960 KV tokens、page size 64、FlashInfer、arrival rate、trace 和 seed 不变，仅关闭 UnifiedRadixCache：

- 有界运行完成 115 个请求，全部 `status=ok` / HTTP 200，无 OOM。
- 运行曾达到 9 个 running requests、token usage 0.99，仍能继续 prefill/decode。
- 对照在获得足够证据后人工结束，因此不是全量 benchmark 结果；它用于排除静态显存配置、trace 与 attention backend 本身。

## 最小结构复现（不启动模型）

用现有 unit-test fake allocator 构造：

```text
root
└── [1, 2]          # shared parent
    ├── [3, 4]      # DRAM resident
    └── [5, 6]      # DRAM resident
```

offload 共享 parent `[1, 2]` 后：

```text
before: evictable=6, total_size=6, leaves=[[5, 6], [3, 4]]
after:  parent_evicted=True, children_resident=[True, True]
        evictable=4, total_size=0, leaves=[]
evict(4) 后两个 children 仍然 resident，evictable 仍为 4
```

这直接证明：KV pages 没有泄漏出 allocator，也不是 page fragmentation；它们仍由树节点引用，但 eviction traversal 无法到达。

## 代码证据

### 1. finish-trigger 制造 mixed-tier hole

- `cache_finished_req()` 在阈值满足时调用 `_offload_exact_prefix()`。
- `_offload_exact_prefix()` 从 matched node 一直走到 root，并尝试 offload path 上每个 resident 节点。
- `_offload_node_to_l3()` / `_release_dram_copy()` 将 `node.value` 置为 `None`，但保留 node 及其 children，以维持 L3 radix metadata。

这里没有检查内部节点是否仍有 DRAM-resident descendants。因此在有分支共享 prefix 的 trace 中，evicted internal node + resident descendants 是必然可达状态。

相关位置：

- `python/sglang/srt/mem_cache/unified_radix_cache.py:613-627`
- `python/sglang/srt/mem_cache/unified_radix_cache.py:1146-1172`
- `python/sglang/srt/mem_cache/unified_radix_cache.py:1195-1208`

### 2. eviction collector 在 hole 处错误剪枝

`_collect_leaves_device()` 的关键逻辑是：

```python
if cur_node.evicted:
    continue
```

这段代码不仅跳过当前 L3 node，还跳过其全部 children。sync `evict()` 和 async `_evict_async()` 都完全依赖这个 collector，因此都无法释放被遮蔽 pages。

相关位置：`python/sglang/srt/mem_cache/unified_radix_cache.py:762-835,1534-1550`。

### 3. scheduler admission 与实际可回收集合脱节

调度器使用：

```text
allocator.available_size() + tree_cache.evictable_size()
```

作为请求可运行预算。`evictable_size_` 包含被遮蔽但仍 resident/unlocked 的节点；collector 却找不到它们。因此 admission 接纳请求后，实际分配阶段无法兑现预算。

相关位置：

- `python/sglang/srt/managers/schedule_policy.py:376-418`
- `python/sglang/srt/mem_cache/common.py:228-287`

### 4. 原生 RadixCache 假设不再成立

原生 RadixCache eviction 会删除 leaf，而不是保留 `value=None` 的内部节点；其 traversal 因而可以安全地把 evicted node 当作无 DRAM descendants。UnifiedRadixCache 为 L3 metadata 保留 evicted nodes，却复用了建立在旧不变量上的 traversal/accounting 辅助逻辑。

`RadixCache._total_size_helper()` 同样在 evicted child 处剪枝，所以 OOM 日志的 `#tokens` 恰好成为“device traversal 可见集合”的旁证。

相关位置：`python/sglang/srt/mem_cache/radix_cache.py:482-509,654-671`。

## 同源的正确性风险

mixed-tier hole 不只导致 OOM，还可能返回非连续 prefix KV。

在最小结构中 offload `[1, 2]`、保留 resident child `[3, 4]` 后，对 `[1,2,3,4]` 做 `match_prefix()`，当前实现返回：

```text
returned_indices = child [3,4] 对应的 KV indices
host_hit_length = 0
last_device_node = [3,4]
```

也就是说，缺失 ancestor `[1,2]` 时，descendant `[3,4]` 的 KV 被当成长度为 2 的连续前缀返回。`match_prefix()` 只从最后节点向上统计“连续 evicted suffix”；如果最后节点 resident，它不会发现中间存在 evicted ancestor。

这可能造成错误 prefix reuse，而不只是性能下降。因此后续修复不能只让 eviction “多遍历 children”，还必须定义并维护跨层 prefix 的连续性。

相关位置：`python/sglang/srt/mem_cache/unified_radix_cache.py:629-697,965-999`。

## 调查阶段建议的修复方向

优先建议恢复一个明确且易验证的不变量：

> DRAM residency 必须 prefix-closed：任何 DRAM-resident node 的所有 ancestors 也必须 DRAM-resident。等价地，evicted/L3-only node 下不能存在 DRAM-resident descendant。

据此：

1. finish-trigger 只能 offload 安全的 device leaves；内部节点只有在所有 DRAM descendants 已释放后才能释放自身 DRAM copy。
2. 对共享 prefix 的 path offload 应 bottom-up，并在遇到仍含 resident sibling/descendant 的 parent 时停止释放 parent（可以只创建 L3 backup，但不能把 parent DRAM copy 释放掉）。
3. async commit 时再次验证该不变量；提交期间树可能 split、restore 或被其他请求加锁，不能只验证 `node.value is operation.value`。
4. `evictable_size()` 必须等于 eviction 真正可发现并可释放的 resident/unlocked token 集合。增加从树重算的 debug sanity check。
5. `evict()` 应返回/记录实际 freed tokens；分配前后校验 allocator free-page 增量。若未达到目标，应让 scheduler 做保守回退，而不是继续相信账面 budget。
6. `match_prefix()` 必须保证返回的 device indices 从 root 开始连续。若允许 mixed-tier hole，则要先恢复缺失 ancestors，再拼接 descendants；否则直接禁止这种拓扑。

另一种方案是全面支持 arbitrary mixed-tier holes：collector 穿过 L3 nodes 查找 DRAM descendants、match/load-back 处理任意交错层级、lock/accounting 按 `node.value` 分层计算。该方案状态空间和并发复杂度明显更高，不适合作为首个修复。

## 修复前应补齐的测试

现有 unit tests 主要是线性路径、`page_size=1`，并且 fake allocator 的 `free()` 不把 pages 放回 `free_pages`，所以没有验证“eviction 后 allocator 能否再次分配”。建议至少补充：

1. 分支树：offload 一条完成路径后，断言不存在 L3-only ancestor + DRAM descendant。
2. sync/async 各自执行 `evict(N)`，断言 allocator `available_size` 实际增加至少 N（考虑 page rounding）。
3. `page_size=64` 的 alloc-evict-alloc 回归，覆盖 `alloc_extend()`。
4. mixed-tier prefix match：返回 indices 必须与 token prefix 连续对应，不能返回 descendant suffix 冒充 prefix。
5. `evictable_size`、protected size、resident token 重算值与 allocator capacity 的长期 invariant test。
6. async parent/child 并发 write、backpressure、stale result、restore/split 交错下的不变量测试。
7. 用户提供的 trace 做 sync、async、disabled 三组端到端回归；至少跑过当前两个稳定故障窗口，并检查无 OOM、无错误 cache reuse。

## 修复验收标准

- 任意时刻不存在 L3-only ancestor 下的 DRAM-resident descendant（若选择 prefix-closed 方案）。
- `available_size + actually_evictable + protected/running allocations` 与 KV pool capacity 在页粒度上守恒。
- 当日志称 `evictable_size >= requested` 时，`evict()` 必须能让后续 allocation 成功，或明确报告被锁定/异步 pending 的真实短缺，不能静默返回。
- sync 与 async 在同一 trace、高并发和 page size 64 下均不再触发 prefill OOM。
- prefix cache 命中结果与禁用 UnifiedRadixCache 的输出一致，避免只修 OOM 而保留错误 KV reuse。

## 最终修复策略与实现

最终采用 prefix-closed DRAM residency 方案，将 finish-trigger 与 pressure eviction 的职责严格拆开：

```text
finish-trigger:   DRAM only -> DRAM + L3 backup
pressure eviction: DRAM + L3 backup -> L3 only
```

实现要点如下：

1. 每个非空、page-aligned 的 finished request 都按根到叶顺序尝试异步写 L3；队列满时非阻塞跳过，优先保留连续可恢复的 ancestor prefix。
2. finish-trigger 成功后保留 `node.value`；写入失败、TP peer failure、backpressure 或 stale result 也不会删除 DRAM subtree。
3. pressure eviction 只从 device leaf 开始，并且不提交或等待 L3 I/O。有现成 L3 backup 时直接释放 DRAM；没有 backup 时删除 leaf 及其不可独立恢复的 L3 descendants。
4. pending finish write 通过 radix lock reference 保护，eviction 跳过这些节点；`_release_dram_copy()` 继续拒绝释放有锁或仍有 resident descendant 的节点。
5. 建立、替换或淘汰 resident 节点的 L3 backup 不改变 `evictable_size_`；只有 DRAM 实际释放或恢复时才改变该计数。

UnifiedRadixCache 固定使用 async backend。`--unified-radix-cache-offload-after-finish-min-tokens` 和 `--unified-radix-cache-write-backend` 已删除，旧命令会报告未知参数。

## 修复后验证

### 单元与静态验证

- `test/srt/test_unified_radix_cache_unit.py` 的 fake allocator 现在会在 `free()` 时真实归还 slot，并暴露 `available_size()`。
- async 回归覆盖自动 finish backup、根到叶提交、leaf-first pressure release、未备份节点直接删除、写入失败、backpressure、stale result、pending write 跳过和 L3 budget 淘汰 resident backup。
- 每条关键拓扑路径都断言不存在 “L3-only ancestor + DRAM-resident descendant”。
- 目标测试、仓库 pre-commit hooks 和 `git diff --check` 必须通过。

### 端到端验证

端到端结果使用相同模型、40960 KV tokens、page size 64、arrival rate 0.02、20 trace instances、seed 42，对比 async UnifiedRadixCache 与 disabled。验收时同时检查：所有请求状态、OOM/scheduler exception、只存在 finish-trigger 新 KV 写入、pressure eviction 没有写盘，以及后续 L3 restore 日志。

## 2026-07-20：异步 restore anchor 竞态

### 新复现与根因

在 `f8bd15349b` 上使用 10 个并发 agent trace、arrival rate 1 和 page size 64 时，finish-trigger 修复后的实现仍可稳定触发 prefill OOM：

- 用户实验在 13:49:33 失败，allocator 仅剩 448 tokens，但 `evictable_size=9600`；树中 51456 个 device-visible tokens 全部带锁，账面可淘汰容量无法兑现。
- 带 debug/profile 的同参数复现在 14:05:29 失败，OOM 前出现 `DRAM release skipped ... resident_descendant=True`，直接证明 L3-only ancestor 下再次出现 resident descendant。
- 两次请求日志均有 513 条；服务退出后其余请求转为 connection-refused，故障边界应以 server 的 scheduler exception 为准。

第二个 mixed-tier hole 来自异步 restore 生命周期，而不是 finish-trigger：

1. restore operation 记录一段连续 L3-only suffix，并在最近的 resident ancestor（anchor）下分配 device slots。
2. 旧实现只在 allocation 调用期间临时增加 anchor 的 radix lock，提交给 restore worker 后立即释放。
3. SSD read/CUDA refill 进行期间，pressure eviction 可以把未锁定的 anchor 释放到 L3。
4. restore completion 只检查 source nodes 和 L3 slots 是否仍有效，没有检查 anchor 是否仍 resident，因而把 suffix 提交回 DRAM。
5. 结果同时破坏 prefix-closed topology 和 lock/accounting：对 L3-only ancestor 执行 `inc_lock_ref()` / `dec_lock_ref()` 会把从未计入 resident cache 的 key 长度错误加入或移出 `evictable_size_`。

### 修复

- restore operation 显式保存 anchor；启动前验证 anchor resident、仍连接 radix root，且 source nodes 仍是从 anchor 开始的连续 parent-child path。
- 成功分配 restore slots 后持续持有 anchor lock，覆盖 worker I/O 和 scheduler result commit。
- commit 时先安装 restored values，再获取 terminal completion lock，最后释放 anchor lock，使保护无窗口地从 anchor 转交给 terminal。
- allocation/submit/read/stale/reset/clear/shutdown 等失败路径通过幂等清理释放 restore slots 和 anchor lock。
- 已提交 operation 的全部 waiter 若 abort，worker 安全结束后丢弃结果，不再创建无人释放的 completion lock。
- debug 模式在 cache-event 和 pressure-eviction 边界重算 prefix-closed、evictable/protected counters 和 device-index ownership；生产路径不增加整树遍历。

### 新增回归

单元测试新增覆盖：restore I/O 期间的 pressure eviction、allocation 前 anchor 已淘汰、anchor-to-terminal lock transfer、唯一 waiter abort、read/submit failure、active restore reset，以及 page size 64 的 restore-evict-alloc 守恒。

### Debug 压测发现的相邻页所有权问题

anchor 修复后的首轮 debug 压测越过了原稳定 OOM 窗口，但整树 device-index 校验在后续 batch 捕获到两个 resident node 共同持有 index 640。对应日志显示，chunked request 的 insert 在部分 L3 node 上遇到并发 restore，返回 `restore-inflight` 后进入提前退出路径。

该路径原本无条件调用 `_free_uninserted_value(value)`。对 finished request 这是正确清理；但 chunked request 会继续执行，并继续通过 `req.prefix_indices` 持有这些 KV indices。paged allocator 将整页归还后，该页会被分配给其他请求，原请求随后又可能把同一页插入 radix tree，最终形成重复物理页所有权并再次造成容量失配。

修复限定为 UnifiedRadixCache 内部：partial split 因非读取错误提前退出时，仅 non-chunked insert 释放未插入 value；chunked insert 保留仍由请求拥有的页。新增 page size 64 回归，强制 `restore-inflight` partial split，并验证 allocator 可用量和 free 记录均不变化。

### 最终验证结果

- 单元测试：UnifiedRadixCache 33 项全部通过，其中包含 restore anchor 生命周期、失败/abort/reset 清理、page size 64 守恒和 chunked `restore-inflight` 页所有权回归。
- I/O 测试：5 项全部通过；目标文件通过 Python 编译、pre-commit hooks 和 `git diff --check`。
- debug/profile 复现：严格使用 10 instances、arrival rate 1、seed 42 和 page size 64，513/513 请求 `status=ok` 且 HTTP 200；完整运行中未触发 mixed-tier、counter 或 device-index ownership 断言。
- 生产参数复现：关闭 debug/profile 后按用户命令完整运行，513/513 请求 `status=ok` 且 HTTP 200；server 日志没有 prefill OOM、scheduler exception、resident-descendant release warning 或 restore/lock cleanup error，loader 日志没有连接拒绝或请求异常。
