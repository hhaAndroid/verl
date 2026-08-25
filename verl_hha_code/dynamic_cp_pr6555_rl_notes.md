# verl PR #6555 Dynamic Context Parallel 学习笔记

这份笔记用尽量直观的方式说明：

- Dynamic Context Parallel（动态 CP、DCP）到底解决什么问题；
- 为什么同一次训练中，4 张卡可以一会儿组成 CP4，一会儿拆成两个 CP2，或者四个 CP1；
- verl PR #6555 怎样把 Megatron-Core 的动态 CP 调度接入 RL 训练；
- rollout、log-prob、reference model、actor update 分别需要改什么；
- 性能提升来自哪里，什么情况下收益很小；
- 当前实现有哪些限制和需要继续验证的地方。

## 0. 分析基准

本文主要基于：

```text
verl PR:     https://github.com/verl-project/verl/pull/6555
PR 标题:     [megatron] feat: add dynamic context parallel scheduling
分析 head:   1118ecc

Megatron PR: https://github.com/NVIDIA/Megatron-LM/pull/5154
依赖 commit: d2e7ec5b
```

截至 2026-08-06，verl PR #6555 仍然是 open 状态，后续实现可能继续变化。

为了把问题说清楚，本文大部分例子使用：

```text
总 GPU 数：4
TP：1
PP：1
静态 CP：4
DP：1

每张 GPU 能承受的 sequence token budget：8K
因此 CP4 最大可以处理约 32K
```

真实训练还可能包含 TP、PP、EP。动态 CP 实际是在固定 TP/PP 坐标下的
`DP × CP` rank 平面上工作，但不影响本文讨论的核心原理。

## 1. 先用一句话理解动态 CP

静态 CP 是：

```text
不管样本多长，4 张卡永远绑在一起处理一个 packed micro-batch。
```

动态 CP 是：

```text
长样本让 4 张卡合作处理；
中等样本让 2 张卡合作处理，允许4张卡中，其余两张卡是 cp2，其余两张卡是 cp1;
短样本让每张卡各自处理一份数据；
这些 forward/backward 的梯度最后仍然汇总到同一次 optimizer step。
```

因此它可以直观地理解成：

```text
根据样本长度，动态地在“Context Parallel”和“Data Parallel”之间借 GPU。
```

但它没有动态改变模型权重切分：

```text
TP 不变；
PP 不变；
EP 不变；
模型参数所在的 rank 不变；
optimizer state 不变。
```

动态变化的只有：

```text
1. 当前 micro-batch 由哪些 DP×CP ranks 处理；
2. 这些 ranks 组成多大的 CP group；
3. sequence token 如何在这个局部 CP group 内切分；
4. loss、梯度和输出如何去重及恢复顺序。
```



## 2. 为什么静态 CP4 会浪费

假设每张 GPU 最多承担 8K sequence tokens。

那么不同长度样本真正需要的最低 CP 大约是：

```text
1K  样本 -> CP1
7K  样本 -> CP1
12K 样本 -> CP2
16K 样本 -> CP2
31K 样本 -> CP4
```

静态 CP4 为了保证 31K 样本不 OOM，会让所有样本都使用 CP4。

处理一条 4K 样本时，静态 CP4 大致变成：

```text
GPU0：约 1K tokens
GPU1：约 1K tokens
GPU2：约 1K tokens
GPU3：约 1K tokens
```

这存在两个问题。

### 2.1 短计算无法隐藏 CP 通信

CP attention 中，每个 rank 只持有一部分 query/token，但需要看到完整序列的 KV，
因此 CP ranks 之间需要交换 KV。

对长序列而言，attention 计算很重，通信可能被计算覆盖。

对短序列而言：

```text
每张卡只计算很少的 tokens
    +
仍要启动 CP collective
    =
通信暴露，GPU 利用率下降
```



### 2.2 四张卡只能一起处理一份短任务

一条 4K 样本本来一张 GPU 就能装下，但静态 CP4 强制四张卡合作。

如果有四条独立的 4K 样本，更高效的方式其实是：

```text
GPU0：4K sample A，CP1
GPU1：4K sample B，CP1
GPU2：4K sample C，CP1
GPU3：4K sample D，CP1
```

四张卡并行处理四份独立数据，而且没有 CP 通信。

## 3. CP group 为什么能动态切换

关键点是：运行时并不会临时创建 NCCL process group。

Megatron-Core 在 model-parallel 初始化阶段就把可能使用的 group 全部创建好。

对于 4 个 DP×CP ranks，会预创建：

```text
CP1 groups:
  [GPU0]
  [GPU1]
  [GPU2]
  [GPU3]

CP2 groups:
  [GPU0, GPU1]
  [GPU2, GPU3]

CP4 group:
  [GPU0, GPU1, GPU2, GPU3]
```

每个 rank 只保存自己所属的 group handle：

```text
GPU0:
  size=1 -> [0]
  size=2 -> [0,1]
  size=4 -> [0,1,2,3]

GPU1:
  size=1 -> [1]
  size=2 -> [0,1]
  size=4 -> [0,1,2,3]

GPU2:
  size=1 -> [2]
  size=2 -> [2,3]
  size=4 -> [0,1,2,3]

GPU3:
  size=1 -> [3]
  size=2 -> [2,3]
  size=4 -> [0,1,2,3]
```

运行时所谓“切换 CP”，本质只是：

```python
cp_group = dynamic_groups[local_cp_size]
```

不是：

```text
forward 开始 -> 创建 process group -> forward 结束 -> 销毁 group
```

所以切换开销很小。

### 3.1 第一个 forward wave

调度器可以给出：

```text
GPU0: sample A
GPU1: sample A
GPU2: sample B
GPU3: sample C
```

因为 GPU0、GPU1 拿到相同的 sample ID 列表，它们组成 CP2：

```text
GPU0-1：sample A，CP2
GPU2：sample B，CP1
GPU3：sample C，CP1
```

attention 期间：

```text
GPU0 <-> GPU1：只在 [0,1] group 内通信
GPU2：CP1，不做 CP 通信
GPU3：CP1，不做 CP 通信
```



### 3.2 下一个 forward wave

调度结果可以变成：

```text
GPU0: sample D
GPU1: sample E
GPU2: sample F
GPU3: sample F
```

于是：

```text
GPU0：CP1
GPU1：CP1
GPU2-3：CP2
```

GPU2、GPU3 选择初始化时已经存在的 `[2,3]` group。

### 3.3 再下一个 forward wave

如果有一条 31K 样本：

```text
GPU0: sample G
GPU1: sample G
GPU2: sample G
GPU3: sample G
```

四张卡都选择 CP4 group：

```text
[GPU0, GPU1, GPU2, GPU3]
```



### 3.4 为什么不同 group 不会互相冲突

`[0,1]` 和 `[2,3]` 是两个不同且不重叠的 NCCL communicator：

```text
communicator A：[GPU0, GPU1]
communicator B：[GPU2, GPU3]
```

两个 CP2 attention 可以并行通信。

它们完成各自的 forward/backward 后，在需要全局梯度归并时，四张卡再进入共同的
DP×CP gradient collective。

### 3.5 当前实现不能任意选卡

当前 group 是连续、对齐、通常为 2 的幂大小的分组。

允许：

```text
[0,1]
[2,3]
[0,1,2,3]
```

一般不允许：

```text
[0,2]
[1,2]
[0,3]
```

这样可以避免初始化所有 rank 组合造成 communicator 数量爆炸，也能保证每个 rank
只通过 `group_size` 就确定自己应该选择哪个局部 group。

## 4. 调度器到底调度什么

调度器的输入应该是保留边界的原始独立样本，例如：

```text
input_ids.offsets().diff()
    = [31K, 12K, 8K, 7K, 5K, 4K, 3K, 2K, 1K, 1K]
```

它不应该只看到已经不可拆分的：

```text
[32K, 32K, 12K]
```

这是理解 verl 动态 batch 和动态 CP 关系的关键。

### 4.1 普通 dynamic batch

普通 dynamic batch 通常按 token 总数将独立样本组成 micro-batch：

```text
sum(sequence_lengths) <= token budget
```

例如：

```text
31K + 1K -> 一个 32K micro-batch
12K + 8K + 7K + 5K -> 一个 32K micro-batch
其他样本 -> 下一个 micro-batch
```

如果所有 32K micro-batch 都固定使用 CP4，那么第二个 micro-batch 即使由很多短序列组成，
也会强制使用四卡 CP 通信。

### 4.2 DCP 重新决定 packing 和 CP size

PR #6555 的 DCP 路径会绕过原来的 `prepare_micro_batches()`，直接把独立样本长度交给
Megatron-Core `DefaultDynamicCPScheduler`。

因此它可以把第二个 32K 工作量改成：

```text
GPU0：若干短样本，pack 到约 8K，CP1
GPU1：若干短样本，pack 到约 8K，CP1
GPU2：若干短样本，pack 到约 8K，CP1
GPU3：若干短样本，pack 到约 8K，CP1
```

四张卡仍然总共处理约 32K tokens，但不再使用 CP4。

### 4.3 最低 CP size

设每 rank 最大 sequence budget 为 `M`，样本长度为 `S`。

最低 rank 数大致为：

```text
ceil(S / M)
```

当前 Megatron 默认调度器通常会再向上取到 2 的幂：

```text
C_min = next_power_of_two(ceil(S / M))
```

当 `M=8K` 时：

```text
S <= 8K       -> CP1
8K < S <=16K -> CP2
16K<S <=32K  -> CP4
```



### 4.4 调度器不仅看 token 总数

对 block-diagonal packed attention，多个独立序列的 attention workload 近似是：

```text
sum(S_i^2)
```

而不是：

```text
(sum(S_i))^2
```

也不是单纯的：

```text
sum(S_i)
```

例如总 token 都是 32K：

```text
8 × 4K:
  attention workload ~ 8 × 4K^2 = 128M

2 × 16K:
  attention workload ~ 2 × 16K^2 = 512M
```

第二种组合的 attention 计算量大约是第一种的 4 倍。

普通 token-based dynamic batch 不能消除这种差异。

当前 MCore 默认 DCP 调度器使用一个简化的 attention workload 估算：

```text
workload(S, C) = S^2 / C
```

对于同一个 CP group 内 pack 的多条序列：

```text
per-rank memory/token constraint:
  sum(S_i / C) <= max_seqlen_per_dp_cp_rank

per-rank workload estimate:
  sum(S_i^2 / C)
```

调度器会在满足 token/memory budget 的前提下，尽量降低所有 ranks 中最大的 workload。

### 4.5 为什么每个 wave 都要填满所有 rank

一个 outer forward wave 中，每个 rank 必须执行相同数量的 forward/backward 调用。

不能出现：

```text
GPU0-1：已经进入下一个 micro-batch
GPU2-3：还在上一个 CP collective
```

否则很容易发生 pipeline 或 collective ordering 死锁。

因此调度器要求每个 wave 中所有 ranks 都有任务。

当任务不够时，它可能：

```text
1. 将多个短样本 pack 到一个局部 group；
2. 扩大某个样本的 CP size；
3. 必要时让整个 DP×CP group 共同处理一组样本。
```

所以动态 CP 的目标不是保证“短样本永远 CP1”，而是优化整个 wave 的关键路径。

### 4.6 不预先强制 pack 到 32K，会不会增加 forward 次数

这是一个很容易产生误解、但对判断 DCP 收益非常重要的问题。

直觉上可能会认为：

```text
静态 CP4：
  先将样本 pack 成两个 32K micro-batch
  -> 只需要 2 次 forward

动态 CP：
  拆出很多 CP1、CP2 小任务
  -> 可能需要 4 次甚至更多 forward
```

但这里应该统计的是：

```text
每张 GPU 顺序执行多少个 outer forward wave
```

而不是：

```text
整个系统里一共有多少个 local CP group
```

同一个 wave 中的多个 local CP groups 是并发执行的，不是串行执行的。

假设：

```text
4 张 GPU
每张 GPU 的 sequence budget = 8K
总训练数据 = 64K tokens
```

无论怎样切 CP，一个 wave 的理论总 token 容量都是 32K：

```text
一个 CP4：
  1 group × 4 ranks × 8K = 32K

两个 CP2：
  2 groups × 2 ranks × 8K = 32K

四个 CP1：
  4 groups × 1 rank × 8K = 32K
```

因此理想的最少 wave 数都是：

```text
64K / 32K = 2 waves
```

静态 CP4 可能是：

```text
Wave 1：GPU0-3 共同处理第一个 32K pack
Wave 2：GPU0-3 共同处理第二个 32K pack

每张 GPU 执行 2 次 forward。
```

动态 CP 可能是：

```text
Wave 1：
  GPU0：8K pack，CP1
  GPU1：8K pack，CP1
  GPU2：8K pack，CP1
  GPU3：8K pack，CP1

Wave 2：
  GPU0：8K pack，CP1
  GPU1：8K pack，CP1
  GPU2：8K pack，CP1
  GPU3：8K pack，CP1

每张 GPU 同样只执行 2 次 forward。
```

虽然一个动态 wave 中存在四个 CP1 forward，但它们由四张 GPU 同时执行。

从单张 GPU 和 wall-clock 关键路径看，这仍然是一次 wave，不是顺序执行四次 forward。

#### local CP group 内也可以继续 packing

CP1 并不表示一张 GPU 一次只能处理一条样本。例如：

```text
GPU0：3K + 2K + 1K + 1K = 7K，CP1 packed
```

CP2 group 同样可以 pack 多条样本：

```text
GPU0-1：12K + 4K = 16K，CP2 packed

每 rank local tokens：
  16K / 2 = 8K
```

所以 DCP 调度器仍然会尽量填满每个 rank 的 per-rank budget，并不会天然让每条短样本产生一个
独立的 outer wave。

#### 一个相同 forward 次数的例子

假设有 8 条 8K 样本，总共 64K。

静态 CP4：

```text
Wave 1：4 × 8K 组成 32K pack，使用 CP4
Wave 2：4 × 8K 组成 32K pack，使用 CP4
```

动态 CP：

```text
Wave 1：四张 GPU 分别处理一条 8K，形成四个 CP1
Wave 2：四张 GPU 分别处理一条 8K，形成四个 CP1
```

两种方式都是每张 GPU 两次 forward。

每个 wave 的有效 attention 计算也近似相同。

静态 CP4 每 rank workload：

```text
4 × 8K^2 / 4 = 64K^2
```

动态 CP1 每 rank workload：

```text
8K^2 = 64K^2
```

主要区别是：

```text
静态 CP4：每条短序列仍有四卡 CP 通信
动态 CP1：没有 CP 通信
```

#### DCP 确实可能产生更多 wave

上面说的是理想 packing。真实调度仍可能出现：

```text
静态 CP4：2 waves
动态 CP：3 或 4 waves
```

可能原因包括：

```text
1. 样本长度组合造成 per-rank capacity 碎片；
2. CP size 只能选择 1、2、4 等离散值；
3. 独立样本数不足，无法填满所有 ranks；
4. 调度器为了平衡 sum(S_i^2 / CP)，不再向当前 wave 塞入样本；
5. THD alignment、padding 或其他模型约束；
6. 默认贪心调度器没有找到全局最优 packing。
```

更多 wave 会带来：

```text
更多 model forward/backward 调用；
更多 kernel launch；
更多 Python 和 pipeline schedule 开销；
更小的 GEMM/attention shape 可能降低 kernel 效率。
```

所以 DCP 是否更快，最终取决于：

```text
DCP 节省的 CP 通信
  + rank 等待时间减少
  + padding 减少

是否大于

额外 wave 开销
  + 小 kernel 效率损失
  + 调度和输出恢复开销
```

即使 DCP wave 更多，也不一定必然更慢。例如：

```text
静态 CP4：
  2 waves × 100 ms = 200 ms

动态 CP：
  3 waves × 45 ms = 135 ms
```

但也完全可能出现负收益：

```text
动态 CP：
  4 waves × 60 ms = 240 ms
```

因此不能只看 forward 次数或 token 数，真正应该比较的是：

```text
sum(每个 outer wave 的关键路径时间)
```

对于“静态 CP4 已经能完美 pack 成少量 32K micro-batches”的场景，建议重点统计：

```text
static_num_microbatch_waves
dynamic_num_microbatch_waves
每个 wave 的有效/padded tokens
每个 wave 的 sum(S_i^2)
每个 wave 的 wall time
CP NCCL 时间
DCP scheduler 时间
```

更准确的结论是：

```text
不预先强制 pack 成 32K，并不意味着 DCP 一定增加 forward 次数；
DCP 可以在一个 wave 内并发形成多个 CP1/CP2 pack，达到同样的 32K 聚合容量。

但如果实际 schedule 产生了更多 outer waves，就必须确认通信和负载均衡收益是否足以覆盖新增开销。
```

## 5. 10 条样本、4 张卡的一次训练示例

假设一次 optimizer step 要训练下面 10 条样本：

```text
31K, 12K, 8K, 7K, 5K, 4K, 3K, 2K, 1K, 1K
```

允许梯度累加，每张 GPU 的 sequence budget 是 8K。

下面只是一种便于理解的可能调度，不代表实际贪心调度器一定给出完全相同的组合。

### Wave 1：长样本

```text
GPU0-3：31K sample，CP4
```

这条样本确实需要四张卡。

动态 CP 在这一轮没有优势，基本等价于静态 CP4。

如果还有一条 1K 样本能在 per-rank budget 和 workload 限制下与 31K 样本共同 packing，
调度器也可能把它放进同一个 CP4 group，避免额外产生一个 wave。

### Wave 2：中等和短样本混合

```text
GPU0-1：12K sample，CP2
GPU2：8K sample，CP1
GPU3：5K + 3K，CP1 packed
```

四张卡都有工作，但只有 GPU0、GPU1 之间发生 CP 通信。

### Wave 3：剩余短样本

一种可能是：

```text
GPU0：7K，CP1
GPU1：4K + 2K + 1K + 1K，CP1 packed
GPU2-3：根据剩余任务和负载决定重新 packing 或扩大局部 CP
```

如果独立任务不足以让四张卡都有工作，调度器可能扩大某些 group，而不是允许部分 rank
完全跳过这个 wave。

### 梯度累加与更新

三个 wave 都执行 forward/backward：

```text
Wave 1 gradients
    +
Wave 2 gradients
    +
Wave 3 gradients
    ->
DP×CP gradient finalize / normalization
    ->
optimizer.step()
```

因此动态 CP 不改变“一批 10 条样本训练一次”的语义，只改变这些样本如何在四张卡和多个
gradient-accumulation micro-batches 之间安排。

## 6. verl PR #6555 的核心代码改动



### 6.1 新增 TensorDict 调度适配层

主要文件：

```text
verl/utils/dynamic_cp_scheduler.py
```

它做的事情包括：

```text
1. 从 jagged NestedTensor 读取每条样本长度；
2. 调用 MCore DefaultDynamicCPScheduler；
3. 得到每个 wave、每个 DP×CP rank 的 sample IDs；
4. 每个 rank 从完整 TensorDict 中选出自己的样本；
5. 推导 local_cp_size 和 local CP rank；
6. 构造本地 padding mask 和真实 local token 数；
7. 记录 sample IDs 和 group leader，供输出恢复使用。
```

关键 metadata：

```text
local_cp_size
_dcp_sample_ids
_dcp_group_leader
_dcp_padding_mask
_dcp_local_num_tokens
```



### 6.2 替换 Megatron micro-batch preparation

主要文件：

```text
verl/workers/engine/megatron/transformer_impl.py
```

普通路径：

```python
micro_batches, indices = prepare_micro_batches(...)
```

DCP 路径：

```python
scheduler = DynamicCPScheduler(...)
micro_batches = scheduler.schedule(...)
```

也就是说，DCP 不是在普通 dynamic batch 已经形成原子 32K micro-batch 后再决定 CP，
而是接管这一层 micro-batch 调度。

### 6.3 在模型初始化时创建动态 group

初始化 Megatron model parallel 时传入：

```python
dynamic_context_parallel=True
```

MCore 会为 DP×CP 平面创建 CP1、CP2、CP4 等预定义 group。

verl 自己在进入 pipeline schedule 前完成数据调度，因此 TransformerConfig 中的：

```python
dynamic_context_parallel
```

反而被设置为 `False`，避免再进入 MCore 自己的 dynamic pipeline scheduler。

可以理解为：

```text
MCore 负责：
  动态 group 的创建
  默认 packing-aware 调度算法
  PackedSeqParams 中的动态 CP 支持

verl 负责：
  TensorDict 数据调度
  forward/backward 接线
  loss/output/router/MTP 语义
```



### 6.4 THD preprocessing 根据 local CP 切 token

主要文件：

```text
verl/models/mcore/util.py
```

原来使用静态：

```python
cp_size = get_context_parallel_world_size()
cp_group = get_context_parallel_group()
```

DCP 下改为：

```python
cp_size = local_cp_size
cp_group = get_dynamic_data_context_parallel_groups(
    group_size=local_cp_size
)
```

随后按照这个局部 group 做 THD sequence partition。

每条 sequence 会 pad 到大致下面的 alignment：

```text
TP × local_CP × 2
```

然后使用 zigzag 分片：

```text
rank 0：最前 chunk + 最后 chunk
rank 1：第二 chunk + 倒数第二 chunk
...
```

这样可以平衡 causal attention 中不同位置 token 的工作量。

### 6.5 PackedSeqParams 携带动态 group

当前 micro-batch 会构造：

```python
PackedSeqParams(
    local_cp_size=local_cp_size,
    cp_group=cp_group,
    ...
)
```

后续所有依赖 CP 的组件，都需要优先读取这个局部 group，不能继续依赖全局静态 CP4。

主要包括：

```text
attention
THD preprocessing/postprocessing
MTP roll_tensor
MoE router padding/replay
模型输出 gather
```



### 6.6 fused forward 传递 local CP 和 padding mask

主要文件：

```text
verl/models/mcore/model_forward.py
verl/models/mcore/model_forward_fused.py
```

新增传递：

```text
local_cp_size
router_padding_mask
MTP loss normalization factor
```

MoE router 必须知道哪些 THD rows 是为了 CP alignment 补出来的 padding，否则 aux/z loss
可能错误地按 padding-inclusive token 数归一化。

### 6.7 MTP 和 router replay 改用每个 micro-batch 的 group

主要文件：

```text
verl/models/mcore/mtp_patch.py
verl/utils/megatron/router_replay_utils.py
verl/utils/megatron/router_replay_patch.py
```

原来这些路径可能直接读取模块上的静态 `self.cp_group`。

DCP 下必须改为：

```text
优先使用 packed_seq_params.cp_group
```

否则 attention 使用 CP2，而 MTP roll 或 router replay 仍使用 CP4，会导致：

```text
token 对不齐
collective 参与者不一致
输出 shape 错误
甚至直接 deadlock
```



### 6.8 loss 和真实 token 数归一化

DCP 强制启用：

```python
calculate_per_token_loss = True
```

原因是同一个 wave 中不同 ranks：

```text
处理的样本不同；
local CP size 不同；
真实 token 数不同；
CP alignment padding 数不同。
```

每个 rank 通过 `_dcp_local_num_tokens` 报告自己的真实 local token 数。

Megatron 在整个 DP×CP 平面 finalize gradients 时对 token count 求和，避免一个 CP2 样本因为在
两张卡上出现就被重复计数两次。

### 6.9 输出只由 local CP leader 保留一份

一个 CP2 group 中的 GPU0、GPU1 共同处理同一组样本。

模型输出在 CP2 group 内恢复完整 sequence 后，只允许 group leader 保留一份：

```text
[0,1] group -> GPU0 作为 leader
[2,3] group -> GPU2 作为 leader
```

随后在整个 DP×CP group 中收集：

```text
sample_ids
loss
metrics
model_output
```

最后按原始 sample ID 顺序重建 jagged NestedTensor。

这一步对 RL 特别重要，因为 log-prob、entropy、router outputs 必须与原始 trajectory 严格对齐。

## 7. 放到完整 RL 流程中看

动态 CP 主要改变 Megatron 训练/打分 engine，不会自动改变 rollout inference engine。

整体流程可以画成：

```text
Prompt
  -> vLLM / SGLang rollout
  -> trajectories
  -> 保留每条 trajectory 的 input_ids 边界
  -> PPO batch / TensorDict
       |
       +-> actor old_log_prob / entropy forward-only
       |
       +-> reference log_prob forward-only
       |
       +-> actor update forward + backward
       |
       +-> critic/value                 当前 PR 不支持 DCP value model
  -> optimizer.step()
  -> 将新 actor 权重同步给 rollout engine
```



### 7.1 Rollout generation 基本不改

如果 rollout 使用 vLLM、SGLang 或 TensorRT-LLM：

```text
动态 CP 不参与 rollout server 的 decode/prefill 调度。
```

rollout 仍然生成每条 trajectory：

```text
prompt tokens
response tokens
response_mask
rollout_log_probs
其他 RL metadata
```

真正需要保证的是：进入 Megatron engine 前，不要把多条 trajectory 永久合并成一个失去边界的
32K `input_ids` row。

### 7.2 Actor/ref 输入分发要适配 DCP

PR #6555 依赖 verl 现有的 replicated-batch 路径：

```text
同一个 DP×CP plane 的每个 rank 都能看到相同的完整 mini-batch；
每个 rank 根据相同 schedule 在本地选择自己的 sample IDs。
```

因此不需要像 MCore 通用实现那样，再通过 input all-to-all 将样本搬到目标 rank。

代价是：

```text
完整 batch 会复制到更多 ranks；
host memory 和输入分发成本可能增加；
但选择 micro-batch 后才搬到 GPU，不要求整批数据同时常驻显存。
```



### 7.3 Actor update 路径

Actor update 执行：

```text
多个动态 CP wave
  -> forward
  -> PPO loss
  -> backward
  -> gradient accumulation
  -> DP×CP gradient finalize
  -> optimizer step
```

这一条路径通常最容易获得收益，因为训练时不需要把每个 token-level model output 全部返回给
trainer，只需要正确计算 loss 和梯度。

### 7.4 Old log-prob / reference log-prob 路径

这两条通常是 forward-only：

```text
actor compute_log_prob
reference compute_log_prob
```

DCP 同样可以减少 CP 通信和改善负载，但最终需要把不同局部 CP groups 产生的 token-level
log-prob 收集起来并恢复样本顺序。

当前 PR 使用的实现包含：

```text
local CP group 内 gather
    ->
leader 将输出 detach 到 CPU
    ->
DP×CP all_gather_object
    ->
按 sample ID 重建 NestedTensor
    ->
必要时再搬回 GPU
```

这可能成为 RL 场景的新开销，尤其是：

```text
batch 很大；
sequence 很长；
需要返回 log_probs、entropy、router routes 等大量 token-level 数据。
```

因此 SFT actor-update benchmark 的收益，不能直接等价成完整 PPO end-to-end 收益。

### 7.5 Reference model 配置必须一致

动态 CP 会改变 Megatron process-group 初始化。

如果 actor 和 reference model colocate，并共享同一套 Megatron parallel state，那么 reference
必须跟随 actor 配置：

```yaml
dynamic_context_parallel: true
max_seqlen_per_dp_cp_rank: 8192
```

即使 reference 只做 forward，也不能在同一进程里假装动态 group 没有初始化。

### 7.6 Critic/value model 当前不支持

PR 当前明确拒绝：

```text
value_model
```

所以典型 PPO 中：

```text
actor/ref 可以使用 DCP；
critic/value engine 暂时不能直接复用这条路径。
```



## 8. 性能提升具体来自哪里



### 8.1 减少不必要的 CP 通信

这是最直接的收益。

```text
CP1：没有 CP KV 通信
CP2：只在两张卡之间通信
CP4：四张卡之间通信
```

短序列从 CP4 变成 CP1/CP2 后，collective 参与者更少，通信 latency 和同步开销下降。

### 8.2 释放 ranks 并行处理其他样本

短样本使用 CP1 后，原来被绑在 CP4 中的其他 ranks 可以处理独立样本。

```text
静态 CP4：
  4 GPUs -> 1 个短 packed micro-batch

动态 CP：
  4 GPUs -> 最多 4 个 CP1 packed micro-batches
```

这通常比单纯减少 padding 更重要。

### 8.3 减少 DP/CP straggler

普通 token packing 只保证每个 micro-batch token 总数相近，但 attention workload 与
`sum(S_i^2)` 更相关。

DCP 使用 `S^2 / CP` 的估算做 packing 和 rank 分配，可以减少：

```text
某些 ranks 很早完成；
某些 ranks 仍在处理长序列；
快 ranks 在 gradient sync 或 pipeline 边界等待。
```



### 8.4 减少 THD alignment padding

每条 sequence 的 THD alignment 与 local CP size 有关：

```text
alignment ~ TP × CP × 2
```

CP4 改成 CP1 后，短序列需要补齐的 alignment 通常更小。

不过对于 1K～32K 的序列，这往往是次要收益，主要收益仍是通信和并行调度。

### 8.5 不需要动态移动模型权重

CP 只切 sequence/activation，不切模型参数。

因此从 CP4 切到 CP1/CP2 时：

```text
不需要重分发 weights；
不需要重建 pipeline graph；
不需要重新切 optimizer state。
```

运行时只选择预创建的 process group，并根据新 group 重切 activation/token。

## 9. 什么情况下收益大

比较适合：

```text
1. 样本长度明显长尾；
2. 为少量超长样本不得不开较大的静态 CP；
3. 同一个 optimizer step 内有足够多独立的中短样本；
4. 中短样本的 CP 通信无法被计算隐藏；
5. 原始样本边界一直保留到 DCP scheduler；
6. DP×CP plane 内有足够并发任务，可以形成多个 CP1/CP2 group。
```

典型例子：

```text
少量 24K～32K 样本
大量 2K～16K 样本
静态为了最长样本配置 CP4
```



## 10. 什么情况下收益小或没有收益



### 10.1 所有独立样本本身都接近 32K

```text
input_ids.offsets().diff() = [31K, 30K, 32K, ...]
```

几乎每条都必须 CP4，动态调度基本退化成静态 CP4。

### 10.2 DCP 只看到不可拆分的 32K 原子 pack

如果进入 scheduler 前已经变成：

```text
input_ids.offsets().diff() = [32K, 32K, ...]
```

即使每个 32K 逻辑上由很多短文档组成，只要边界没有暴露给 DCP，它仍只能选择 CP4。

### 10.3 一个 optimizer step 中独立样本太少

例如只有一条 12K 样本：

```text
最低只需要 CP2
```

但另外两张卡没有其他任务。为了保证所有 ranks 执行相同 wave 数，调度器可能把它扩大成 CP4。

梯度累加允许容纳更多独立样本，因此通常有利于 DCP，但前提是 batch/sampler 确实提供了足够样本。

### 10.4 总计算被极少数超长样本支配

attention workload 与 `S^2` 相关。

例如：

```text
1 × 31K workload ~ 961
9 × 1K  workload ~   9
```

即使九条 1K 样本通过 DCP 加速很多，整体 step 仍主要耗在必须 CP4 的 31K 样本上。

### 10.5 CP 通信原本已经完全被计算隐藏

如果硬件互联很好、序列计算足够重、CP communication 已经完全 overlap，缩小 CP 的收益会变小。

### 10.6 forward-only 输出收集抵消收益

actor/ref log-prob 阶段可能产生大量 token-level output。

如果 CPU staging 和 `all_gather_object` 成为瓶颈，DCP 在模型 forward 中省下的时间可能被输出恢复抵消。

## 11. PR 报告的性能结果应怎样理解

PR #6555 给出的 Qwen3-30B-A3B SFT benchmark：

```text
TP1 / PP1 / EP8 / static CP4 topology
BF16
max_seqlen_per_dp_cp_rank = 4096
序列长度：1K～16K 长尾

Fixed CP4：209.97K tok/s
Dynamic CP：337.16K tok/s
吞吐提升：60.6%

Fixed CP4 loss mean：0.02104872
Dynamic CP loss mean：0.02105354
loss mean 相对差：0.0229%
```

吞吐 `+60.6%` 等价于约 `1.606x`。

处理相同 token 数时，step time 下降约：

```text
1 - 209.97 / 337.16 = 37.7%
```

这个 workload 很适合 DCP：

```text
16K 需要 CP4；
4K 以下可以 CP1；
4K～8K 可以 CP2；
长度分布明显长尾。
```

但它主要是 SFT forward/backward benchmark，并没有完整覆盖 RL 中：

```text
rollout generation
old_log_prob
reference log_prob
reward/value
advantage
actor update
权重同步
```

因此它证明了 Megatron 训练 kernel 和调度层有很大潜力，但不能直接认为 PPO 端到端一定提升 60%。

## 12. 当前限制

PR 文档明确列出的限制包括：

```text
不支持 FP8；
不支持 multimodal；
不支持 value model；
不支持 distillation；
不支持 virtual pipeline parallelism；
router replay R2/R3 要求 moe_router_fusion=False；
fused linear cross entropy 只支持统一的 scalar temperature。
```

此外，DCP 要求：

```text
use_remove_padding=True
输入为 THD / jagged NestedTensor
max_seqlen_per_dp_cp_rank 为正整数
DP×CP world size 至少为 2 且为偶数
actor/ref 等 colocated engines 一致初始化动态 groups
```



## 13. 非 2 次幂 DP×CP 的源码风险

当前 PR 声称偶数、非 2 次幂的 DP×CP layout 也可以使用，例如 6、14。

但从依赖的 MCore `d2e7ec5b` 源码看，这里存在一个值得继续验证的边界问题。

例如：

```text
DP×CP = 6
```

初始化出的局部动态 groups 主要是：

```text
CP1
CP2
full CP6
```

中间未必存在 CP4 group。

但调度器对长度位于：

```text
2M < S <= 4M
```

的序列会计算出最低 CP4，后续查询 CP4 group 时可能失败。

现有 verl distributed test 主要覆盖四卡 CP1/CP2，MCore 的非 2 次幂测试也没有完整覆盖所有中间
group lookup 情况。

在这一边界完全修正前，更稳妥的生产配置是：

```text
让 DP×CP world size 为 2 的幂。
```



## 14. 推荐配置方法

假设静态配置为：

```text
static CP = 4
最大 packed sequence = 32K
```

那么每 rank budget 应该是：

```text
32K / 4 = 8K
```

配置大致为：

```yaml
actor_rollout_ref:
  actor:
    megatron:
      context_parallel_size: 4
      dynamic_context_parallel: true
      max_seqlen_per_dp_cp_rank: 8192
      use_remove_padding: true
```

reference model 使用 Megatron 且与 actor colocate 时，需要保持一致：

```yaml
actor_rollout_ref:
  ref:
    megatron:
      context_parallel_size: 4
      dynamic_context_parallel: true
      max_seqlen_per_dp_cp_rank: 8192
```

不要把 `max_seqlen_per_dp_cp_rank` 写成 32768。

它表示的是：

```text
每一个 DP×CP rank 的 sequence budget
```

而不是：

```text
整个静态 CP4 group 的总 packed sequence budget
```



## 15. 实际验证时建议打印什么



### 15.1 确认 scheduler 看到的是独立样本

```python
print(input_ids.offsets().diff().tolist())
```

期望看到：

```text
[31744, 12288, 8192, 7168, 5120, ...]
```

而不是统一的：

```text
[32768, 32768, ...]
```



### 15.2 打印每个 wave 的 rank assignment

建议记录：

```text
global step
micro-batch wave index
DP×CP rank
sample IDs
sample lengths
local_cp_size
local valid tokens
local padded tokens
group leader
```

期望能观察到类似：

```text
wave 0:
  rank0 ids=[0] local_cp=4
  rank1 ids=[0] local_cp=4
  rank2 ids=[0] local_cp=4
  rank3 ids=[0] local_cp=4

wave 1:
  rank0 ids=[1]   local_cp=2
  rank1 ids=[1]   local_cp=2
  rank2 ids=[2]   local_cp=1
  rank3 ids=[3,4] local_cp=1
```



### 15.3 分阶段统计 RL 性能

不要只看完整 step time，至少拆分：

```text
rollout generation time
actor old_log_prob time
reference log_prob time
critic/value time
actor update time
DCP scheduling time
forward-only output collection time
optimizer time
weight sync time
```

吞吐同时统计：

```text
valid tokens/s
response tokens/s
samples/s
```

并保证静态 CP 和 DCP 使用：

```text
相同 checkpoint；
相同样本及样本顺序；
相同 global batch；
相同 TP/PP/EP/CP topology；
相同 per-rank token budget；
相同 optimizer 和 recompute；
相同 loss mask。
```



## 16. 最后总结

可以把动态 CP 记成下面这张图：

```text
固定 4 张 GPU、固定一套模型参数

遇到 31K：
  [ GPU0 GPU1 GPU2 GPU3 ] -> CP4

遇到两个 12K：
  [ GPU0 GPU1 ] [ GPU2 GPU3 ] -> 两个 CP2

遇到很多短样本：
  [ GPU0 ] [ GPU1 ] [ GPU2 ] [ GPU3 ] -> 四个 CP1

多个 wave 做梯度累加
  -> 全局 token-aware gradient finalize
  -> 一次 optimizer step
```

它的核心收益是：

```text
减少短序列不必要的 CP 通信；
让释放出来的 ranks 并发处理其他样本；
按照 sequence^2 workload 改善负载均衡；
减少 THD alignment padding；
不需要动态搬运模型权重。
```

它最重要的使用前提是：

```text
DCP scheduler 必须看到原始独立样本边界，并接管 micro-batch packing。
```

如果所有数据在进入 DCP 前已经被固化成不可拆分的 32K rows，那么调度器看到的每条数据都需要
CP4，动态 CP 自然没有发挥空间。

对于 RL，最需要继续实测的是：

```text
actor update 能获得多少收益；
old/ref log-prob 的输出收集会抵消多少收益；
完整 PPO step 中 DCP 阶段占多少比例；
长样本是否已经支配绝大部分 attention workload。
```
