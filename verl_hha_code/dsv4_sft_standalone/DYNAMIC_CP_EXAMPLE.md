# DeepSeek-V4：先学静态 Sliding-Window CP，再学 Dynamic CP

本文只解释 DSv4 的 Context Parallel（CP）训练 forward。阅读顺序刻意由简单到复杂：

```text
第一阶段：pure sliding-window（W layer）
  只理解 contiguous 切分、左边界 P2P、Q/KV projection 和 unfused CSA

第二阶段：先 HCA、再 CSA
  HCA 先增加 compressor；CSA 再在 HCA 思路上增加 learned indexer

第三阶段：Dynamic CP
  Attention 算法不变，只把固定 CP group 换成本 microbatch 的 runtime group
```

第一次阅读时，先完整看完第 2、3 节。没有理解 pure sliding-window 以前，不建议先研究
compressor、indexer 或 Dynamic scheduler。

对应程序：

- 静态 CP：[`train_dsv4_sft.py`](./train_dsv4_sft.py)，入口
  [`run_cp2.sh`](./run_cp2.sh)；
- Dynamic CP：[`train_dsv4_dynamic_cp_sft.py`](./train_dsv4_dynamic_cp_sft.py)，入口
  [`run_dynamic_cp.sh`](./run_dynamic_cp.sh)；
- 完整模型结构：[`DSV4_FORWARD_ANALYSIS.md`](./DSV4_FORWARD_ANALYSIS.md)。

## 1. 先记住核心逻辑

### 1.1 Pure sliding-window CP 的核心

以中间的 CP rank 为例，它只做六件事：

```text
1. 本 rank 只保存自己负责的连续 token hidden
2. 从左邻接收一小段 boundary_hidden，同时把自己的尾部发给右邻
3. Q 只由 local_hidden 生成
4. KV 由 cat(boundary_hidden, local_hidden) 生成
5. 分别按 local 起点和 boundary 起点对 Q/KV 应用 unfused RoPE
6. unfused CSA 让每个本地 Q 只访问合法的 causal sliding window
```

先统一 boundary P2P 的宽度定义。MCore 不会分别为 sliding window 和 compressor 通信，
而是先取二者需求的最大值：

```python
d_comp = 0  # pure sliding-window W layer 没有 compressor
d_window = max(csa_window_size, d_comp)
```

本节假设 `csa_window_size=64`，因此：

```text
d_window = max(64, 0) = 64
```

每个有右邻的 rank 一次固定发送自己最后 64 行 hidden states。这里先把它称为本层的“最大
P2P boundary 窗口”：它是多个使用方所需宽度的最大值，并不是额外发送多份数据。后面的
HCA/H layer 仍使用同一个公式，只是把 `d_comp` 从 0 改为 128，于是单次 P2P 扩大到 128。

最小代码主线：

```python
# 1. 邻居 P2P：W layer 的 d_window=max(64,0)=64，只通信一次
boundary_hidden = exchange_cp_boundary_hidden(
    local_hidden,
    compress_ratio=0,
    csa_window_size=64,
    cp_group=cp_group,
)

# 2. Q 只处理本地 token（这里用 q_proj 简写真实的 down/up projection + norm）
q = q_proj(local_hidden)

# 3. KV 同时处理左边界和本地 token
kv_input = torch.cat([boundary_hidden, local_hidden], dim=0)
kv = kv_norm(kv_proj(kv_input))

# 4. unfused RoPE：Q 从本地起点开始，KV 还要覆盖左边界
local_start = cp_rank * local_rows
query = apply_thd_cp_local_rope_unfused(
    q,
    ...,
    global_start=local_start,
)
kv = apply_thd_cp_local_rope_unfused(
    kv,
    ...,
    global_start=local_start - boundary_rows,
)

# RoPE 完成后，按原来的拼接点拆出 boundary/local KV
boundary_kv, local_kv = kv[:boundary_rows], kv[boundary_rows:]

# 5. pure sliding-window W layer 没有 compressed KV
kv_full = torch.cat([boundary_kv, local_kv], dim=0)

# 6. topk_idxs 在这里就是稀疏表示的 causal sliding-window mask
output = unfused_compressed_sparse_attn(query, kv_full, ..., topk_idxs, ...)
```

这里最容易漏掉的是 `KV` 的 RoPE 起点：它不是 `local_start`，而是
`local_start - boundary_rows`，因为拼接后的 KV 第一行来自左邻 boundary。对于 CP4、
每 rank 128 rows、boundary 64 rows 的 rank 1，就是 `Q` 从 128 开始、`KV` 从 64 开始。

### 1.2 先 HCA/H layer，再 CSA/C layer

CSA 和 HCA 都先完整执行 1.1 的共同前半段：

```python
# 每层只做一次 boundary P2P。发送宽度同时覆盖 sliding window 和 compressor：
# d_window = max(csa_window_size, d_comp)
boundary_hidden = exchange_cp_boundary_hidden(
    local_hidden,
    compress_ratio,
    csa_window_size,
    cp_group,
)
query, boundary_kv, local_kv = build_raw_qkv_and_apply_unfused_rope(
    local_hidden,
    boundary_hidden,
    ...,
)
```

这里不是先为 sliding window 发一次、再为 compressor 发一次。同一次 P2P 得到的
`boundary_hidden` 被两个分支复用：其中靠近切点的 `csa_window_size` 行供 raw sliding-window
Attention 使用；根据 block 切点选出的若干行供 compressor 补齐跨 rank block。当前实现为：

| layer | `csa_window_size` | `d_comp` | 单次 P2P 的 `d_window` |
|---|---:|---:|---:|
| W | 64 | 0 | 64 |
| HCA/H（ratio=128） | 64 | 128 | 128 |
| CSA/C（ratio=4，重叠 compressor） | 64 | 8 | 64 |

表中的宽度是每个有右邻的 rank 实际发送的固定行数，不是两种通信量相加。不同 Transformer
layer 的 hidden states 不同，因此不同层仍须各自执行自己的这一次 P2P。

下面只写它们在 pure sliding-window 之上新增的最小逻辑。

#### 1.2.1 HCA / H layer：ratio=128，没有 indexer

HCA 同样压缩 hidden，但它不训练 indexer，也不做 top-k 选择；所有因果可见的高压缩比
blocks 都加入 Attention：

```python
# 1. 同一 sequence 每 128 个连续 token 形成一个 compression block；
#    如果 block 横跨 CP 切点，就用 boundary_hidden + local_hidden 把它补完整
hidden_compact, group_ids, layout = prepare_cp_compressor_input(
    local_hidden,
    boundary_hidden,
    ratio=128,
    ...,
)

# 2. 本地生成 compressed KV，并在 CP group 内 AllGather
compressed_kv_local = attention_compressor(hidden_compact, group_ids)
compressed_kv_all = cp_all_gather(compressed_kv_local)

# 3. 最终 KV 数据源仍然由 raw window 与 compressed context 组成
kv_full = torch.cat(
    [boundary_kv, local_kv, compressed_kv_all],
    dim=0,
)

# 4. 不经过 indexer：直接加入所有因果可见的 compressed blocks
topk_idxs = build_window_plus_all_visible_compressed_indices(...)
output = unfused_compressed_sparse_attn(query, kv_full, ..., topk_idxs, ...)
```

这里的 **compression block 不是 NCCL/process group**，只是 compressor 的 token 分块。
下面完整画出“先有哪些 local/boundary hidden，再决定由谁压缩”的过程。假设一条长度为
640 的 sequence 使用 CP=4，每个 rank 持有连续的 160 个 token。

**第一步：先按 CP 划分 local hidden**

```text
global token: 0                                                          639
              |------------------------- one sequence --------------------|

CP partition:
              |------ rank 0 ------|------ rank 1 ------|------ rank 2 ------|------ rank 3 ------|
local token:         0..159               160..319             320..479             480..639
```

因此在进入 HCA attention 时，rank 0/1 最初各自只有：

```text
rank 0 local_hidden = token   0..159
rank 1 local_hidden = token 160..319
```

**第二步：先做一次固定 128-row boundary P2P**

H layer 的 `d_window=max(window_size=64, d_comp=128)=128`。每个 rank 把自己的最后 128
行发送给右邻：

```text
rank 0 的 local_hidden
|----------------------------- token 0..159 ------------------------------|
            |================ send tail: token 32..159 ====================> rank 1

P2P 之后 rank 1 手中有：

boundary_hidden = token  32..159   # 从 rank 0 收到的固定 128 rows
local_hidden    = token 160..319   # rank 1 原本持有的 160 rows

候选输入视图：  [32........................159][160.......................319]
                 <---- boundary_hidden ----><------- local_hidden -------->
```

完整的邻居交换结果是：

```text
rank 0：boundary 无有效数据，local=  0..159，向 rank 1 发送  32..159
rank 1：boundary= 32..159，local=160..319，向 rank 2 发送 192..319
rank 2：boundary=192..319，local=320..479，向 rank 3 发送 352..479
rank 3：boundary=352..479，local=480..639，没有右邻
```

rank 0 没有左邻，因此不从其他 rank 获得有效 boundary；它只需向 rank 1 发送自己的 tail。

**第三步：独立于 CP 切点，按 sequence 每 128 token 定义 HCA blocks**

```text
HCA blocks:
|-- block 0:   0..127 --|-- block 1: 128..255 --|-- block 2: 256..383 --|
|-- block 3: 384..511 --|-- block 4: 512..639 --|

CP cuts:
|------ rank 0:   0..159 ------|------ rank 1: 160..319 ------|
                               ^                              ^
                         block 1 在这里跨 rank              block 2 在这里跨 rank
```

注意：block 的边界由 sequence 起点和 `ratio=128` 决定，不会因为 CP 在 token 160 处切开
就重新从 160 开始计数。

**第四步：block 的最后一个 token 在哪个 rank，就由哪个 rank 生成该 compressed row**

```text
block 0 = token   0..127，最后 token=127 在 rank 0 -> rank 0 负责
block 1 = token 128..255，最后 token=255 在 rank 1 -> rank 1 负责
block 2 = token 256..383，最后 token=383 在 rank 2 -> rank 2 负责
block 3 = token 384..511，最后 token=511 在 rank 3 -> rank 3 负责
block 4 = token 512..639，最后 token=639 在 rank 3 -> rank 3 负责
```

以 rank 1 负责的 `block 1` 为例，它从刚才的两个输入中只选择：

```text
block 1 = token 128.....................................................255
          |-- boundary 的 128..159 --|---------- local 的 160..255 ----------|
                    32 rows                         96 rows
                       \                              /
                        +---------- 128 rows --------+
                                      |
                               HCA compressor
                                      |
                           1 个 compressed KV row
```

`boundary_hidden` 中更早的 token 32..127 对 `block 1` 无用；`local_hidden` 中 token
256..319 属于下一个 block，也不由 rank 1 在此生成。`prepare_cp_compressor_input()` 负责按
上述 ownership 规则筛选并紧凑排列。P2P 固定接收 128 行是为了覆盖任意切点位置，不需要的
rows 不会被放进当前 block。源码变量仍叫 `group_ids`，但这里的 `group` 含义是 token
block，不是通信组。

`group_ids` 可以简单理解为 compact 之后每个 block 随身携带的“原始序号”。例如 HCA
的 `ratio=128`：

```text
sequence 内的 block 0 = token   0..127 -> group_id=0 -> RoPE position=0
sequence 内的 block 1 = token 128..255 -> group_id=1 -> RoPE position=128
sequence 内的 block 2 = token 256..383 -> group_id=2 -> RoPE position=256
```

上例中 rank 1 将 boundary/local rows 整理成 `hidden_compact` 后，block 1 的数据已经脱离
原始 THD 排列；与它配套的 `group_id=1` 让 compressor 仍能计算
`position_id = group_id * ratio = 128`，给生成的 compressed KV 应用正确 RoPE。它不是 CP
rank、不是通信 group，也不是最终 `kv_full` 的 row index；block 的 rows 已由
`hidden_compact` 排列好，跨 rank 的物理 row 映射由 `seq_to_rank_row` 另外负责。对于固定容量
中的 padding slot，源码使用 `group_id=-1` 表示无效。

所以 HCA 相比 W layer 只新增：

```text
compressor
+ attention compressed KV AllGather
```

#### 1.2.2 CSA / C layer：ratio=4，再增加 learned indexer

CSA 保留 HCA 的“压缩 KV 并 AllGather”思路，但不让 query 使用所有 compressed blocks，
而是训练一个 indexer 从中选择 top-k：

```python
# 1. ratio 从 128 改为 4；仍先补齐横跨 CP 切点的 compression blocks
hidden_compact, group_ids, layout = prepare_cp_compressor_input(
    local_hidden,
    boundary_hidden,
    ratio=4,
    ...,
)

# 2. CSA 新增的 Indexer 路径：compressed K AllGather + unfused top-k
q_indexer, indexer_weights = build_local_indexer_query(local_hidden, ...)
k_indexer_local = indexer_compressor(hidden_compact.detach(), group_ids)
k_indexer_all = cp_all_gather(k_indexer_local)
compressed_topk = unfused_indexer_topk(
    q_indexer,
    indexer_weights,
    k_indexer_all,
    ...,
)

# 3. 与 HCA 相同：真正参与 Attention 的 compressed KV 也要 AllGather
compressed_kv_local = attention_compressor(hidden_compact, group_ids)
compressed_kv_all = cp_all_gather(compressed_kv_local)

# 4. 每个 query 只加入 indexer 选中的 compressed blocks
kv_full = torch.cat([boundary_kv, local_kv, compressed_kv_all], dim=0)
topk_idxs = build_window_plus_selected_compressed_indices(compressed_topk, ...)
output = unfused_compressed_sparse_attn(query, kv_full, ..., topk_idxs, ...)
```

所以 CSA 相比 HCA 又新增：

```text
indexer compressed K AllGather
+ learned top-k
```

三种 layer 的最小区别因此是：

```text
W：raw sliding window
H：raw sliding window + 所有可见的 ratio=128 compressed KV
C：raw sliding window + indexer 选中的 ratio=4 compressed KV
```

MCore 用同一个 `CompressedSparseAttention` 类表示三种情况，根据 `compress_ratio` 决定
是否创建 compressor/indexer，源码见
[`csa.py:1459`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1459)。

#### 1.2.3 进入 Dynamic CP 前，先总结三种 layer 的通信

每种 layer 都至多做一次 raw boundary P2P；HCA/CSA 再根据各自需要收集压缩后的表示：

| layer | 单次 boundary P2P | Indexer-K AllGather | Attention compressed-KV AllGather |
|---|---:|---:|---:|
| W | 64 rows | 无 | 无 |
| HCA/H | 128 rows | 无 | 1 次 |
| CSA/C | 64 rows | 1 次 | 1 次 |

HCA 的一次 AllGather 收集真正参加 Attention 的 compressed KV：

```text
local compressed KV
        -> CP AllGather
        -> global compressed KV
        -> 本 rank 的 local Q 直接访问所有因果可见 blocks
```

CSA 因为增加 learned Indexer，所以相比 HCA **额外多一次 Indexer-K AllGather**：

```text
local indexer K --第一次 AllGather--> global compressed indexer K --+
                                                                  |
local indexer Q（不通信）------------------------------------------+--> top-k IDs

local attention compressed KV --第二次 AllGather--> global attention compressed KV
                                                            |
top-k IDs ---------------------------------------------------+--> 只访问选中的 blocks
```

这两次 AllGather 收集的不是同一个 tensor：第一次收集 Indexer 用来计算 top-k 的
compressed K，第二次收集真正参与 sparse Attention 的 compressed KV。Indexer Q 始终保留
在本 rank，不需要 AllGather。两者通信的也都是压缩表示，而不是所有 raw-token K/V。

### 1.3 Dynamic CP 不会改变上述 Attention

```text
静态 CP：每个 microbatch 都使用建模时固定的 cp_group
动态 CP：scheduler 给本 microbatch 选择 CP1/CP2/CP4/... group
```

选好 group 以后，CP2/CP4 仍执行相同的 boundary P2P、projection、compressor 和 CSA；
CP1 则直接走非 CP 的普通 THD forward。

## 2. 第一阶段准备：静态 CP 怎样切输入

### 2.1 代码归属

| 标记 | 所有者 | 在 CP 流程中的职责 |
|---|---|---|
| `[standalone]` | 本目录 | 构造 synthetic SFT batch、发起训练 |
| `[verl]` | verl | Dynamic scheduler 的 TensorDict/engine 接线 |
| `[Bridge]` | Megatron-Bridge | 根据 HF DSv4 config 构建 MCore 模型 |
| `[MCore]` | Megatron-Core | THD 切分、进程组、DSv4 Attention、通信和 backward |
| `[PyTorch/NCCL]` | PyTorch/NCCL | P2P、AllGather、ReduceScatter、AllToAll |

DSv4 CP Attention、boundary P2P 和 Dynamic groups 都由 `[MCore]` 实现；verl 没有重写
这些算子。本目录的 standalone 也没有 import verl，只保留了最小训练 glue。

### 2.2 为什么是 THD contiguous

THD 可以先简单理解为：把一个 microbatch 的 token 沿 `T` 维排成一行。

假设一条 sequence 有 512 个 token，静态 CP=4：

```text
cp_rank 0: token   0..127
cp_rank 1: token 128..255
cp_rank 2: token 256..383
cp_rank 3: token 384..511
```

每个 rank 只保存 128 个 token 的 embedding/hidden activation。

`[MCore]` 正式切分 API 在
[`get_cp_slice_for_thd()`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule_utils.py#L15)：

```python
local_rows = total_tokens // cp_size
row_slice = slice(cp_rank * local_rows, (cp_rank + 1) * local_rows)
batch[key] = batch[key][row_slice]
```

standalone 的调用在
[`pack_for_contiguous_cp()`](./train_dsv4_sft.py#L235)：

```python
get_cp_slice_for_thd(
    batch,
    cp_group,
    cp_partition_mode="contiguous",
)

packed_seq_params = PackedSeqParams(
    qkv_format="thd",
    cp_partition_mode="contiguous",
    ...,
)
```

DSv4 当前只支持这种连续区间。普通 causal Attention 常用的 zigzag CP 不适用于这里，
因为 DSv4 的左边界通信假设每个 rank 拥有一个连续 token block。

### 2.3 前面的图主要是单条 sequence；真实 packed THD 还要处理 sequence 边界

前面以及第 3～5 节为了单独讲清 P2P、HCA 和 CSA，主要使用“一条长 sequence 被 CP
切开”的例子。真实 SFT packed batch 可以包含多条 sequence。核心机制不变，但必须始终
区分两种编号：

```text
global THD row：token 在整个 packed tensor 中的物理行号
sequence position：token 在自己那条 sequence 内的位置，遇到新 sequence 重新从 0 开始
```

`cu_seqlens_padded` 保存每条 sequence 在 global THD rows 中的边界。下面用一个能被 CP=4
整除的 packed batch 举例：

```text
sequence A：长度  96 -> global rows   0..95，sequence positions   0..95
sequence B：长度 224 -> global rows  96..319，sequence positions  0..223
sequence C：长度 320 -> global rows 320..639，sequence positions  0..319

cu_seqlens_padded = [0, 96, 320, 640]
total_rows = 640
local_rows = 640 / CP4 = 160
```

#### 2.3.1 contiguous CP 只切平铺后的物理 rows

```text
packed THD：
|--- A: 0..95 ---|--------------- B: 96..319 ---------------|---------------- C: 320..639 ----------------|

CP partition：
|---- rank 0: global 0..159 ----|---- rank 1: 160..319 ----|---- rank 2: 320..479 ----|---- rank 3: 480..639 ----|

rank 0 local = A position   0..95 + B position   0..63
rank 1 local =                     B position  64..223
rank 2 local =                                           C position   0..159
rank 3 local =                                                               C position 160..319
```

所以一个 rank 可能同时持有多条 sequence 的片段；一个 CP 切点也可能正好落在 sequence
边界上，或者落在某条 sequence 中间。切分 API 只切 `tokens/labels/loss_mask/...`，完整的
`cu_seqlens_padded=[0,96,320,640]` 仍保留在每个 CP rank 上，不能切成 rank-local
`cu_seqlens`，也不能简单减去本 rank 起点。

源码：[`get_cp_slice_for_thd()`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule_utils.py#L15)。

#### 2.3.2 boundary P2P 仍发固定 shape，但不保证全是同一条 sequence

以 W layer 的 64-row P2P 为例：通信本身只取物理上的最后 64 rows，不解析
`cu_seqlens_padded`。

```text
rank 0 -> rank 1：global  96..159 = B position   0..63
rank 1 -> rank 2：global 256..319 = B position 160..223
rank 2 -> rank 3：global 416..479 = C position  96..159
```

收到以后：

```text
rank 1 的 local 从 B position 64 开始：
    boundary 中的 B position 0..63 有效，可以作为 sliding window 左上下文

rank 2 的 local 从新 sequence C position 0 开始：
    boundary 全属于上一条 sequence B，必须全部忽略

rank 3 的 local 从 C position 160 开始：
    boundary 中的 C position 96..159 有效
```

因此“固定 P2P 收到了 64/128 rows”不等于“这些 rows 都能参与 Attention”。P2P 只搬运
候选数据，最终由 sequence-aware indices 过滤。

#### 2.3.3 sliding window 和 RoPE 都必须在每条 sequence 内重置

对于每个 local query，layout kernel 先根据 `cu_seqlens_padded` 找到它所属 sequence 的
`seq_start`，再执行：

```python
window_start = max(global_q - window_size + 1, seq_start)
```

所以 C 的 position 0 不可能访问物理上紧挨着它的 B 尾部。源码：
[`build_attention_indices()`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_layout_kernels.py#L755)。

RoPE 也不能直接使用 global THD row，而要恢复 sequence-relative position：

```python
sequence_id = bucketize(global_row, cu_seqlens_padded)
position_id = global_row - cu_seqlens_padded[sequence_id]
```

因此 A/B/C 都从 RoPE position 0 开始。boundary KV 使用同一个换算，所以 rank 2 收到的 B
尾部虽然可以被投影和做 RoPE，最后仍不会进入 C query 的合法 indices。源码：
[`_thd_cp_position_ids()`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L24)。

#### 2.3.4 HCA/CSA compression blocks 也必须按 sequence 分开

HCA `ratio=128` 时，不能因为 packed tensor 物理连续，就把 A 的尾部和 B 的开头凑成一个
128-token block。上例应分别计算：

```text
sequence A，len= 96：floor( 96/128)=0 个完整 block

sequence B，len=224：
    B block 0 = B position 0..127
    最后 token B127 位于 rank 1 -> rank 1 负责
    B position 128..223 是不足 128 的 tail，不压缩

sequence C，len=320：
    C block 0 = C position   0..127 -> rank 2 负责
    C block 1 = C position 128..255 -> 跨 rank 2/3，由 rank 3 负责
    C position 256..319 是 tail，不压缩
```

这里 B block 0 和 C block 0 的 `group_id` 都是 0，因为 `group_ids` 在每条 sequence 内
分别从 0 开始。`prepare_cp_compressor_input()` 使用 `cu_seqlens_padded` 筛选 boundary/local
rows，保证 compressor 不跨 sequence；不足一个 ratio 的尾部留给 raw sliding window。

#### 2.3.5 packed 静态 CP 的几个硬约束

```text
1. contiguous CP 要求 total padded THD rows 能被 cp_size 整除，否则无法等长切分。
2. 每个 rank 的 local_rows 必须 >= d_window；H layer 通常至少需要 128 rows。
3. 所有 CP ranks 必须持有一致的全局 cu_seqlens_padded metadata。
4. sequence 数据 rows 被切分，但 cu_seqlens_padded 不切、不改成本地坐标。
5. padding rows 要放在 sequence 尾部，并通过 loss_mask 排除训练 loss。
6. P2P 收到的是固定大小的物理候选窗口；是否同 sequence 由后续 indices 决定。
7. compressor 每条 sequence 单独分 block，不能跨 packed sequence 边界凑满 ratio。
```

compressed K/KV AllGather 还需要每个 rank 输出相同的固定容量，因此本地无效 slots 会 padding；
AllGather 后再由 `seq_to_rank_row` 把每条 sequence 的逻辑 compressed block 映射到 rank-major
buffer。这个 metadata 布局不会改变“任何 query 只能访问自己 sequence”的原则。

## 3. 第一阶段：pure sliding-window W layer 的完整代码流程

这一节只讨论：

```text
compress_ratio = 0
apply_dsa_kernel_fusion = False
csa_window_size = 64
CP = 4
单条 sequence 长度 = 512
每 rank local token 数 = 128
```

这里继续使用单条 sequence 是为了只展示 W layer 的 forward 主线；多 sequence packed 时
P2P 调用完全相同，额外的 sequence 边界处理已经集中说明在第 2.3 节。

只跟踪 `cp_rank=1`，它负责 token 128..255。

### 3.1 进入 Attention 时各 rank 手里有什么

Embedding 已经在本地 token IDs 上完成，因此：

```text
rank 0 hidden_states = token   0..127，shape=[128,1,H]
rank 1 hidden_states = token 128..255，shape=[128,1,H]
rank 2 hidden_states = token 256..383，shape=[128,1,H]
rank 3 hidden_states = token 384..511，shape=[128,1,H]
```

此时 rank 1 没有 rank 0 的 hidden，也还没有计算 Q/K/V。

### 3.2 Attention.forward 发起 boundary P2P

源码：
[`deepseek_v4_hybrid_attention.py:256`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L256)。

```python
cp_group = self.pg_collection.cp
cp_size = cp_group.size()

use_thd_cp = cp_size > 1 and packed_seq_params.qkv_format == "thd"
if use_thd_cp:
    boundary_hidden = exchange_cp_boundary_hidden(
        hidden_states,
        self._dsv4_compress_ratio,
        self.config.csa_window_size,
        cp_group,
    )
```

W layer 中：

```text
compress_ratio = 0
d_comp = 0
d_window = max(csa_window_size=64, d_comp=0) = 64
```

计算代码在
[`exchange_cp_boundary_hidden()`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L190)：

```python
d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
d_window = max(int(csa_window_size), d_comp)
hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
```

### 3.3 `_LeftBoundaryExchange.forward()` 实际发送什么

源码：
[`csa_cp_utils.py:123`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L123)。

```python
boundary = tensor.new_zeros((d_window,) + tensor.shape[1:])
ops = []

if cp_rank > 0:
    ops.append(irecv(boundary, src=cp_rank - 1))

if cp_rank + 1 < cp_size:
    send_tail = tensor[-d_window:].contiguous()
    ops.append(isend(send_tail, dst=cp_rank + 1))

for request in dist.batch_isend_irecv(ops):
    request.wait()

return boundary
```

把 rank 1 的 token 范围代进去：

```text
rank 1 自己持有 token 128..255
rank 1 send_tail  = token 192..255 -> 发给 rank 2

rank 0 自己持有 token   0..127
rank 0 send_tail  = token  64..127 -> rank 1 收到
```

通信完成以后 rank 1 有两块 hidden：

```text
local hidden      = token 128..255，shape=[128,1,H]
boundary_hidden   = token  64..127，shape=[ 64,1,H]
```

CP4 的完整 P2P 关系：

```text
rank 0 token  64..127 -> rank 1
rank 1 token 192..255 -> rank 2
rank 2 token 320..383 -> rank 3
```

rank 0 没有左邻，因此返回全零 boundary；最后一个 rank 没有右邻，因此只接收不发送。
这些无效零 rows 后面不会被窗口 mask 选中。

这里是一次邻居单跳 P2P，不是 AllGather，也不是 raw-KV Ring Attention。

### 3.4 为什么只发给右边一个 rank 就够了

代码要求：

```text
local_rows >= d_window
```

当前 rank 需要的最远 raw token 不超过 `d_window`，而左邻至少持有 `d_window` 个连续
token，所以所需原始窗口一定全部在直接左邻，不会继续跨到左邻的左邻。

如果 `local_rows < d_window`，MCore 会在 P2P 前报错：

```python
if tensor.shape[0] < d_window:
    raise RuntimeError("local rows >= D_window is required")
```

### 3.5 收到 hidden 后怎样计算 Q/KV

控制流回到
[`get_query_key_value_tensors()`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L583)。

Q 只对应 rank 1 负责输出的 128 个本地 query：

```python
q_compressed, _ = self.linear_q_down_proj(hidden_states)
q, _ = self.linear_q_up_proj(q_compressed)
```

KV 则需要覆盖左边界和本地 token：

```python
boundary_rows = boundary_hidden.shape[0]                       # 64
kv_projection_input = torch.cat([boundary_hidden, hidden_states], dim=0)
kv, _ = self.linear_kv_proj(kv_projection_input)               # 192 rows
kv = self.kv_layernorm(kv)
```

真实代码在
[`deepseek_v4_hybrid_attention.py:658`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L658)
和
[`deepseek_v4_hybrid_attention.py:713`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L713)。

shape 是：

```text
Q projection input  = local hidden                  = [128,H]
KV projection input = cat(boundary, local hidden)   = [192,H]
```

为什么交换 hidden 而不是提前在 rank 0 计算好 KV？因为 HCA/CSA 的 compressor 也需要
boundary hidden。统一发送 hidden 可以同时服务 raw boundary KV 和后面的 compressor。

### 3.6 Unfused RoPE 怎样处理这两段数据

rank 1 的本地 Q 对应 token 128..255，拼起来的 KV 对应 token 64..255。因此 unfused
RoPE 必须从不同位置开始：

```python
local_start = cp_group.rank() * q.shape[0]     # 1 * 128 = 128

query = apply_thd_cp_local_rope_unfused(
    q,
    ...,
    global_start=local_start,                  # 128
)

kv = apply_thd_cp_local_rope_unfused(
    kv.unsqueeze(-2),
    ...,
    global_start=local_start - boundary_rows,  # 64
)
```

源码在
[`deepseek_v4_hybrid_attention.py:783`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L783)。

完成 projection 和 RoPE 后再按 64-row 拼接点切开：

```python
boundary_kv = kv[:boundary_rows]  # token 64..127 的 KV
key = value = kv[boundary_rows:]  # token 128..255 的本地 KV
```

这里的 `global_start` 只用来给 RoPE/packed sequence 找到正确 token 位置；它不是额外维护
的一份 KV tensor。

### 3.7 两块 KV 怎样进入 unfused CSA

`Attention.forward()` 调用 core attention：

```python
core_attention(
    query,
    key,
    value,
    ...,
    x=hidden_states,
    boundary_hidden=boundary_hidden,
    boundary_kv=boundary_kv,
)
```

源码：
[`deepseek_v4_hybrid_attention.py:315`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L315)。

`CompressedSparseAttention.forward()` 检测 THD+CP 后进入：

```python
output = self._forward_thd_cp(
    query,
    key,
    x,
    qr,
    boundary_hidden,
    boundary_kv,
    packed_seq_params,
)
```

源码：[`csa.py:1836`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1836)。

W layer 没有 compressor，所以 compressed KV 为空：

```python
kv_local = key.squeeze(-2).squeeze(1)
compressed_kv = empty
kv_full_thd = torch.cat([boundary_kv, kv_local, compressed_kv], dim=0)
```

源码：[`csa.py:2309`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2309)
与 [`csa.py:2447`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2447)。

MCore 随后构造 `topk_idxs`。在 W layer 中不要把它想复杂：它就是 causal
sliding-window mask 的稀疏写法，告诉每个本地 query 应该从 `kv_full_thd` 读取哪些
boundary/local KV；同时通过 `cu_seqlens` 阻止 packed sequences 相互访问。

最后明确走 unfused 分支：

```python
output = unfused_compressed_sparse_attn(
    query,
    kv_full_thd,
    self.attn_sink.float(),
    topk_idxs,
    self.softmax_scale,
)
```

源码：[`csa.py:2553`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2553)。

输出仍然只有 rank 1 的 128 个 query rows，不会在层尾把完整 512-token hidden gather 回来。

### 3.8 Pure sliding-window 的 backward

rank 1 使用了 rank 0 的 boundary hidden，因此 backward 必须把这部分激活梯度还给 rank 0。

```text
forward : rank 0 hidden tail    -> rank 1 boundary_hidden
backward: rank 1 grad_boundary -> rank 0 hidden tail
```

`_LeftBoundaryExchange.backward()` 的代码：

```python
if cp_rank > 0:
    isend(grad_boundary, dst=cp_rank - 1)

if cp_rank + 1 < cp_size:
    irecv(recv_grad, src=cp_rank + 1)

grad_input[-d_window:] = recv_grad
```

源码：[`csa_cp_utils.py:159`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L159)。

PyTorch 会把右邻传回的 boundary gradient 与本 rank 其他本地路径产生的 hidden gradient
自动相加。

### 3.9 到这里应该形成的理解

Pure sliding-window W layer 的 CP 通信只有：

```text
forward：向右发送 hidden tail，从左接收 boundary hidden
backward：boundary gradient 沿相反方向返回 owner
```

没有 raw K/V AllGather，没有 compressed-KV AllGather，也没有 indexer-K AllGather。

## 4. 第二阶段第一步：HCA/H layer 在 W layer 上增加压缩上下文

先看更简单的 HCA：`compress_ratio=128`，有 compressor，但没有 indexer。以下步骤全部
建立在第 3 节的 pure sliding-window 上。

### 4.1 不变的前半段

H layer 仍然先执行：

```text
boundary hidden P2P
-> Q 只投影 local hidden
-> KV 投影 cat(boundary hidden, local hidden)
-> unfused RoPE
-> 得到 boundary_kv 和 local_kv
```

H layer 的 boundary 还要满足 128-token compression block：

```text
d_comp=128
d_window=max(csa_window_size,128)
```

假设 `csa_window_size=64`，这里的 `d_window=128`：H layer **只发送一次 128-row
`boundary_hidden`**，不是一次 128-row compressor P2P 加一次 64-row sliding-window P2P。
同一份 128 rows 中，最后 64 rows 满足 raw sliding window；compressor 再从其中选择补齐
跨 CP 切点 block 所需的 rows。因此每个 rank 的 `local_rows` 也必须至少为 128。

### 4.2 第一个新增步骤：构造 compressor input

H layer 需要把同一条 sequence 中每 128 个连续 token 压成一个 compressed row。CP 切点可能
落在 compression block 中间，
所以 compressor 同样需要刚才收到的 `boundary_hidden`。

源码：[`csa.py:2332`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2332)。

```python
hidden_compact, compressed_group_ids, seq_to_rank_row = prepare_cp_compressor_input(
    x,                 # local hidden
    boundary_hidden,   # 左邻 hidden
    cu_seqlens,
    cu_seqlens_compressed,
    local_start,
    cp_size,
    ratio=128,
)
```

第一次阅读只需理解输出：

```text
hidden_compact：本 rank 负责生成的完整 128-token compression blocks 的输入
compressed_group_ids：每个 block 在各自 sequence 中的编号（源码变量名沿用 group）
seq_to_rank_row：AllGather 后恢复逻辑顺序所需的内部映射
```

后两个是 MCore layout metadata，不需要手工维护。

### 4.3 第二个新增步骤：compressed KV AllGather

每个 rank 只压缩自己负责的 compression blocks。为了让本地 query 也能访问其他 CP ranks 产生的长距离
compressed context，需要在 CP group 内 AllGather：

```python
compressed_kv_local, _ = self.compressor._forward_thd(hidden_compact, ...)

compressed_kv_rank_major = gather_from_sequence_parallel_region(
    compressed_kv_local.squeeze(1),
    group=cp_group,
)
```

源码：[`csa.py:2435`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2435)。

### 4.4 第三个新增步骤：把所有因果可见的 compressed blocks 加入 Attention

HCA 没有 indexer。最终 KV 数据源是：

```python
kv_full_thd = torch.cat(
    [
        boundary_kv,              # 左邻的原始窗口 KV
        local_kv,                 # 本地原始 KV
        compressed_kv_rank_major, # CP group 的长距离 compressed KV
    ],
    dim=0,
)
```

随后仍调用同一个 unfused sparse Attention。`topk_idxs` 包含：

```text
本地 causal sliding-window raw KV
+ 当前 query 因果可见的所有 ratio=128 compressed KV
```

### 4.5 HCA 的通信总结

```text
1. boundary hidden P2P
2. attention compressed KV AllGather
```

compressed-KV gather 使用 MCore autograd wrapper：

```text
forward  : AllGather
backward : ReduceScatter
```

这样远端 query 对某个 compressed KV 产生的梯度会回到真正生成它的 rank。

## 5. 第二阶段第二步：CSA/C layer 在 HCA 思路上增加 indexer

现在再看 `compress_ratio=4` 的 CSA。它保留第 4 节的整体结构：

```text
boundary P2P
-> prepare compressor input
-> compressed KV AllGather
-> raw window + compressed context 的 unfused sparse Attention
```

主要变化有两个：

```text
1. compression ratio 从 128 变成 4
2. 不再使用所有 compressed blocks，而是增加 learned indexer 选择 top-k
```

### 5.1 Ratio=4 的 compressor input

```python
hidden_compact, compressed_group_ids, seq_to_rank_row = prepare_cp_compressor_input(
    local_hidden,
    boundary_hidden,
    ...,
    ratio=4,
)
```

如果 `csa_window_size=64`：

```text
d_comp=8
d_window=max(64,8)=64
```

### 5.2 CSA 新增：indexer compressed K AllGather

每个 rank 为 indexer 生成本地 Q、weights 和 compressed K：

```python
q_indexer_local, weights_indexer_local = build_local_indexer_query(...)
indexer_compressed_local, _ = indexer.compressor._forward_thd(hidden_compact, ...)
```

然后收集整个 CP group 的 indexer compressed K：

```python
k_indexer_rank_major = gather_from_sequence_parallel_region(
    indexer_compressed_local.squeeze(1),
    group=cp_group,
)
```

源码：[`csa.py:2403`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2403)。

unfused indexer 为每个本地 query 选择 compressed top-k：

```python
compressed_topk, _ = compute_cp_indexer_topk(
    q_indexer_local,
    weights_indexer_local,
    k_indexer_seq_major,
    ...,
    use_fused=False,
)
```

### 5.3 与 HCA 相同：attention compressed KV AllGather

真正参与 Attention 的 compressed KV 仍由 attention compressor 生成并 AllGather：

```python
compressed_kv_local, _ = self.compressor._forward_thd(hidden_compact, ...)
compressed_kv_rank_major = gather_from_sequence_parallel_region(
    compressed_kv_local.squeeze(1),
    group=cp_group,
)
```

最终：

```python
kv_full_thd = torch.cat([boundary_kv, local_kv, compressed_kv_rank_major], dim=0)
topk_idxs = build_window_plus_selected_compressed_indices(compressed_topk, ...)
output = unfused_compressed_sparse_attn(query, kv_full_thd, ..., topk_idxs, ...)
```

### 5.4 CSA 的通信总结

```text
1. boundary hidden P2P
2. indexer compressed K AllGather
3. attention compressed KV AllGather
```

因此学习上的递进就是：HCA 先教会我们“怎样跨 CP 收集 compressed KV”；CSA 再多解决
“compressed blocks 太多时，怎样用 learned indexer 只选重要 top-k”。

## 6. 静态 CP 的完整模型 forward

到这里可以把 Attention 放回完整 Transformer layer：

```text
本地 input_ids
-> 本地 embedding
-> DSv4 layer
     -> norm/MHC
     -> W/H/C Attention（按前面对应路径）
     -> residual
     -> norm/MHC
     -> MoE router
     -> EP AllToAll dispatch
     -> local/shared experts
     -> EP AllToAll combine
     -> residual
-> final norm
-> LM head / local token loss
-> backward
-> finalize_model_grads
-> optimizer step
```

CP 与 EP 是不同通信维度：

- Attention boundary/gather 使用 CP group；
- MoE token dispatch/combine 使用 EP group；
- 参数梯度同步使用对应的 DP/DP×CP group。

层与层之间始终保留 `L_local` rows，不会每经过一层就 gather 回完整 sequence。

### 6.1 静态 W/H/C 通信对照

| Attention 类型 | Boundary P2P | Indexer K AllGather | Compressed KV AllGather |
|---|---:|---:|---:|
| W，ratio=0 | 有 | 无 | 无 |
| H，ratio=128 | 有 | 无 | 有 |
| C，ratio=4 | 有 | 有 | 有 |

## 7. 第三阶段：Dynamic CP 只新增哪几步

现在才进入 Dynamic CP。先把静态流程压缩为：

```text
固定 cp_group
-> contiguous slice
-> W/H/C Attention 在固定 group 上执行
```

Dynamic CP 变成：

```text
预创建 CP1/CP2/CP4/... groups
-> scheduler 给本 microbatch 选择 group
-> 按 runtime group contiguous slice
-> W/H/C Attention 在 runtime group 上执行
```

Attention 内部算法没有换。

## 8. Dynamic groups 为什么要预创建

`[MCore] initialize_model_parallel(dynamic_context_parallel=True)` 会在模型 forward 前创建
多种 NCCL groups。源码：
[`create_dynamic_dp_cp_groups()`](./third_party/Megatron-LM/megatron/core/parallel_state.py#L421)。

对 8 个 DP×CP ranks：

```text
CP1: [0] [1] [2] [3] [4] [5] [6] [7]
CP2: [0,1] [2,3] [4,5] [6,7]
CP4: [0,1,2,3] [4,5,6,7]
CP8: [0,1,2,3,4,5,6,7]
```

一个 scheduled microbatch 可以从这些预创建 groups 中选出互不重叠的一组来覆盖 8 个 ranks。
例如下面就是合法布局：

```text
sample A，runtime CP2：rank [0,1]       -> 两个 rank 各处理 A 的连续一半
sample B，runtime CP1：rank [2]         -> rank 2 独立处理完整 B
sample C，runtime CP1：rank [3]         -> rank 3 独立处理完整 C
sample D，runtime CP4：rank [4,5,6,7]   -> 四个 rank 各处理 D 的连续四分之一

8 ranks 的本轮布局：
|---- CP2 ----|-- CP1 --|-- CP1 --|------------ CP4 ------------|
|    0,1      |    2    |    3    |           4,5,6,7           |
```

同一个 runtime CP group 内的 ranks 得到相同 `sample_ids` 和同一份 packed metadata，再按
`local_cp_size` 切开 THD rows；CP1 直接走非 CP forward，不执行跨 rank boundary P2P 或
AllGather。各 group 之间处理不同样本，Attention collective 也只发生在各自 group 内。

当前 group pool 要求 power-of-two 且自然对齐，因此 `[0,1]` 和 `[4,5,6,7]` 可用，但不能
临时拼出 `[1,2]` 的 CP2 或 `[2,3,4,5]` 的 CP4。这里的 0..7 是同一个 DP×CP plane 内的
rank 编号；存在 TP/PP/EP 时，还必须保持其他模型并行坐标兼容，不能按全局 GPU 编号任意
组队。`DefaultDynamicCPScheduler` 会根据长度和负载自动决定实际排列，不保证每轮恰好采用
上面这个示意顺序。

不能在每个 forward 中临时、无序地创建 NCCL groups，否则不同 ranks 的 group 创建顺序
不一致时容易死锁，而且 communicator 初始化本身也有开销。

### 8.1 只有一个模型：为什么初始化 DCP=True，TransformerConfig 里却是 False

这里没有“静态模型”和“动态模型”两份模型。模型只构建一次，所有 runtime CP sizes 共用
同一套参数。CP 改变的是一条 sequence 的 activation/token rows 由几个 ranks 协作处理，
不会因为 CP1 切到 CP4 就重新构建或重新加载模型参数。

容易混淆是因为配置分成三个层次：

| 层次 | 关键配置 | 职责 |
|---|---|---|
| `parallel_state` 初始化 | `context_parallel_size=1` | 创建常规/default CP group；这里的默认宽度是 1 |
| `parallel_state` 初始化 | `dynamic_context_parallel=True` | 额外预创建可复用的 CP1/CP2/CP4/CP8 groups |
| provider 的 `TransformerConfig` | `dynamic_context_parallel=False` | 不把这份模型配置标成由原生 MCore DCP 模式接管；verl engine 自己编排 |
| 每个 microbatch 的 metadata | `PackedSeqParams.cp_group`、`local_cp_size` | 告诉同一个 Attention 本次真正使用哪个 runtime group |

standalone 的初始化代码实际上是：

```python
# 第一个层次：初始化 distributed process groups
parallel_state.initialize_model_parallel(
    context_parallel_size=1,
    dynamic_context_parallel=True,
)
```

它的含义不是“模型只能 CP1”，而是：常规 CP 默认 group 的大小为 1，同时在 DP×CP plane
上额外准备好不同大小的 runtime groups。

provider 构造同一个模型时则设置：

```python
# 第二个层次：TransformerConfig
context_parallel_size = 1
dynamic_context_parallel = False
sequence_packing_scheduler = "default_dynamic_cp"
```

这里的 `False` 也不是关闭整个功能，而且严格来说 scheduler 从来不在 Transformer layer
内部运行。原生 MCore 的调度入口位于 training/data stack；verl 不使用它的 `training.py`
主循环，而是在自己的 engine 进入 pipeline forward/backward 之前显式调用同一个 MCore
scheduler 类。真正的 runtime 选择再通过 metadata 传入：

```text
同一个 DSv4 model / 同一套 weights
                       |
microbatch A：PackedSeqParams.cp_group=[0]       -> 本次 Attention 用 CP1
microbatch B：PackedSeqParams.cp_group=[0,1]     -> 本次 Attention 用 CP2
microbatch C：PackedSeqParams.cp_group=[4,5,6,7] -> 本次 Attention 用 CP4
```

Attention.forward 会暂时用 `PackedSeqParams.cp_group` 替换 default group，完成本次 forward
后再恢复。所谓“default CP1”只是没有 runtime override 时的默认通信组，不是另一份模型，
也不是把 runtime CP 限死为 1。

verl 对应设置在
[`transformer_impl.py:307`](../../verl/workers/engine/megatron/transformer_impl.py#L307)。

### 8.2 Dynamic CP 只能到 CP8 吗

不是。可用上限不是固定 8，而是由一个 DP×CP plane 中可参与 Dynamic CP 的 rank 数 `N`
决定。MCore 默认按 power-of-two 预创建自然对齐的子组，并把整个 plane 也作为最大 group：

```text
min_dynamic_context_parallel_size=1，N=8：
    CP1 / CP2 / CP4 / CP8

min_dynamic_context_parallel_size=1，N=16：
    CP1 / CP2 / CP4 / CP8 / CP16

min_dynamic_context_parallel_size=1，N=32：
    CP1 / CP2 / CP4 / CP8 / CP16 / CP32
```

其中小于 `N` 的 groups 由 `create_dynamic_dp_cp_groups()` 创建；大小正好等于 `N` 时，
MCore 直接复用完整的 DP-with-CP group。当前 8 卡 standalone 的 `N=8`，所以文档只画到
CP8，并不是 DCP 功能的硬编码上限。

用户可控制的主要是：

```text
N：DP×CP plane 的总 ranks，决定理论最大 runtime CP size
min_dynamic_context_parallel_size：最小允许的 group size，默认 1
max_seqlen_per_dp_cp_rank：每 rank 的 token workload 目标，影响 scheduler 给样本选择多大 CP
sequence lengths / packing：共同决定每个 scheduled microbatch 的实际 CP size
```

例如 `N=16` 且最小值为 1 时，可以取得：

```python
cp16_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=16)
```

通常不需要用户逐条指定 `CP16`；scheduler 先近似计算：

```text
minimum CP = next_power_of_two(sequence_length / max_seqlen_per_dp_cp_rank)
```

再结合 packing 和负载均衡选择实际 group，它有时也会选大于 minimum 的 group。对当前
DSv4 还必须同时满足 `local_rows >= d_window`：H layer 的 `d_window=128`，所以不能为了追求
更大的 CP 把每 rank 的实际 THD rows 切到 128 以下。

对本 standalone/verl 配置，可以把 `context_parallel_size=1` 理解为推荐的 default group；
MCore 的 group builder 本质上工作在整个 DP×CP plane 上，并不是用这个 1 限制最大 runtime
CP。当前示例保持它为 1，不建议在复现实验时同时改 default CP 和 Dynamic CP，以免把两种
配置层次混在一起。

### 8.3 如果由原生 MCore 调度，它是在什么时候运行的

`dynamic_context_parallel=True` 本身不会在 Attention.forward 中突然执行一段调度算法。
MCore 的配置初始化会先自动补上：

```python
if config.dynamic_context_parallel:
    config.sequence_packing_scheduler = "default_dynamic_cp"
```

源码：[`model_parallel_config.py:502`](./third_party/Megatron-LM/megatron/core/model_parallel_config.py#L502)。

原生 MCore 训练循环在**每个 optimizer iteration 的整组 forward/backward 开始之前**检查
这个 scheduler 名称：

```python
if config.sequence_packing_scheduler is not None:
    data_iterator, num_microbatches, ... = wrap_data_iterator(
        data_iterator,
        config,
        get_num_microbatches(),
    )

losses = forward_backward_func(
    data_iterator=data_iterator,
    num_microbatches=num_microbatches,
    ...,
)
```

源码：[`training.py:2718`](./third_party/Megatron-LM/megatron/training/training.py#L2718)。

因此准确的时间关系是：

```text
一个 optimizer iteration 开始
        |
        v
一次性取出本 iteration 原计划使用的全部 samples/microbatches
        |
        v
汇总 DP×CP plane 上所有 sequence lengths
        |
        v
DefaultDynamicCPScheduler 为整批数据生成完整计划
  - 哪些 samples pack 在一起
  - 每个 scheduled microbatch 划分哪些 CP1/2/4/... groups
  - 每个 rank 应拿到哪些 sample IDs
        |
        v
按计划重路由并 pack 数据，生成新的 scheduled data iterator
        |
        v
进入 pipeline forward_backward_func
        |
        +--> scheduled microbatch 0：读取自己的 local_cp_size/cp_group，完成所有 layers
        +--> scheduled microbatch 1：可以换另一个 local_cp_size/cp_group，完成所有 layers
        +--> ...
        |
        v
optimizer step
```

所以有两个不同粒度：

```text
调度算法：每个 optimizer iteration 调用一次，为本轮全部 scheduled microbatches 制定计划
应用计划：每个 scheduled microbatch forward 前读取一次自己的 local_cp_size/group
Attention layers：不重新调度；同一个 microbatch 的所有 layers 复用同一 runtime group
```

原生 MCore 的具体数据步骤在
[`DefaultDynamicCPScheduler.run()`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule.py#L210)：

```text
1. 从原 data iterator 取样本，并 AllGather 全局 sequence lengths
2. 按长度/packing/workload 生成 rank assignments
3. 用 DP×CP all_to_all_single 把实际 sample tensors 发到目标 ranks
4. 在目标 ranks 上构造 packed microbatches，并写入 local_cp_size
5. 返回一个新的 iterator；scheduled microbatch 数也可能与原来不同
```

这里的 AllToAll 不是“把任意 Python sample 或任意 shape tensor 直接扔给 NCCL”。当前文本
路径能够工作，是因为 scheduler 的输入协议非常受限：`tokens/position_ids/labels/loss_mask`
都是可按 sequence length 展平的一维字段，`original_seq_len/padded_seq_len` 是每样本一个
scalar。MCore 先根据全局 sample IDs 和 lengths 算出每个 source/destination 的精确 split
sizes，然后对每一种 data key 分别执行一次 variable-split `all_to_all_single`：

```text
生成 send/recv sample-ID plan
-> 每个 key 按 sample 顺序 flatten 成一个 contiguous send buffer
-> all_to_all_single(input_split_sizes, output_split_sizes)
-> 接收方根据 sample IDs/lengths 把 flat buffer 拆回各 samples
```

不同 dtype 的字段不能直接共用一个 NCCL tensor，所以源码循环 `data_keys`，不是整个 batch
只调用一次 A2A。接收方也不会从 NCCL 自动得到 shape；shape/offset 来自调度前共同算好的
metadata。

这套硬编码协议不能直接覆盖多模态。例如 `pixel_values` 可能具有不同 H/W，此外还要同步
`image_grid_thw`、image/token placeholder 映射、vision masks/embeddings 等；当前 unpack 逻辑
把除两个 length scalars 之外的字段都假设成“长度等于 text sequence length”，原生
packing 代码还会剔除未列入白名单的自定义字段。因此不能只增加一个 image tensor A2A 就
认为语义完整。

当前 verl Dynamic CP 明确拒绝含 `vision_config` 的模型，检查在
[`transformer_impl.py:121`](../../verl/workers/engine/megatron/transformer_impl.py#L121)。未来若要
支持，至少要选择一种完整的数据协议：

```text
方案 A：先把图像编码为与 placeholder 对齐的 visual tokens/embeddings，再按 packed-token
        协议调度；还要决定 vision encoder 在哪个 ranks 执行及梯度怎样汇总。

方案 B：按 sample ID 单独路由多模态字段；先通信每张图的 shape/offset/grid metadata，
        再按 dtype flatten pixel/embedding buffers 做 variable-split A2A，接收后重建并校验
        image-to-text mapping。

方案 C：像 verl 当前文本 DCP 一样预先把完整多模态 batch 复制到 DP×CP plane，运行时只做
        本地 sample selection；实现简单一些，但显存、CPU/GPU 搬运和图像重复编码代价更高。
```

所以当前 A2A 看起来短，是因为复杂性被“文本 THD schema + 预先计算的 split metadata”约束
住了，并不代表它已经是通用 sample router。

每个 scheduled microbatch 真正要 forward 时，
[`get_batch_on_this_rank_for_sequence_packing()`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule.py#L561)
执行：

```python
local_cp_size = batch["local_cp_size"]
cp_group = get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
get_cp_slice_for_thd(batch, cp_group, ...)
packed_seq_params = PackedSeqParams(
    local_cp_size=local_cp_size,
    cp_group=cp_group,
    ...,
)
```

verl 的时间点相同，也是进入本轮 pipeline forward/backward 之前调度，但数据来源不同：verl
把一份 jagged/NestedTensor batch 复制在整个 DP×CP plane 上，所以不需要原生 MCore 那次样本
reroute；[`TransformerImpl.forward_backward_batch()`](../../verl/workers/engine/megatron/transformer_impl.py#L895)
调用 verl 的薄 adapter，adapter 内部仍实例化 MCore 的 `DefaultDynamicCPScheduler` 来生成
assignments，然后直接从本地复制的 batch 选择对应 sample IDs。

### 8.4 官方所说的 “zero-overhead execution” 到底有没有代码例子

有，但必须先区分两个版本，否则很容易把一个 **Draft 原型**当成当前 MCore 的默认实现。
本节专门核对 NVIDIA/Megatron-LM 官方仓库，不讨论 verl。

#### 8.4.1 截至 2026-08-21，最新 `dev` 有可运行 DCP 示例，但它是同步调度

本次实际拉取并检查的远端版本是：

```text
main: 5b2a621f9644c6359365b488fa2fa76b756539cd
dev:  04fa3cee75a3fbad595d93ac50d49128f0dd816b
```

`dev` 已经有官方的 8 卡示例：

- [Dynamic CP benchmark README](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/examples/dynamic_context_parallel/README.md)；
- [benchmark_dcp.sh](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/examples/dynamic_context_parallel/benchmark_dcp.sh)；
- 对应已合入 `dev` 的 [PR #5123](https://github.com/NVIDIA/Megatron-LM/pull/5123)。

默认就是一个很接近当前问题的 8 GPU 例子：

```text
TP=1, CP=4, PP=1, DP=2
micro_batch_size=1
num_microbatches=8
variable-length THD mock samples: 128..8192 tokens

baseline:
  --sequence-packing-scheduler dp_balanced

DCP:
  --dynamic-context-parallel
  --sequence-packing-scheduler default_dynamic_cp
  --min-dynamic-context-parallel-size 1
```

运行命令是：

```bash
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

不过，这个已合并示例**没有实现后台 async solver**。最新 `dev` 的真实调用顺序仍是：

```text
train_step(iteration N)
  |
  |-- wrap_data_iterator(...)
  |     |
  |     |-- get_batch_and_global_seqlens()
  |     |     `-- 取出本 iteration 全部 samples + gather lengths
  |     |
  |     |-- DefaultDynamicCPScheduler.get_groups_and_subsamples()
  |     |     `-- next_hdp_group_packing_aware() 同步生成 plan
  |     |
  |     |-- reroute_samples_to_dcp_ranks()
  |     |     `-- 对实际 sample tensors 执行 variable-split all_to_all_single
  |     |
  |     `-- build_packed_microbatches()
  |
  `-- forward_backward_func(...)
```

可以直接从下面两个代码位置确认先后关系：

- [`training.py:2893-2933`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/training/training.py#L2893-L2933)：
  `wrap_data_iterator()` 返回后才调用 `forward_backward_func()`；
- [`data_schedule.py:220-408`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule.py#L220-L408)：
  fetch、schedule、sample A2A、packing 全在 `scheduler.run()` 内同步完成。

因此，最新 `dev` 里的这个 benchmark 能说明 **DCP 功能和收益怎样跑**，不能作为博客所说
“solver 与 GPU iteration overlap”的代码例子。

## 9. 从普通 GBS=512 到 Draft 异步 scheduler：先把数据主线走通

这一章从最普通的 MCore 数据并行开始，不预设已经理解 batch、packing 或 CP。前半章分析
最新 `dev` 已合并的流程，后半章再进入 NVIDIA/Megatron-LM Draft PR #2959。

> **Draft 状态说明：第 9.7–9.13 节分析的 async data-sampler scheduler 尚未合入 `main`
> 或 `dev`，不能当成当前 MCore 稳定接口；第 9.14 节再说明当前多模态 payload 缺口。保留
> Draft 是为了学习官方设想的“先调度下一批，再与当前 GPU iteration 重叠”的设计。**

### 9.1 先固定一个贯穿本章的 8 卡例子

为了不让 TP、PP、CP、DP 同时变化，先固定：

```text
GPU 总数 = 8
TP = 1
PP = 1
global_batch_size = 512 条原始样本
micro_batch_size = 1 条原始样本
```

这里的 `512` 指一次 optimizer step 在整个训练任务中共同消费的 **512 条不同原始样本**。
不管后面把它们 pack 成多少个 THD microbatches，也不管一条长序列用了多少个 CP ranks，
原始样本数仍然是 512。

先记住四个名字：

```text
original sample
  dataset 中的一条样本，例如一段 conversation；本例一次 optimizer step 有 512 条。

global batch
  本次 optimizer step 在所有数据副本上合计消费的 original samples；本例为 512。

microbatch
  pipeline schedule 一次 forward/backward 调用处理的数据单位。多次 microbatch 累积梯度后，
  才执行一次 optimizer.step()。

packed microbatch
  把多条可变长 original sequences 在 T 维拼成一条 THD buffer；仍然保留 cu_seqlens，
  Attention 不允许不同原始 sequence 互相看见。
```

最容易混淆的一点是：

```text
pack 不是丢掉样本，也不是把四条 conversation 变成一条训练样本。
pack 只是让四条短序列共享一次更饱满的 GPU microbatch。
loss 仍按各自 token 和 sequence 边界计算。
```

### 9.2 第一层：不开 CP，也不开 packing，普通 MCore 怎样消费 GBS=512

现在配置：

```text
CP = 1
DP = world_size / (TP * PP * CP) = 8
```

8 个 GPU 是 8 个完整的数据副本。一次 optimizer step 的 512 条样本平均分给 8 个 DP ranks：

```text
每个 DP rank 的原始样本数 = 512 / 8 = 64
num_microbatches = 512 / (DP8 * micro_batch_size1) = 64
```

可以把 rank-local 数据想成：

```text
GPU0: 64 条不同样本
GPU1: 64 条不同样本
...
GPU7: 64 条不同样本
合计: 512 条不同样本
```

一次 optimizer iteration 的主线是：

```text
MegatronPretrainingSampler
  -> 每个 DP rank 产生自己的 dataset indices

DataLoader
  -> 每次给本 rank 一个 microbatch；本例 MBS=1

forward_backward_func(num_microbatches=64)
  -> microbatch 0 forward/backward
  -> microbatch 1 forward/backward
  -> ...
  -> microbatch 63 forward/backward

DP gradient synchronization
  -> 8 个模型副本汇总梯度

optimizer.step()
  -> 至此消费完整 GBS=512
```

这条路径没有 Dynamic CP 的 `wrap_data_iterator()`，也没有 sample reroute A2A。若样本长度不同，
最传统的做法是把每个 microbatch pad 到固定 `seq_length`；MBS=1 时虽然没有同一 microbatch
内部的样本互相补齐，模型通常仍按固定 shape 执行，短样本会浪费 padding 计算。

### 9.3 第二层：仍然 CP=1，但开启 sequence packing

现在仍是 8 个 DP ranks，不使用 CP，只增加：

```text
--sequence-packing-scheduler dp_balanced
--max-seqlen-per-dp-cp-rank <一个 packed microbatch 的长度上限>
```

这里必须区分“DataLoader 取样”和“真正 packing”：

```text
Dataset.__getitem__
  -> 原始文本已经完成 tokenize，得到每条 sample 自己的 tensor fields

DataLoader / identity collate
  -> 只返回 List[sample]
  -> 此时 tokens 仍是一条条分开的

wrap_data_iterator()
  -> 收集本 optimizer step 的原始 samples
  -> 调度、重平衡
  -> build_packed_microbatches()
       -> _pack_sequences()
       -> 到这里才真正 torch.cat tokens/labels/loss_mask/position_ids
```

因此 MCore 的这个 pack 既不是 raw text 层面的拼字符串，也不是 Attention 内部临时拼接；它
发生在 dataset 已生成 token tensors 之后、模型 forward 之前。

最新 MCore 中，真正拼接发生在：

- [`data_schedule_utils.py:_pack_sequences()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule_utils.py#L221)；
- [`data_schedule_utils.py:build_packed_microbatches()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule_utils.py#L466)。

`_pack_sequences()` 生成的大致结构是：

```python
packed = {
    "tokens": cat([sample0.tokens, sample1.tokens, ...]),
    "labels": cat([sample0.labels, sample1.labels, ...]),
    "loss_mask": cat([...]),
    "position_ids": cat([...]),
    "cu_seqlens": [0, len0, len0 + len1, ...],
    "cu_seqlens_padded": [...],
    "max_seqlen": max(len0, len1, ...),
}
```

以 GBS=512 为例，`wrap_data_iterator()` 先从每个 DP rank 取出原计划的 64 条，本轮合计仍是
512 条。随后 gather lengths，在全局 512 条里重新决定哪些短序列 pack 在一起、每个 packed
microbatch 归哪个 DP rank。

假设仅为了举例，平均每 4 条短序列能 pack 成一个 THD microbatch：

```text
512 original samples
-> 128 packed microbatches
-> DP8 每 rank 处理 16 个 packed microbatches
-> 这 16 个 microbatches 累计仍包含 64 条 original samples
```

实际 pack 数由真实长度决定，不保证恰好是 4。重要的是 scheduler 返回的有效
`num_microbatches` 可以从原来的 64 变成 16 或其他值，但 optimizer step 仍覆盖 512 条原始
样本。

此时配置中的 `micro_batch_size=1` 可以理解为“每个数据副本一次 forward 一个 **packed
container**”；这个 container 内部可以通过 `cu_seqlens` 携带多条 original sequences。因此
它与“一次 forward 只训练一条 conversation”已经不是同一个含义。

#### 9.3.1 为什么 CP=1 的 packing 路径也可能出现 sample A2A

因为流程是“先按普通 DP 读取，再看全局长度重新分配”：

```text
1. GPU0 原本读到 sample 17
2. scheduler 发现 sample 17 与 GPU5 的几条短样本 pack 在一起最平衡
3. 最终 packed microbatch 被分配给 GPU5
4. sample 17 的 tokens/labels/... 必须从 GPU0 搬到 GPU5
```

因此当前已合并 MCore 会对实际 sample fields 调用 variable-split
`torch.distributed.all_to_all_single()`。代码在
[`reroute_samples_to_dcp_ranks()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule_utils.py#L347)。

所以：

```text
普通 DP、无 packing：没有 sample reroute A2A
DP-balanced packing：可能需要 sample reroute A2A，即使 CP=1
```

#### 9.3.2 为什么先 AllGather lengths，后面还要 AllToAll payload

这里确实有两阶段 collective，但不是把同一批数据通信两次：

```text
阶段 1：AllGather metadata，目的是“制定共同计划”
  每个 rank 发送：本地有几条 sample + 每条 sample 的 padded_seq_len
  每个 rank 得到：全局 sample_id -> length 表

阶段 2：AllToAll payload，目的是“执行共同计划”
  scheduler 已决定每条 sample 最终属于哪个 packed microbatch/target rank
  各 source rank 再发送：tokens、labels、loss_mask、position_ids 等实际 tensors
```

以本节 `CP1/DP8/GBS512` 为例：

```text
每 rank 原先有 64 条

第一次 AllGather：
  gather 8 个 local sample counts（通常每个都是 64）

第二次 AllGather：
  gather 8 * 64 = 512 个 int32 lengths
  纯长度数据约 512 * 4 bytes = 2 KiB，不含 tokens

本地 Python scheduler：
  8 个 ranks 根据相同的 512 个 (sample_id, length) 独立运行确定性算法
  -> 都得到相同 assignment plan，所以不必再广播完整 plan

随后 payload AllToAll：
  按 plan 对 tokens、labels、loss_mask、position_ids 等字段分别调用
  variable-split all_to_all_single
```

长度 AllGather 的源码是
[`_get_global_seqlens_and_ids()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule_utils.py#L165-L218)：

```python
torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)
torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)
```

payload A2A 则在
[`reroute_samples_to_dcp_ranks()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule_utils.py#L347-L463)：

```python
for key in data_keys:
    torch.distributed.all_to_all_single(
        output=recv_tensor,
        input=send_tensor,
        output_split_sizes=recv_lens_split,
        input_split_sizes=send_lens_split,
        group=dp_cp_group,
    )
```

因此它不是“一次 AllGather 就自动完成均衡”。AllGather 只让所有 ranks **知道**该怎样均衡；
真正改变 sample 所在 rank，仍必须执行 A2A。即使某轮 plan 恰好没有跨 rank 迁移，当前函数
仍会进入这些 collective，只是相应 split 可能为 0 或只发给自己。

还要注意 group 范围：

```text
CP1/DP8：length AllGather 的 dp_group 就是 8 ranks；payload A2A 的 dp_cp_group 也是 8 ranks。

静态 CP4/DP2：每条 CP lane 在自己的 DP2 group 内 gather unique sample lengths；
               payload reroute 使用 DPxCP 的 8-rank group。
```

最后再把“默认”说严谨：

```text
config.sequence_packing_scheduler is None
  -> 不调用 wrap_data_iterator()
  -> 没有上述 length AllGather 和 sample payload A2A

dp_balanced / default_dynamic_cp
  -> 调用 wrap_data_iterator()
  -> 当前已合并同步路径包含 length AllGather + payload A2A

Draft async sampler
  -> 先根据 indices/lengths 规划，再让目标 rank 直接读取
  -> Draft 原型避免 payload A2A；博客设想的 distributed probing 仍可能 gather 轻量 metadata
```

#### 9.3.3 这是为了预训练吗？已知长度的 SFT 能否直接在 sampler 层均衡

先说结论：这套“先读取，再 AllGather length，再 A2A 搬 sample”的同步实现，**不是预训练在算法上必须这样做**。
它更像是当前 MCore 为了兼容通用 `data_iterator` 所采用的实现折中。传统 MCore 数据入口虽然叫
`build_pretraining_data_loader()`，但 sequence-packing scheduler 面向的是一般的变长训练数据；固定长度
GPT 预训练样本反而通常不需要按长度重新 packing。

当前 scheduler 接收到的是一个已经按 DP rank 切分好的 iterator，而不是“可随机访问的全局 dataset +
全局 length table”：

```text
普通 sampler 已经先决定：sample 17 由 rank 0 读取
                                   ↓
data_iterator 真正产出 sample 17 后，scheduler 才看到 padded_seq_len
                                   ↓
scheduler 又发现：sample 17 放到 rank 5 的 pack 更平衡
                                   ↓
此时 sample 已经在 rank 0，只能通过 payload A2A 搬到 rank 5
```

这个接口顺序的优点是能兼容更多数据源：外部 iterator、streaming/on-the-fly 数据、动态 tokenization、
动态截断，以及只有取出 sample 后才能确定最终长度的数据。代价就是，对于本来已经知道全部长度的数据，
会发生本可避免的 metadata collective 和 payload reroute。

文本 SFT 如果已经离线保存了**模型实际使用的长度**，完全可以改成 sampler-first：

```text
离线 metadata：sample_id -> post-tokenization padded_seq_len
                             │
                             v
同一 seed/epoch 生成全局 shuffled sample ids
                             │
                             v
从中取出本轮 GBS=512 的 ids，在 CPU 上先做全局 packing/bin-packing
                             │
                             v
把 packed microbatches 分配给 DP ranks
                             │
            ┌────────────────┼────────────────┐
            v                v                v
       rank 0 sampler   rank 1 sampler   ... rank 7 sampler
       只 yield 属于    只 yield 属于        只 yield 属于
       rank 0 的 ids    rank 1 的 ids        rank 7 的 ids
            │                │                │
            v                v                v
       各 rank 直接从 dataset/storage 读取最终属于自己的 samples
                             │
                             v
                 本地 collate/pack，然后进入模型
```

这样不需要把 tokens/labels 从一个 GPU rank A2A 到另一个 GPU rank。调度计划有两种生成方式：

1. 所有 ranks 都持有同一份 length table、shuffle seed 和 epoch，在 CPU 上确定性地计算同一份全局计划，
   然后只保留分给自己的 ids；运行时甚至不需要分布式通信。
2. 一个 coordinator 计算计划，再 scatter/broadcast 少量 sample ids；仍有 metadata 通信，但没有大 payload A2A。

不过不能直接使用“每个 rank 先拿自己的 64 条，再在本地按长度排一下”。那只能改善每个 rank 内部的 packing，
无法把 rank 0 的一组长样本与 rank 5 的一组短样本进行全局均衡。正确顺序必须是：

```text
global shuffle -> 取全局 512 条 -> global pack/assign -> 最后才按 rank 分发 indices
```

SFT 还要确认 length table 记录的不是原始文本字符数，而是经过 chat template、tokenization、EOS、truncation、
多模态 token expansion 以及 scheduler 对齐之后的 `padded_seq_len`。只要这些步骤会随机变化或依赖运行时处理，
离线长度就可能失效。纯文本、预 tokenized 的 SFT 最适合 sampler-first；在线 RL、streaming 数据和目前尚未完整
接入 length-probe 契约的多模态数据更难直接采用。

因此 Draft async sampler 所要解决的核心问题，正是把调度时点从“样本已经读到错误 rank 以后”前移到
“DataLoader 读取以前”。它不是让均衡消失，而是让均衡只改变 **index 的去向**，避免再移动样本 payload。

#### 9.3.4 MCore 现在是否已经提供这种 length-aware sampler

以本文核对的 MCore `dev` commit `04fa3cee` 为准，**已合并代码还没有提供生产可用的 sampler-first、
length-aware 全局 packing sampler**。标准 `build_pretraining_data_loader()` 内置的是：

```text
dataloader_type=single
  -> MegatronPretrainingSampler             # 顺序取样并按 DP rank 切片

dataloader_type=cyclic
  -> MegatronPretrainingRandomSampler       # 按 epoch shuffle/shard，再按 DP rank 取样

full validation
  -> MegatronFullValidationSampler

dataloader_type=external
  -> 用户自己提供 DataLoader / sampler
```

源码在
[`megatron/training/datasets/data_samplers.py`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/training/datasets/data_samplers.py#L18-L115)。
`DpBalancedScheduler` 和 `DefaultDynamicCPScheduler` 的名字里虽然有 scheduler，但它们不是 PyTorch
`Sampler`：它们接收已经产出 samples 的 `data_iterator`，所以仍走本章介绍的 length AllGather 和 payload
A2A 路径。

Draft PR `#2959` 中则出现过两个明确面向 SFT 的原型：

```text
MegatronSFTSampler
MegatronSFTPrefetchDPBalancedSampler
```

其中 `MegatronSFTPrefetchDPBalancedSampler` 的方向正是 sampler-first：后台进程先查看一个 global batch 的
indices/lengths，计算 packing 与 rank assignment，然后每个 DP rank 只 yield 分给自己的 indices。这样目标
rank 的 DataLoader 会直接读取最终 samples，不需要事后 payload A2A。

不过这个 Draft 原型还不能当成现成稳定功能：

```python
def get_numel(self, idx):
    data = self.dataset[idx]          # 为探测长度实际读取 sample
    return [idx, data["tokens"].numel()]

# TODO: use distributed `get_numel` to reduce io pressure.
```

也就是说，它还没有标准化的 `sample_id -> length` metadata 接口；当前原型中每个 rank 的后台进程会探测
整个 global batch，可能造成重复 I/O。它位于 Draft 分支的
`megatron/legacy/data/data_samplers.py`，尚未进入上述最新 `dev` 数据管线。

所以当前实际选择是：

```text
想直接使用已合并 MCore 功能
  -> 使用同步 sequence-packing wrapper，接受 metadata AllGather + payload A2A

纯文本 SFT 已有离线 lengths，想消除 payload A2A
  -> 通过 dataloader_type=external 提供自定义 length-aware sampler/DataLoader，
     或预先生成 packed dataset/epoch packing plan；这部分目前需要用户侧实现

想研究 NVIDIA 正在探索的方案
  -> 参考 Draft #2959，但不要把它视为已发布、已稳定的 MCore API
```

### 9.4 第三层：静态 CP4 + packing

8 卡开启 `CP=4` 后：

```text
DP = 8 / CP4 = 2
```

一个 step 仍然只有 512 条不同的原始样本，因此两个 DP 副本各负责 256 条：

```text
GPU0..3：共同处理第 1 组 256 条
GPU4..7：共同处理第 2 组 256 条
```

当前实现中，同一 CP4 group 的四张卡会先读到相同的 256 条样本，但它们不是四份不同训练样本，
所以 GBS 仍是 512。

接下来的流程就是：

```text
每个 DP 副本原本有 256 条
  -> 在 dp_group（size=2）内 AllGather 全局 512 条的 lengths
  -> 根据 lengths 计算均衡后的 packing plan
  -> 在 dp_cp_group（DP×CP，size=8）内用 A2A
     把实际 samples 重新分发到目标 ranks
  -> 把多条短样本 pack 成 n 个 packed THD microbatches
  -> forward 前对每个完整 packed microbatch 做 CP4 切分
  -> 四张卡分别计算约 1/4 token rows
```

因此你可以先简单理解为：

> **先在 DP2 group 内收集 lengths，再在 DP×CP 的 8-rank group 内重新分发 samples，
> 然后 pack，最后应用 CP4 切分。**

CP 切分发生在
[`get_batch_on_this_rank_for_sequence_packing()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule.py#L568)
里的 `get_cp_slice_for_thd()`。

### 9.5 第四层：当前已合并的同步 Dynamic CP

仍用初始化配置 `CP=4, DP=2`，但 Dynamic CP 会预创建 DP×CP plane 上的 CP1/CP2/CP4/CP8
子组。权重没有重新切分；runtime 只是为每个 scheduled microbatch 选择不同协作组。

一次 wave 可以概念性地长这样：

```text
GPU0..3: 一个很长的 packed microbatch，runtime CP4
GPU4..5: 一个中等 packed microbatch，runtime CP2
GPU6:    一个短 packed microbatch，runtime CP1
GPU7:    另一个短 packed microbatch，runtime CP1

8 张卡在同一时段被 4+2+1+1 完整覆盖。
```

对 GBS=512，当前同步实现的完整顺序是：

```text
train_step(iteration N)
  |
  |-- 从原 DataLoader 取出本轮 512 条 original samples
  |
  |-- get_batch_and_global_seqlens()
  |     `-- 汇总 sample IDs 和 lengths
  |
  |-- DefaultDynamicCPScheduler.get_groups_and_subsamples()
  |     `-- 决定怎样 pack、目标 ranks、每组 CP1/2/4/8
  |
  |-- reroute_samples_to_dcp_ranks()
  |     `-- actual tokens/labels/... 做 sample A2A
  |
  |-- build_packed_microbatches()
  |     `-- 真正 cat，并写 local_cp_size
  |
  `-- forward_backward_func(scheduled_num_microbatches)
        `-- 每个 microbatch 根据 local_cp_size 选择预创建 group
```

注意先后关系：**512 条实际数据已经按普通 DataLoader 规则读到各 rank，scheduler 才开始
工作**。因此 scheduler 改变归属后，需要 sample payload A2A。

在最新 `dev`，这整段都在本 iteration 的 forward/backward 之前同步发生：

- [`training.py:2893-2933`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/training/training.py#L2893-L2933)；
- [`data_schedule.py:DpBalancedScheduler.run()`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule.py#L220-L408)。

### 9.6 到这里先对照：pack 在哪、A2A 为什么出现

| 模式 | DP/CP（本例） | 512 条样本何时分配 | pack 在哪里 | sample payload A2A |
|---|---:|---|---|---|
| 普通 DP | DP8/CP1 | DataLoader 直接按 DP rank 分 | 不 pack | 不需要 |
| DP packing | DP8/CP1 | 先普通读取，再按长度重排 | `build_packed_microbatches()` | 需要 |
| 静态 CP4 packing | DP2/CP4 | 先读到两个 DP replicas，再按长度重排 | 同上；forward 前再切 CP4 | 需要 |
| 当前同步 Dynamic CP | 初始化 DP2/CP4，runtime CP1/2/4/8 | 先读取，再同步求 plan | 同上并写 `local_cp_size` | 需要 |
| Draft async sampler | runtime Dynamic CP | **先求 plan，再按最终 indices 读取** | 后台不 pack；主进程只做 packing | 不需要这次 reroute A2A |

这张表里最后一行才是下面要学习的 Draft。它不是把 A2A 异步化，而是把 planning 提前到
真实样本读取之前，从因果关系上消除“读错 rank 后再搬一次”的需要。

### 9.7 Draft 代码在哪里

NVIDIA 博客的 `scheduler` 链接实际指向
[NVIDIA/Megatron-LM PR #2959: Hybrid cp example](https://github.com/NVIDIA/Megatron-LM/pull/2959)。
该 PR 的 head 是 `3b309c627e902913aafbb6f459f528a4ea06736c`，包含这些很明确的提交：

```text
0d2c232  add hotswitch solver
37678db  add gpu_timer and pipeline simulator
f1fec85  add profile memory; add multiprocess solver
266c548  add flag for async scheduler and new scheduler
3b309c6  limit thread number in background process
```

但是它截至本次检查仍是 Draft，目标是 `dev`，并没有被 `main` 或 `dev` 包含。因此下面解释
的是**官方仓库里可以阅读的研究原型**，不是当前稳定 API。

### 9.8 Draft 开关怎样把调度从训练主进程搬到 data sampler

先继续使用前面的假设：8 张 GPU，一个 global batch 固定包含 512 条原始样本。现在连续训练两个
global batches：

```text
global batch 0：sample 0    ... sample 511
global batch 1：sample 512  ... sample 1023
```

Draft 最通俗的流程是：

```text
训练开始前：
  后台先查看 batch 0 这 512 条的 lengths
  -> 决定怎样 pack
  -> 决定每个 packed microbatch 交给哪些 GPU、使用 CP1/2/4/8
  -> 得到 batch 0 的调度计划

训练 batch 0：
  各 GPU 的 DataLoader 按计划直接读取自己最终需要的 samples
  -> 主进程只负责 pack，不再重新求解，也不再用 A2A 搬 samples
  -> GPU 对 batch 0 执行 forward/backward

与此同时：
  后台查看 batch 1 的另外 512 条 lengths
  -> 提前算出 batch 1 的 packing 和 GPU/CP 分配计划

batch 0 结束后：
  主进程直接取出已经算好的 batch 1 计划
  -> 各 GPU 按计划读取 batch 1
  -> pack
  -> forward/backward batch 1
```

把 CPU 和 GPU 放在同一条时间线上就是：

```text
时间 ----------------------------------------------------------------->

主进程/GPU： [等待 batch 0 计划] [读取、pack、训练 batch 0] [读取、pack、训练 batch 1]
后台 CPU：    [规划 batch 0]       [规划 batch 1----------] [规划 batch 2----------]
```

第一批仍可能等待，因为 batch 0 的计划还没有提前准备。进入稳定训练后，只要后台规划 batch 1 的时间
不超过 GPU 训练 batch 0 的时间，下一轮取计划时就不需要额外等待。这就是 Draft 所说的调度与训练
overlap。

它与当前已合并同步方案的核心差别只有一句话：

```text
当前同步方案：先把 512 条读到各 rank，再规划，所以需要 sample A2A。
Draft 方案：  先规划这 512 条属于谁，再让目标 rank 读取，所以不需要 sample A2A。
```

这个 Draft 原型有一个直接缺点：**每个 rank 的后台进程都会先探测完整 512 条**。它对每个 index
调用一次 `dataset[idx]`，取出 `tokens.numel()`；规划完成后，DataLoader 又会正式读取分给本 rank 的
samples。因此可能出现重复 I/O、tokenization，8 ranks 还会重复做相同的 512 条长度探测。

但这不是全局规划必须付出的代价。也可以把 512 条探测任务均匀分给 8 ranks，每个 rank 只探测 64 条，
然后在 8-rank scheduling group 内 AllGather 这 512 个 lengths；大家拿到完整长度表后再计算相同计划。这样仍然只通信轻量
metadata，不需要搬 tokens/labels 的 payload A2A。对于长度已知的文本 SFT，更好的办法是所有 ranks
直接读取共享的离线 `sample_id -> length` 表，连样本探测都不需要。

真正不可以的是：每个 rank 只知道自己的 64 条 lengths，彼此又完全不交换 metadata。这样只能做
本地 packing，无法根据全局 512 条长度进行跨 rank 均衡。

对应到代码，Draft 增加：

```bash
--async-hybrid-context-parallel-scheduler
--hybrid-context-parallel-scheduler only_packing_no_scheduling
```

两者必须一起用。创建 DataLoader 时，代码不再选择普通的 `MegatronSFTSampler`，而是：

```python
if args.async_hybrid_context_parallel_scheduler:
    assert args.hybrid_context_parallel_scheduler == "only_packing_no_scheduling"
    batch_sampler = MegatronSFTPrefetchDPBalancedSampler(...)
```

关键设计是把职责拆成两半：

```text
MegatronSFTPrefetchDPBalancedSampler
  -> 提前看下一 global batch 的 lengths
  -> 运行 cost model / solver / pipeline simulator
  -> 直接决定每个 DP rank 后续应该读取哪些 dataset indices

OnlyPackingNoSchedulingScheduler
  -> 不再求解
  -> 只消费 sampler 已写好的 sample IDs、local_cp_size
  -> pack 成 MCore forward 所需 microbatches
```

这也是为什么它不能同时再用 `balanced_with_pp`：否则 sampler 和
`wrap_dataloader()` 会重复调度一次。

### 9.9 后台进程、两条 Queue 和一拍预取

`MegatronSFTPrefetchDPBalancedSampler.__init__()` 创建的不是 Python thread，而是一个
`torch.multiprocessing` 子进程：

```python
ctx = mp.get_context("fork")
self._queue1 = ctx.Queue()  # 主进程 -> 后台进程：待规划的 sample indices
self._queue2 = ctx.Queue()  # 后台进程 -> 主进程：完成的 schedule plan

self._prefetch_process = ctx.Process(
    target=self.prefetch_batch,
    args=(self._queue1, self._queue2),
    name="prefetch_batch",
    daemon=False,
)
self._prefetch_process.start()
```

后台进程只保留一个 CPU thread，避免 solver 与 DataLoader/训练主进程争抢整个 CPU：

```python
torch.multiprocessing._set_thread_name("pt_prefetch_batch")
torch.set_num_threads(1)

while True:
    full_batch = queue1.get()
    batch_data = self.prepare_batch(full_batch)
    queue2.put(batch_data)
```

其中 `prepare_batch()` 的核心是：

```python
batch_numel = [self.get_numel(idx) for idx in batch]

groups, sample_id_groups, cp_sizes = (
    self.data_scheduler.get_groups_and_subsamples(
        batch_numel,
        self.config,
        return_cp_sizes=True,
    )
)
```

主进程 sampler 的逻辑则可以压缩为：

```python
# 启动时先提交 batch 0
queue1.put(global_batch_0_indices)

while training:
    # 取当前 batch 的计划；第一次一定可能等待
    groups, sample_id_groups, cp_sizes = queue2.get()

    # 在开始 yield 当前 batch 前，立刻提交 batch N+1
    queue1.put(next_global_batch_indices)

    # DataLoader 按已经决定好的 rank-local indices 真正加载 batch N
    for microbatch_idx in range(len(sample_id_groups)):
        yield [
            (sample_id, num_microbatches_left, local_cp_size),
            ...,
        ]
```

完整源码集中在 Draft head 的
[`data_samplers.py:195-321`](https://github.com/NVIDIA/Megatron-LM/blob/3b309c627e902913aafbb6f459f528a4ea06736c/megatron/legacy/data/data_samplers.py#L195-L321)。

### 9.10 所谓 overlap，时间线上到底重叠了什么

```text
启动阶段
  main:  queue1.put(plan request for batch 0)
  child: probe lengths(batch 0) -> solve(batch 0)
  main:  queue2.get()                          # 冷启动，可能等待

iteration N
  main/GPU: load assigned samples N -> pack N -> forward/backward N -> optimizer N
  child CPU: probe lengths N+1 -> cost model -> solver -> pipeline simulator N+1
             `---------------- 与上面整段训练计算重叠 ----------------'

iteration N+1 边界
  main: queue2.get(plan N+1)
        |- 后台已经完成：几乎立即返回
        `- 后台尚未完成：仍然在这里等待，不是数学意义上的绝对 0 开销
```

所以 “zero-overhead” 更准确的工程含义是：在 steady state 且满足

```text
T_probe(N+1) + T_solver(N+1) <= T_train(N)
```

时，CPU 调度延迟不再增加 iteration 的关键路径。如果下一批特别难求解、storage/CPU 竞争严重，
或者 GPU iteration 很短，`queue2.get()` 仍会暴露 stall。

### 9.11 这个 Draft 为什么不再需要 sample-tensor reroute A2A

这点很容易误解。不是把 sample A2A 也异步到了后台进程，而是**改变了数据读取顺序**：

```text
最新 dev 的同步路径：
  每个 DP rank 先读自己原来的 samples
  -> 全局 plan 决定目标 ranks
  -> actual tokens/labels 用 NCCL all_to_all_single 重路由

Draft async sampler 路径：
  后台先根据 dataset indices/lengths 得到全局 plan
  -> sampler 直接向每个 DP rank yield 它最终应该处理的 indices
  -> DataLoader 从源数据直接读取正确 samples
  -> 后面只 packing，不再执行 sample-tensor A2A
```

放回本章的 512 条样本例子：

```text
后台收到：global batch N 的 512 个 indices
           [N*512, N*512+1, ..., (N+1)*512-1]

后台只 probe lengths 并求解，例如得到：
  sample 17 -> GPU4..5 组成的 runtime CP2 group

随后 sampler 直接让 GPU4 和 GPU5 的 DataLoader 读取 index 17。
sample 17 不会先作为实际 tokens 落到 GPU0，再从 GPU0 A2A 搬到 GPU4/GPU5。
```

如果一个 sample 分配给 CP2/CP4，多个 CP peers 都必须获得它。Draft 最直接的实现可能让这些
peers 分别从 dataset 读取同一个 index，因此“省掉网络 sample A2A”不等于数据完全没有代价；
它可能换来重复 storage I/O、tokenize，尤其在多模态下还可能重复 image/video decode。

因此它隐藏的是 CPU probing/solver，并通过“先规划、后读取”消除了那次 actual sample A2A；
它不是让 NCCL A2A 与 forward 神奇地重叠。

sampler yield 的三元组最终由 dataset 解码：

```python
(idx, num_microbatches_left, cp_size)

ret["num_micro_batches_left"] = num_microbatches_left
ret["local_cp_size"] = cp_size
```

随后 `OnlyPackingNoSchedulingScheduler` 从第一个 sample 的
`num_micro_batches_left + 1` 恢复本 global batch 有多少 scheduled microbatches，并把每个
microbatch 的 `local_cp_size` 写进 packed metadata。forward 再据此选择预创建的 CP group。

### 9.12 哪些开销从来没有被这套机制隐藏

```text
没有消失 1：每层 Attention 的 boundary P2P
没有消失 2：HCA/CSA compressed K/KV AllGather
没有消失 3：Indexer 所需通信和计算
没有消失 4：真正加载/预处理当前 samples 的成本
没有消失 5：如果后台 solver 没赶上，iteration 边界的 queue2.get() 等待
```

预创建 CP1/2/4/... process groups 只是消除了 runtime 创建 NCCL communicator 的成本，也
没有消除这些 group 上之后真实发生的通信。

### 9.13 为什么不能直接把 Draft 当成生产代码

源码中还能看到几个非常明确的原型痕迹：

- `get_numel(idx)` 实际调用 `dataset[idx]` 后才读 `tokens.numel()`，可能完整读取/分词一遍；旁边
  仍有 `TODO: use distributed get_numel to reduce io pressure`，尚不是博客描述的完善 distributed
  lightweight probing；
- 每个训练 rank 都创建自己的后台进程并求解，而不是只有一个中心 scheduler；
- tuple index 到 `num_microbatches_left/local_cp_size` 的接线在该版本主要加在 mock SFT dataset；
- sampler 的迭代推进、后台进程退出和通用 dataset schema 仍有明显未收尾之处；
- PR 页面仍标记 Draft、未完成 review/checklist，且该 head 不是最新 `dev` 的祖先。

所以当前最可靠的结论是：

```text
最新 dev：有官方可运行 8 卡 DCP benchmark；scheduler 在 train_step 内同步执行。

Draft #2959：有博客所说 async data-sampler solver 的清晰代码原型；
             用一拍预取把 N+1 的 probing/solver 与 N 的 GPU 训练重叠，
             并通过先计划后读取避免 sample-tensor reroute A2A；尚未合并。

“zero overhead”：是理想 steady-state 下新增 scheduler latency 被覆盖，
                 不是整个 Dynamic CP 没有通信，也不是任何 batch 都保证 0 stall。
```

### 9.14 博客使用了 VLM/video 场景，但公开代码没有给出多模态 payload 流程

这个疑问是成立的。NVIDIA 博客先用两类场景说明 sequence-length 长尾：

```text
VLM training：不同 image/video samples 展开后的 token 数差异很大
video DiT：高分辨率、长视频可能产生数万 latent tokens
```

博客中的 Nsight 图也确实来自 VLM training；但文章最后公开的定量表格其实是 Llama-13B
在 GitHub/CommonCrawl 文本数据集上的结果，并不是端到端 VLM benchmark。文章后面的 MCore
接线也只讨论了：

```text
sequence lengths
sample IDs
THD tokens
cu_seqlens / max_seqlen
local cp_size / cp_group
```

它没有定义下面这些原始多模态字段怎样被 probe、重路由、pack 和重建：

```text
pixel_values / pixel_values_videos
image_grid_thw / video_grid_thw
每个 sample 有几张图或几段视频
media 与 text placeholder 的对应关系
timestamps、resolution、frame count 等 metadata
vision encoder / VAE 在哪个 ranks 执行及梯度怎样处理
```

核心原因是必须区分两个层次：

```text
调度控制面（planner）
  输入只要 sample_id、有效 token length、估算 FLOPs/显存
  输出是 grouping、target ranks、local_cp_size
  -> 原理上可以与 text/image/video 模态无关

数据面（payload movement）
  必须知道每个字段是 token-aligned、sample-aligned 还是 media-aligned
  还必须知道字段应该 slice、broadcast、copy、重新读取还是 variable-shape A2A
  -> 当前公开 MCore 实现主要是 text-only THD schema
```

最新 `dev` 的代码非常直接地证明了这个限制。`DpBalancedScheduler` 只声明六种输入字段：

```python
return [
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "original_seq_len",
    "padded_seq_len",
]
```

随后 `scheduler.run()` 根据 PP stage 建立 `keys_to_keep`，并删除不在集合内的其他字段。源码
注释甚至明确写着：custom dataset metadata 会被静默丢弃，需要使用者自己扩展。代码位置：

- [`data_schedule.py:154-163`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule.py#L154-L163)；
- [`data_schedule.py:320-335`](https://github.com/NVIDIA/Megatron-LM/blob/04fa3cee75a3fbad595d93ac50d49128f0dd816b/megatron/core/datasets/data_schedule.py#L320-L335)。

Draft #2959 也没有补上这部分。它的异步 probing 只是：

```python
data = self.dataset[idx]
numel = data["tokens"].numel()
```

即 scheduler 只看最终 token length；它没有处理 images、videos 或 media mapping。该 Draft
能够解释“怎样异步求解下一批”，不能解释“怎样端到端训练多模态 batch”。

这一缺口在 NVIDIA/Megatron-LM 官方仓库中已经有单独的开放 issue：
[Support multimodal payloads in Dynamic-CP sequence packing scheduler #5683](https://github.com/NVIDIA/Megatron-LM/issues/5683)。
issue 列出的缺失字段正包括 `pixel_values`、`pixel_values_videos`、`image_grid_thw` 和
`video_grid_thw`，并指出它们不能按 text token length 使用当前 unpack/reroute/pack 逻辑。

#### 9.14.1 博客里的多模态 benchmark 可能怎样工作

公开资料没有给出唯一答案。下面只能作为根据现有架构作出的合理推断，不能写成已公开实现：

```text
可能路径 A：scheduler 之前已经完成视觉编码
  image/video -> vision encoder 或 VAE -> visual/latent token sequence
  DCP 只看到 token-aligned THD/latent tensors
  -> 对 scheduler 而言确实与文本相似

可能路径 B：先根据 dataset index 中的轻量 metadata 调度
  metadata 提前给出 text tokens + visual/latent tokens 的预计长度
  sampler 先决定目标 ranks
  目标 ranks 再直接从 storage 读取自己的完整 image/video sample
  -> 与 Draft 的“先计划、后读取”最接近，也不需要路由 raw pixels

可能路径 C：依赖未公开的上层训练/data pipeline
  MCore 只接收已经整理好的 token/latent payload
  NeMo、内部 VLM/DiT pipeline 负责原始媒体读取、编码和 mapping
```

无论是哪条路径，公开博客展示的都是**多模态 workload 使用 DCP 后的负载平衡收益**，不是
公开 MCore 已经提供 raw multimodal payload router 的证据。

#### 9.14.2 如果公开 MCore 真正支持多模态，还需要补什么

比较干净的接口应该把 planner 和 payload policy 分离：

```text
1. planner
   sample metadata -> sample grouping + target ranks + local_cp_size

2. payload policies
   token-aligned: tokens/labels/loss_mask/position_ids，按 THD rows 切分
   sample-aligned: sample IDs、media counts，跟随 sample 路由
   media-aligned: pixel/latent tensors，按每张图/视频的 offsets 与 shapes 路由
   replicated: 某些 grid/config metadata，在目标 CP group 内 broadcast/copy

3. reconstruction
   根据 image_grid_thw/video_grid_thw、offsets 和 placeholder mapping
   在目标 rank 重建完整 multimodal sample

4. model placement
   明确 vision encoder/VAE 是调度前执行、目标 rank 执行，还是单独并行执行；
   同时定义激活和梯度怎样进入后续 Dynamic CP group
```

所以目前不能仅凭博客中的 VLM 图，就认为最新 MCore Dynamic CP 已经端到端支持多模态。更
准确的说法是：**DCP 的规划算法天然可以使用多模态样本展开后的 token cost；公开的数据
重路由和 packing 实现仍是 text-only，端到端多模态 payload 支持尚未完成。**

#### 9.14.3 后续持续跟踪的 issue

固定跟踪：

- [NVIDIA/Megatron-LM issue #5683: Support multimodal payloads in Dynamic-CP sequence packing scheduler](https://github.com/NVIDIA/Megatron-LM/issues/5683)

最后核对日期：`2026-08-21`。当时 GitHub 页面显示 `Open`，类型为 enhancement，已有
assignee，处于等待维护者后续处理的状态。后续关注时不能只看 issue 是否关闭，建议逐项检查：

```text
[ ] 是否出现关联 PR 或 dev commit
[ ] scheduler planner 与 payload movement 是否拆成可扩展接口
[ ] 是否支持 pixel_values / pixel_values_videos
[ ] 是否支持 image_grid_thw / video_grid_thw 和其他 media metadata
[ ] 是否保留 text placeholder 与 image/video payload 的一一映射
[ ] variable-shape media tensors 是重新读取、A2A、broadcast，还是先编码成 latent tokens
[ ] vision encoder / VAE 的执行位置与梯度路径是否有明确设计
[ ] 同步 DefaultDynamicCPScheduler 是否有 multimodal 单元测试
[ ] Draft async sampler 或新的异步实现是否也覆盖 multimodal probing/prefetch
[ ] 是否出现端到端 VLM/video Dynamic CP 示例与性能结果
```

只有当 payload schema、数据搬运、模型接线和端到端测试同时出现，才能认为“多模态 Dynamic
CP 正在支持”已经从规划阶段进入了可用实现阶段。

### 9.15 verl 已经会平衡 batch，Dynamic CP 到底还在哪些场景有收益

#### 9.15.1 先给结论

这个疑问是合理的。verl 静态训练路径本来就做了两层平衡：

1. controller 用 [`_balance_batch()`](../../verl/trainer/ppo/ray_trainer.py#L1157) 把样本尽量均匀地分给 DP ranks；
2. 每个 rank 再用 [`rearrange_micro_batches()`](../../verl/utils/seqlen_balancing.py#L348) 把样本尽量装成负载相近的 microbatches。

所以 Dynamic CP 不是“静态 CP 一定很差”的补丁。假如静态 packing 已经满足下面三点：

```text
每个 pack 都装满真实 token
+ 各 pack 的实际运行时间接近
+ 固定 CP 的通信占比很低
```

那么 DCP 很可能没有明显收益，甚至会因为调度和动态组处理略慢。它主要解决的是静态
packing **仍然消除不了的剩余不均衡和固定 CP 通信**。

下面统一使用这个例子：

```text
8 张 GPU
每卡 token 上限 C = 32K
静态配置 CP4 × DP2

因此，一个 CP4 pack 最多放 4 × 32K = 128K token；
一轮可以同时运行两个这样的 pack。
```

#### 9.15.2 对照组：这种情况基本没有收益

```text
pack A: 128K 有效 token，实际计算耗时 100 ms
pack B: 128K 有效 token，实际计算耗时 102 ms

静态 CP4:
GPU 0~3 -> A，每卡约 32K
GPU 4~7 -> B，每卡约 32K
```

这里容量、计算和通信都已经比较理想。DCP 即使重新组合 CP1/2/4，也不能凭空减少模型所需
计算。**这正是“DCP 只是锦上添花”的典型场景。**

#### 9.15.3 场景一：为了偶发超长样本，被迫把固定 CP 开得很大

假设绝大多数 step 只有 16K~32K 序列，但偶尔出现一条 200K 序列：

```text
固定 CP:
  为了让 200K 能放下，只能长期使用 CP8
  即使当前 step 全是 32K，仍然让 8 卡共同处理一个 CP8 pack

Dynamic CP:
  普通 step -> 8 个 CP1，或若干 CP2，并行处理短序列
  遇到 200K -> 临时使用 CP8
```

这里的主要收益不是“多装了 token”，而是短序列 step 不再长期支付 CP8 的 P2P/AllGather
通信。RL 每个 step 的输出长度变化很大时，这个场景比离线 SFT 更常见。

如果离线 SFT 已知所有长度，并且可以把长、短样本分别组 epoch 或分别选择静态配置，那么
DCP 的优势会缩小。

#### 9.15.4 场景二：一条超长序列形成关键路径

假设本轮有一条 128K 长序列和许多短序列。静态 `CP4 × DP2` 最多只能给这条长序列 4 卡；
另一个 CP4 group 可能早已处理完短序列并等待：

```text
时间轴（示意）

静态 CP4:
GPU 0~3  [----------- 128K 长序列 -----------]
GPU 4~7  [-- 短序列 --][       等待          ]

DCP 的一种可能排法:
GPU 0~7  [------ 用 CP8 缩短长序列关键路径 ------]
         [随后用多个 CP1/CP2 并行处理短序列]
```

只有当长序列确实主导 step 时间时，这样才可能更快；DCP solver 会比较不同分组，而不是固定
总把最长序列设为 CP8。对 dense attention，长序列的二次项使这个收益通常更明显；对 DSv4，
收益要按实际 kernel 时间确认。

#### 9.15.5 场景三：token 都装满了，但真实计算量并不相等

相同的 128K 总 token 不等于相同计算量：

```text
pack A = 1 条 128K 序列
pack B = 128 条 1K 序列
```

两边的线性层、MoE token 数接近；但带序列长度平方项的模块差别很大。对 DSv4，完整 attention
不是 dense `L²`，但仍有残余的非线性成本，例如：

```text
HCA compressed attention      近似含 sum(L_i² / 128)
CSA Indexer candidate scoring 近似含 sum(L_i² / 4)
SWA、CSA top-k attention、MLP  更接近线性项
```

verl 当前平衡公式是为 dense 7B 校准的
`24576 × L + L²`，见 [`calculate_workload()`](../../verl/utils/seqlen_balancing.py#L27)。它会比只看
token 数好，但并不是 DSv4 的真实计时模型。因此“两个 pack 都是 128K”以后，仍可能存在
straggler；DCP 改变每个 pack 使用的 CP size，可能进一步压低最长的那一项。

需要特别注意：MCore DCP scheduler 本身也使用近似 cost，并不自动知道 DSv4 每个 fused kernel
的真实耗时。所以这是**可能有收益的场景，不是必然有收益**。

#### 9.15.6 场景四：尾批或业务约束使 pack 无法任意拼接

若任意短样本都能跨组重排，并且每次都能恰好填满 128K，那么 DCP 没有额外的“装箱容量”
优势。但真实训练有时还受这些约束：

```text
最后一个 batch 样本不足
同一 UID / rollout group 必须放在一起
max_num_seqs、microbatch 或 PP divisibility 限制
在线数据到达时不能等待任意久再凑满
一条序列不可拆给两个独立 pack
```

例如受分组约束、只剩四个不能合并的 40K packs：静态 CP4 一轮只能跑两个，需要两轮；DCP
可以给每个 pack 分配 CP2，8 卡一轮跑完四个。GBS=512、可全局任意重排时，这类尾部占比
通常较小，因此收益往往只是锦上添花；小 batch 或约束较多时才会放大。

#### 9.15.7 对 DSv4 应该重点看什么

DSv4 已经用 SWA、HCA、CSA 把 dense attention 的主要 `L²` 成本降下来了，因此不能直接照搬
dense GPT 的 DCP 加速数字。它更可能从下面两点获益：

1. 短序列不再使用过大的固定 CP，从而少做 sliding-window 边界 P2P、compressed K/KV
   AllGather 等通信；
2. 极长序列的 HCA/CSA Indexer 残余成本形成关键路径时，临时给它更大的 CP group。

反过来，如果 profiler 已经显示各静态 CP4 ranks 的 step time 很接近，而且 CP 通信只占很小
比例，就不应仅因为“DCP 理论更灵活”而期待明显收益。

一个实用判断方式是同时记录：

```text
静态 packing 的有效 token 填充率
每个 pack / rank 的实际 forward+backward 时间离散程度
CP P2P 与 AllGather 的时间占比
DCP 最终选择 CP1/CP2/CP4/CP8 的频率
```

如果填充率接近 100%、耗时也均衡、通信占比很低，那么结论就是：**verl 的原有平衡已经做得
足够好，DCP 在该 workload 上确实只会是很小的增益，甚至没有增益。**

## 10. Runtime group 怎样进入 Attention

### 10.1 Standalone 的最小接线

当前 rank 根据 scheduler assignment 查询预创建 group：

```python
cp_group = parallel_state.get_dynamic_data_context_parallel_groups(
    group_size=local_cp_size,
)
```

再按该 group 切 contiguous rows，并构造：

```python
PackedSeqParams(
    qkv_format="thd",
    local_cp_size=local_cp_size,
    cp_group=cp_group,
    cp_partition_mode="contiguous",
    ...,
)
```

代码在
[`pack_dynamic_microbatch()`](./train_dsv4_dynamic_cp_sft.py#L213)。

### 10.2 Attention 临时切换到 runtime group

`DSv4HybridSelfAttention.forward()`：

```python
original_cp_group = self.pg_collection.cp
cp_group = original_cp_group

if packed_seq_params.local_cp_size is not None:
    cp_group = packed_seq_params.cp_group

self.pg_collection.cp = cp_group

# boundary P2P、compressed gather、CSA 都读取这个 runtime group
...

self.pg_collection.cp = original_cp_group
```

源码：
[`deepseek_v4_hybrid_attention.py:256`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L256)。

于是同一个 scheduled microbatch 中：

```text
ranks 0..1：在 CP2 group 上执行第 3～5 节的 Attention
rank 2：CP1，不执行 boundary P2P
rank 3：CP1，不执行 boundary P2P
ranks 4..7：在 CP4 group 上执行第 3～5 节的 Attention
```

这里沿用第 8 节的 group 布局，只说明 runtime group 怎样进入 Attention，不再重复 MCore
scheduler 为什么产生某一种具体 assignments。

### 10.3 verl 中的调用栈

verl 没有重写 MCore scheduler 或 Attention，主要负责把训练 batch 接进去：

```text
MegatronEngine.forward_backward_batch
-> verl DynamicCPScheduler.schedule
-> 每 rank 得到 sample_ids + local_cp_size
-> forward_step 读取 local_cp_size
-> gptmodel_forward_model_engine
-> preprocess_thd_engine
     -> 查询 runtime cp_group
     -> 构造 contiguous local rows
     -> 构造 PackedSeqParams
-> MCore GPTModel.forward
-> DSv4HybridSelfAttention.forward
```

关键位置：

- scheduler adapter：
  [`verl/utils/dynamic_cp_scheduler.py:112`](../../verl/utils/dynamic_cp_scheduler.py#L112)；
- engine 调度入口：
  [`transformer_impl.py:882`](../../verl/workers/engine/megatron/transformer_impl.py#L882)；
- `local_cp_size` 传入 model forward：
  [`transformer_impl.py:1277`](../../verl/workers/engine/megatron/transformer_impl.py#L1277)；
- THD preprocess：
  [`verl/models/mcore/util.py:335`](../../verl/models/mcore/util.py#L335)。

## 11. Dynamic CP 的 loss 与 backward

Standalone 每个 rank 对本地 token loss 求和：

```python
loss_sum = (token_losses * local_loss_mask).sum()
valid_tokens = local_loss_mask.sum()
loss_sum.backward()
finalize_model_grads(model, num_tokens=local_num_tokens)
```

`finalize_model_grads()` 在 DP×CP group 汇总有效 token 数，再按全局 token 数归一化梯度。
源码：
[`finalize_model_grads.py:454`](./third_party/Megatron-LM/megatron/core/distributed/finalize_model_grads.py#L454)。

Backward 通信按层类型出现：

```text
W layer：boundary gradient 反向 P2P
H layer：compressed-KV ReduceScatter + boundary gradient P2P
C layer：indexer-K/compressed-KV ReduceScatter + boundary gradient P2P
MoE：EP AllToAll 的反向通信
最后：DDP 参数梯度同步与 token normalization
```

verl 如果需要完整的 per-sequence model output，会在本 microbatch 的 runtime CP group 内
收集 local output shards，去除 padding，再由每组 leader 输出一份结果。这个步骤属于
训练框架的输出恢复，不改变 Attention forward。

## 12. 运行结果

### 12.1 静态 CP2

```bash
cd /mnt/shared-storage-user/huanghaian/code/verl/verl_hha_code/dsv4_sft_standalone
NUM_LAYERS=2 TRAIN_STEPS=1 bash run_cp2.sh
```

2026-08-21 在 8×H200 重新验证：

```text
MODEL_READY layers=2 cp=2 ep=8 fused_dsa=False local_params=60,371,633
STEP_OK step=0 loss=9.689681 grad_norm=7.647571 update=True
SFT_SMOKE_SUCCESS cp=2 layers=2 steps=1 fused_dsa=False
```

### 12.2 Dynamic CP

```bash
bash run_dynamic_cp.sh
```

```text
DCP_PLAN:
  ranks 0-3: CP4 / seq512
  ranks 4-5: CP2 / seq256
  rank 6: CP1 / seq128
  rank 7: CP1 / seq64

DYNAMIC_STEP_OK step=0 loss=9.680516 global_response_tokens=646 grad_norm=7.974459 update=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1 fused_dsa=False
```

权重随机初始化，因此 loss 只证明 forward、backward、通信和 optimizer step 成功，不
代表模型质量，也不能用静态与动态两个 loss 直接比较性能。

## 13. 只保留这些常见错误

### 13.1 DSv4 CP 必须是 THD contiguous

```text
qkv_format="thd"
cp_partition_mode="contiguous"
```

不能只改 metadata，却仍按 zigzag 切输入。

### 13.2 Local rows 不能小于 boundary

```text
local_rows >= max(csa_window_size, d_comp)
```

否则单跳左邻无法提供完整 raw window/compressor input。

### 13.3 default CP=1 不代表 runtime 永远使用 CP1

要检查的是：

```text
PackedSeqParams.local_cp_size
PackedSeqParams.cp_group
Attention 实际读取的 runtime group
```

模型只有一份；default group 和 runtime override 的区别见第 8.1 节。

### 13.4 每个 scheduled microbatch 的 collective 顺序必须一致

不同 CP 子组可以同时执行 CP4/CP2/CP1，但所有 ranks 必须执行相同数量的 transformer
layers 和 scheduled microbatches，才能与随后跨 8 ranks 的 EP AllToAll 保持一致顺序。

### 13.5 不要重复计算 CP peers 的 token 数

每个 rank 只上报本地 shard 的有效 token 数，再由 `finalize_model_grads` 汇总。把整条
sequence 长度在每个 CP peer 重复上报会错误缩放梯度。

## 14. 推荐阅读顺序

只按下面顺序看源码：

1. contiguous THD 切分：
   [`data_schedule_utils.py:15`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule_utils.py#L15)；
2. Attention 发起 boundary exchange：
   [`deepseek_v4_hybrid_attention.py:256`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L256)；
3. P2P forward/backward：
   [`csa_cp_utils.py:123`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L123)；
4. unfused Q/KV projection + RoPE：
   [`deepseek_v4_hybrid_attention.py:658`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L658)；
5. pure sliding-window `_forward_thd_cp`：
   [`csa.py:2279`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2279)；
6. HCA compressor 与 compressed-KV gather：
   [`csa.py:2332`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2332)、
   [`csa.py:2435`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2435)；
7. CSA 在压缩路径上增加的 indexer：
   [`csa.py:2361`](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2361)；
8. Dynamic groups：
   [`parallel_state.py:421`](./third_party/Megatron-LM/megatron/core/parallel_state.py#L421)；
9. Dynamic scheduler 与 verl adapter：
   [`data_schedule.py:405`](./third_party/Megatron-LM/megatron/core/datasets/data_schedule.py#L405)、
   [`dynamic_cp_scheduler.py:112`](../../verl/utils/dynamic_cp_scheduler.py#L112)；
10. MCore Draft async sampler：
    [PR #2959](https://github.com/NVIDIA/Megatron-LM/pull/2959)、
    [`data_samplers.py:195-321`](https://github.com/NVIDIA/Megatron-LM/blob/3b309c627e902913aafbb6f459f528a4ea06736c/megatron/legacy/data/data_samplers.py#L195-L321)；
11. 当前多模态 payload 缺口：
    [NVIDIA/Megatron-LM issue #5683](https://github.com/NVIDIA/Megatron-LM/issues/5683)。

先把第 1–5 步看懂，再依次看第 6 步 HCA 和第 7 步 CSA；最后看第 8–9 步当前 Dynamic
CP，再单独看第 10 步的 Draft async 原型和第 11 步的多模态限制。这正好对应“静态 pure
sliding-window → HCA → CSA → 当前 Dynamic CP → 异步调度研究原型 → 尚未解决的数据面”的
学习顺序。
