# DeepSeek-V4 SFT Forward：从 8×H200 实际运行到源码与通信

> 本文分析的是本目录独立工程实际构造出的 DeepSeek-V4 随机初始化模型，而不是只根据配置文件猜结构。reference 结论来自 8 张 H200 上的 CP=1 与 Dynamic CP 两次带 forward hook、CUDA event 和 `torch.profiler` 的运行；随后又真实跑通 fused CP=1 与 fused Dynamic CP 的完整 forward/backward/optimizer step。分析逐项对照固定版本的 Megatron-Bridge、Megatron-Core、Transformer Engine、FlashMLA、cuDNN Frontend、CuTeDSL 和 fast-hadamard-transform 源码。

## 1. 先给结论

这次实际跑通的 forward 可以压缩成下面一条主线：

```text
token ids [B,S]
  -> vocab-parallel embedding                         [MCore]
  -> THD/SBH hidden [T,1,1024]
  -> 复制为 4 条 mHC residual stream [T,1,4096]       [MCore + torch.compile]
  -> 重复 2 层：
       mHC 聚合 4->1 -> RMSNorm
       -> DSv4HybridSelfAttention
            Q-LoRA + 单头共享 KV(MQA)                  [Transformer Engine linear]
            sliding window / 4x compressed sparse attn [MCore CSA]
            grouped output projection                  [PyTorch einsum + TE linear]
       -> mHC 混合回 4 streams
       -> mHC 聚合 4->1 -> RMSNorm
       -> Hash-MoE router
       -> EP All-to-All -> 本地 expert -> All-to-All    [MCore + NCCL + TE MLP]
       -> 加 shared expert
       -> mHC 混合回 4 streams
  -> learned mHC output contraction 4->1
  -> final RMSNorm
  -> vocab projection [T,1,12800]
  -> per-token vocab-parallel cross entropy [B,S]
```

最重要的四个判断是：

1. **Bridge 不执行模型 forward。** Bridge 负责把 HF config 翻译成 MCore provider，并选择 MCore module spec；真正的 forward 在 Megatron-Core 中，线性层主要由 Transformer Engine 实现。
2. **“DeepSeek-V4-Flash”不等于无条件调用 FlashAttention/FlashMLA。** 原始 profiler case 明确设置 `apply_dsa_kernel_fusion=False`，因此记录到 MCore 的 PyTorch reference：`gather + einsum + softmax + einsum`。后来显式覆盖该开关并补齐 kernel 依赖后，H200 上的 FlashMLA forward + cuDNN DSA backward 以及 Dynamic CP 已端到端跑通；两组结果必须分开理解。
3. **Dynamic CP 是真实 MCore DSv4 contiguous-CP 路径。** 它实际出现了左边界 P2P、压缩 KV 的 CP AllGather 和 CuTeDSL layout kernel，不是用 mock 模拟。
4. **MoE 是最明确的跨卡热点。** EP=8、8 个 experts，因此每张卡一个本地 expert；每层 forward 实测 1 次 token-count AllGather 和 3 次 All-to-All。

## 2. 分析范围与可信度边界

### 2.1 哪些是实际运行，哪些不是

带 hooks/profiler 的 reference 运行条件：

| 项目 | 值 |
|---|---:|
| GPU | 8×H200 |
| dtype | BF16 |
| 层数 | 2（为了同时覆盖 ratio=0 和 ratio=4） |
| hidden size | 1024 |
| attention heads | 16 |
| head dim / RoPE dim | 256 / 32 |
| Q LoRA rank | 256 |
| output LoRA rank / groups | 256 / 4 |
| attention schedule | `[0, 4]`，即 window-only + 4x CSA |
| window / index top-k | 64 / 32 |
| experts / top-k / EP | 8 / 4 / 8 |
| mHC streams | 4 |
| TP / PP | 1 / 1 |
| HF 权重 | **不加载，随机初始化** |
| MTP | 关闭 |

另有一组 fused 验证保持 hidden size=1024、2 层、8 experts/EP8，但 kernel-facing shape
恢复为 DSv4 生产约束：64 attention heads、head dim=512、indexer=64×128、top-k=512、
window=128。原因不是为了增加模型规模，而是 FlashMLA/cuDNN DSA kernel 不接受 reference
smoke case 的 head dim=256、top-k=32。

本文没有声称：

- 两层 tiny 模型的耗时能代表 43/61 层官方模型；
- 随机权重的 loss 或 hash routing 有模型质量意义；
- 带 Python hook 和 profiler 的单次 forward 是可靠性能 benchmark；
- 当前 H200 reference kernel 是生产最优配置。

实际训练链路此前也完成了 forward、backward、梯度同步和 optimizer step：

```text
CP=1, 4 layers: loss=9.672159 -> 9.626540, finite grad norm, update=True
CP=2, 4 layers: loss=9.636520 -> 9.650787, finite grad norm, update=True
Dynamic CP, 2 layers: loss=9.679252, 646 response tokens,
                      grad_norm=7.973556, update=True
Fused CP=1, 2 layers: loss=9.679355, grad_norm=6.337935, update=True
Fused Dynamic CP, 2 layers: loss=9.643462, 646 response tokens,
                            grad_norm=7.047264, update=True
```

### 2.2 代码归属

| 组件 | 归属 | 本项目是否修改第三方源码 |
|---|---|---|
| HF config 到 provider 的翻译 | Megatron-Bridge | 否 |
| GPTModel、mHC、CSA、MoE、loss | Megatron-Core | 否 |
| Q/KV/MLP GEMM | Transformer Engine，经 MCore wrapper 调用 | 否 |
| lm-head GEMM | MCore `ColumnParallelLinear` | 否 |
| Dynamic CP layout kernel | NVIDIA Cutlass CuTeDSL，经 MCore 调用 | 否 |
| indexer Hadamard rotation | fast-hadamard-transform，经 MCore 调用 | 否 |
| fused sparse forward | FlashMLA，经 MCore adapter 调用 | **是，仅第三方 FlashMLA 的 SM90 构建裁剪** |
| fused indexer/attention backward | cuDNN Frontend `cudnn.DSA`，经 MCore 调用 | 否，安装 MCore lockfile 提交 |
| 集合通信 | PyTorch distributed + NCCL，经 MCore 调用 | 否 |
| synthetic SFT batch、运行入口、debug hooks | 本独立项目 | 是，本目录新增代码 |
| verl Python 包 | **本 forward 没有 import** | 未修改 |

这很重要：本项目模拟的是 verl 调用 Bridge/MCore 的模型侧边界，但下面分析到的 `GPTModel.forward` 及其所有核心算子，属于 MCore，不属于 verl。verl、Bridge 和 MCore 源码都没有为开启 fused DSA 而被 patch；唯一补丁位于独立的 `third_party/FlashMLA`。

### 2.3 先澄清：`CompressedSparseAttention` 同时承载 W、CSA 和 HCA

MCore 没有分别定义 `CSA`、`HCA` 两个 core-attention 类。三种 DSv4 attention layer
共用同一个
[CompressedSparseAttention](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1459)，
运行形态由本层的 `compress_ratio` 决定：

| `compress_ratio` | HF `layer_type` | MCore 实际构造 | 核心语义 |
|---:|---|---|---|
| `0` | `sliding_attention` | 无 Compressor、无 Indexer | W：只使用 local sliding-window KV |
| `4` | `compressed_sparse_attention` | Compressor + `CSAIndexer` | CSA：window + 4× compressed KV，再由 learned indexer 选择 compressed top-k |
| `128` | `heavily_compressed_attention` | 只有 Compressor，无 Indexer | HCA：window + 128× compressed KV，使用全部因果可见 compressed blocks |

Bridge 的映射表位于
[deepseek_v4_bridge.py:94](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L94)：

```python
_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
```

统一类内部先在 `compress_ratio > 1` 时构建共享的 `Compressor`，再只在
`compress_ratio == 4` 时构建 `CSAIndexer`，见
[csa.py:1520](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1520)。
HCA 的 128× 压缩已经使 compressed KV 数量很少，所以不再通过 learned indexer 二次
筛选；代码直接生成所有因果可见 compressed indices。文件名仍叫 `csa.py`，是因为它
作为整个 compressed-attention family 的实现文件，其中的 `Compressor` 本来就是 CSA
与 HCA 共用模块。

本文 standalone 的两层诊断配置是：

```text
compress_ratios = [0, 4]
layer 0 = W
layer 1 = CSA
```

因此后面的实测 shape、hook 和 profiler 只覆盖 W+CSA，没有覆盖 `ratio=128` 的 HCA；
看到模块树中只有 `CompressedSparseAttention/CSAIndexer` 并不表示 MCore 不支持 HCA。

## 3. 实际构造出的模型

### 3.1 模块树

两层诊断模型在每个 rank 上是：

```text
GPTModel
├── embedding: LanguageModelEmbedding
│   └── word_embeddings: VocabParallelEmbedding(12800, 1024)
├── decoder: TransformerBlock
│   ├── layer 0: HyperConnectionTransformerLayer
│   │   ├── input_layernorm: RMSNorm
│   │   ├── self_attention: DSv4HybridSelfAttention
│   │   │   ├── q_down: TELinear(1024 -> 256)
│   │   │   ├── q_norm: RMSNorm
│   │   │   ├── q_up: TEColumnParallelLinear(256 -> 16*256=4096)
│   │   │   ├── kv: TEColumnParallelLinear(1024 -> 256)
│   │   │   ├── kv_norm: RMSNorm
│   │   │   ├── core: CompressedSparseAttention(ratio=0, window-only)
│   │   │   ├── grouped output parameter
│   │   │   └── linear_proj: TERowParallelLinear(1024 -> 1024)
│   │   ├── pre_mlp_layernorm: RMSNorm
│   │   ├── mlp: MoELayer
│   │   │   ├── router: TopKRouter(hash layer)
│   │   │   ├── experts: SequentialMLP(1 local expert/rank)
│   │   │   └── shared_experts: SharedExpertMLP
│   │   ├── self_attention_hyper_connection: HyperConnectionModule
│   │   └── mlp_hyper_connection: HyperConnectionModule
│   └── layer 1: 同上，但 core 为 ratio=4 的 CSA
│       ├── attention Compressor(head_dim=256)
│       └── CSAIndexer
│           └── indexer Compressor(head_dim=32, Hadamard rotation)
│   └── final_layernorm: RMSNorm
└── output_layer: LinearCrossEntropyModule(1024 -> 12800)
```

两层时每个 rank 的本地参数量是 `60,371,633`。embedding 和 output 没有 tie，各自都有一个 `(12800, 1024)`、即 13,107,200 参数的矩阵。

### 3.2 一个实测发现：MoE 实际宽度是 2048，不是源码字面上的 512

`train_dsv4_sft.py` 构造 config 时同时传了：

```python
intermediate_size=2048
moe_intermediate_size=512
```

但是当前 Transformers `DeepseekV4Config.attribute_map` 把 `intermediate_size` 映射为 `moe_intermediate_size`。最终 Bridge provider 的实际快照是：

```text
ffn_hidden_size=2048
moe_ffn_hidden_size=2048
moe_shared_expert_intermediate_size=2048
```

所以当前本地 expert 和 shared expert 实际都是：

```text
fc1: 1024 -> 4096       # SwiGLU gate + up，2 * 2048
activation: SwiGLU
fc2: 2048 -> 1024
```

这不是 forward 错误，而是缩模配置的别名陷阱。若以后要把 expert 真正缩到 512，必须先修正 HF config 构造并重新验证 provider 快照，不能只看传给构造函数的字面参数。

对应位置：

- 本项目 config：[train_dsv4_sft.py:45](./train_dsv4_sft.py#L45)
- Transformers config：`.../transformers/models/deepseek_v4/configuration_deepseek_v4.py:112`
- Bridge 通用字段映射：[model_bridge.py:405](./third_party/Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py#L405)

## 4. 总入口：GPTModel 怎么把各部分串起来

MCore 的总入口是 [gpt_model.py:509](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L509)：

1. `_preprocess()` 调 embedding；
2. 把 `input_ids` 一并传给 decoder，供 hash-MoE 使用；
3. `TransformerBlock.forward()` 逐层调用；
4. `_postprocess()` 做 lm-head 与 cross entropy。

### 4.1 `multi_latent_attention=True` 不等于 core attention 是经典 MLA

这里有一个很容易被配置名误导的地方。DSv4 Hybrid 的 KV 并没有走经典 MLA 的
`kv_down -> latent KV -> kv_up` 路径，但 MCore 仍要求
`multi_latent_attention=True`。在当前实现中，这个 flag 已经不只是“选择 MLA
attention 算法”，它还同时表示启用一组 **MLA 风格的基础设施**：

- `q_lora_rank`、`qk_pos_emb_head_dim`、`v_head_dim` 等 `MLATransformerConfig` 字段；
- Q 的低秩 `q_down -> norm -> q_up` 投影；
- NoPE/PE 维度拆分以及 MLA 风格的 fused RoPE；
- 由具体 Attention 管理 RoPE，而不是由 `GPTModel` 下发全局 RoPE；
- `mla_up_proj` selective recompute 等已有代码路径。

因此应该把模块拆成“外层投影”和“core attention”两部分理解：

```text
DSv4HybridSelfAttention                 # MLA 风格的投影/RoPE 外壳
├── hidden -> q_down -> norm -> q_up    # Q 仍有低秩表示
├── hidden -> linear_kv_proj            # 不是经典 MLA latent-KV 路径
├── per-layer Rotary / YaRN
└── CompressedSparseAttention           # DSv4 的 core attention
    ├── Compressor                      # 这里才产生压缩 KV
    ├── CSAIndexer                      # ratio=4 时选择 top-k
    └── dense/window/sparse attention   # 按 W/C/H 层执行
```

这点可以直接从
[deepseek_v4_hybrid_attention.py:658](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L658)
看出来：代码虽然沿用了 `kv_compressed` 这个旧变量名，但实际上直接令
`kv_compressed = hidden_states`，并明确注释“真正的 compressed KV 在后面的 CSA
compressor 产生”。模块组装则在
[experimental_attention_variant_module_specs.py:145](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L145)：顶层是
`DSv4HybridSelfAttention`，其 `core_attention` 是 `CompressedSparseAttention`，不是
经典 MLA core。

作为对照，MCore 较早的 `experimental_attention_variant="dsa"` 确实采用
`MLASelfAttention` 外壳，只把其中的 core 替换成 `DSAttention`，见
[experimental_attention_variant_module_specs.py:93](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L93)。所以“DSA”本来就是加在 MLA
投影体系上的稀疏 core，而不是与 MLA 完全互斥的另一套 QKV 投影。

当前 DSv4 校验直接要求这个 flag 为 true，见
[transformer_config.py:1573](./third_party/Megatron-LM/megatron/core/transformer/transformer_config.py#L1573)。这更像一个复用 MLA
配置和代码路径的架构标记，名字存在一定的历史包袱；不能据此得出“DSv4 使用经典
MLA latent-KV attention”的结论。

### 4.2 为什么 DSv4 不使用 `GPTModel` 的全局 RoPE

普通 GPT 所谓“全局计算 RoPE”，只是在 `GPTModel._preprocess()` 生成一次位置角度表，
再把同一张表传给所有层；每层仍必须把它分别应用到自己产生的 Q/K。它不是在
embedding 后一次性旋转了整个模型的 Q/K。

DSv4 的 W/C/H 层不能共用同一张表：

| 层 | compression ratio | 本层 RoPE |
| --- | ---: | --- |
| W | 0 | 普通 `RotaryEmbedding` + `rotary_base` |
| C | 4 | `YarnRotaryEmbedding` + `csa_compress_rotary_base` |
| H | 128 | `YarnRotaryEmbedding` + `csa_compress_rotary_base` |

而且压缩 KV 还需要用 `0, ratio, 2*ratio, ...` 对应回原 token 位置。`GPTModel`
既不知道当前层是 W/C/H，也不知道 Compressor 的位置映射，因此
`multi_latent_attention=True` 会让
[gpt_model.py:312](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L312)
跳过全局 RoPE，由每个 `DSv4HybridSelfAttention` 按本层 ratio 创建和管理 Rotary/YaRN。
如果错误地把 flag 设为 false，`GPTModel` 会下发全局 RoPE，而
[deepseek_v4_hybrid_attention.py:240](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L240)
明确断言传入的 `rotary_pos_emb` 必须为 `None`。

“每层管理”也不等于每次 forward 都重新计算完整 sin/cos。普通 Rotary 和 YaRN 的
`forward()` 都有 LRU cache，fused 路径还有 `cos_cached/sin_cached`；相同长度和 dtype
通常直接复用缓存。每层每次都无法省掉的是把 RoPE 应用到该层的新 Q/K，因为各层
Q/K 本来就不同。

### 4.3 一次 SFT forward 从入口到 loss 的完整主线

前面的说明容易让人只看到零散模块。下面先不展开每个模块内部公式，只按真实调用
顺序把一次 forward 串起来。当前工程使用 THD packed 输入；其中 `T` 是本 rank 的
packed token 数，`H=1024`，mHC residual stream 数 `n=4`。

```text
train_dsv4_sft.py
└── GPTModel.forward(input_ids, labels, packed_seq_params)
    │
    ├── GPTModel._preprocess()
    │   └── LanguageModelEmbedding
    │       input_ids [1,T] -> embedding [T,1,H]
    │       （DSv4 不在这里生成全局 RoPE）
    │
    ├── TransformerBlock.forward()
    │   ├── mHC input_expand
    │   │   [T,1,H] -> [T,1,n*H]
    │   │
    │   └── for each HyperConnectionTransformerLayer
    │       ├── attention mHC pre
    │       │   n 条 residual stream -> 聚合后的单流 [T,1,H]
    │       ├── input RMSNorm
    │       ├── DSv4HybridSelfAttention.forward()        # 第 7 节完整展开
    │       │   ├── 本层 RoPE/CP boundary 准备
    │       │   ├── q_down -> q_up -> Q + RoPE
    │       │   ├── kv_proj -> 共享 KV + RoPE
    │       │   ├── CompressedSparseAttention.forward()  # 第 8 节展开 W/C/H
    │       │   ├── inverse RoPE
    │       │   └── grouped wo_a -> output projection
    │       ├── attention mHC post
    │       │   H_res^T @ old_streams + H_post * attention_output
    │       ├── MLP mHC pre
    │       ├── pre-MLP RMSNorm
    │       ├── MoE.forward()
    │       │   router/hash -> dispatch A2A -> local experts
    │       │   -> combine A2A -> shared expert
    │       └── MLP mHC post
    │           H_res^T @ old_streams + H_post * moe_output
    │
    ├── learned_output_contract
    │   [T,1,n*H] -> [T,1,H]
    ├── final RMSNorm
    │
    └── GPTModel._postprocess()
        ├── LM head: [T,1,H] -> [T,1,vocab]
        └── vocab cross entropy -> token loss [1,T]
```

对应的顶层源码调用链是：

1. [gpt_model.py:509](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L509)：模型入口；
2. [gpt_model.py:312](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L312)：embedding 和 RoPE 预处理；
3. [transformer_block.py:824](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L824)：第一次 PP stage 展开 mHC streams；
4. [transformer_block.py:895](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L895)：逐层调用；
5. [transformer_layer.py:1933](./third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py#L1933)：一层内严格按照 attention 再 MoE 的顺序执行；
6. [transformer_block.py:952](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L952)：末 stage 收缩四流并做 final norm；
7. [gpt_model.py:623](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L623)：LM head 和 loss。

PP>1 时，上述 `TransformerBlock` 只执行本 stage 所拥有的层；中间 stage 从 PP
P2P 接收 `[T,1,n*H]`，最后一个 stage 才执行四流收缩、final norm、LM head 和 loss。
CP>1 时每个 rank 只持有自己的 contiguous token 段，但 Attention 为窗口边界、
Compressor 和全局 sparse KV 额外通信，具体见第 9 节。

## 5. Embedding：从 `[B,S]` 到 `[T,1,H]`

CP=1 诊断输入是两条长度 128 的 packed sequence：

```text
input_ids                         (1, 256) int64
VocabParallelEmbedding output    (1, 256, 1024) bf16
LanguageModelEmbedding output    (256, 1, 1024) bf16
```

主要逻辑在：

- [language_model_embedding.py:99](./third_party/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py#L99)
- [tensor_parallel/layers.py:198](./third_party/Megatron-LM/megatron/core/tensor_parallel/layers.py#L198)

`VocabParallelEmbedding` 沿 vocabulary 维切权重。TP>1 时，每张卡只对自己词表范围内的 token 查表，其他位置置零，然后通过 TP reduce 合成完整 embedding；本次 TP=1，因此这里没有真实跨卡通信。随后从 `[B,S,H]` 转置为 MCore decoder 使用的 `[S,B,H]`。本项目使用 THD packing，dummy batch 维为 1，因此也可把第一维理解成 packed token 数 `T`。

Embedding 用的是 MCore 的 `VocabParallelEmbedding`，底层 lookup 是 PyTorch `F.embedding`，不是 Transformer Engine GEMM。

## 6. mHC：为什么层内 hidden 变成了 4096

### 6.1 Block 入口先复制成四条 stream

`TransformerBlock.forward()` 在第一个 PP stage 把：

```text
[T,1,1024] -> [T,1,4,1024] -> [T,1,4096]
```

四条 stream 初值都是 embedding 的拷贝。源码是：

- [transformer_block.py:825](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L825)
- [hyper_connection.py:584](./third_party/Megatron-LM/megatron/core/transformer/hyper_connection.py#L584)

### 6.2 Attention 和 MoE 前各有一个 mHC

每个 `HyperConnectionModule` 都接收 `[T,1,4096]`，输出实测为：

```text
aggregated  [T,1,1024]   # 真正送入 attention/MLP
H_res       [T,1,4,4]    # 四条 residual stream 的混合矩阵
H_post      [T,1,4]      # 子层结果写回四条 stream 的权重
residual    [T,1,4096]   # 原四流 residual view
```

它先用 FP32 `mapping_proj: 4096 -> 24` 产生：

```text
H_pre:  4
H_post: 4
H_res:  4*4=16
总计:   24
```

`H_pre/H_post` 经 sigmoid，`H_res` 经 Sinkhorn-Knopp 迭代投影到近似双随机矩阵。然后：

```text
单流子层输入 = sum_j(H_pre[j] * stream[j])
下一层四流状态 = H_res^T @ old_streams + H_post * sublayer_output
```

核心源码：

- mapping 和 Sinkhorn：[hyper_connection.py:30](./third_party/Megatron-LM/megatron/core/transformer/hyper_connection.py#L30)
- 聚合与 forward：[hyper_connection.py:445](./third_party/Megatron-LM/megatron/core/transformer/hyper_connection.py#L445)
- attention/MLP 外围编排：[transformer_layer.py:1933](./third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py#L1933)

本次 `use_fused_mhc=False`，因此使用 MCore 的 native PyTorch 实现，但多个小算子带 `@torch.compile`。Profiler 中确实出现多组 `CompiledFxGraph`，这就是 mHC 的投影、Sinkhorn、aggregate 和 residual mix 被 Inductor 编译后的结果。Bridge 只在 Blackwell（SM100+）默认打开 fused mHC；H200 是 Hopper，所以这里走 native compile 是预期行为。

### 6.3 Block 出口再从四流收缩成一流

所有层结束后不是简单平均，而是 `learned_output_contract()`：先在 FP32 中根据完整 4096 hidden 算 sigmoid gate，再加权求和成 1024，最后转回 BF16。源码在 [hyper_connection.py:137](./third_party/Megatron-LM/megatron/core/transformer/hyper_connection.py#L137) 和 [transformer_block.py:952](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L952)。

## 7. Attention 重点一：Q、共享 KV 与 grouped output

### 7.1 `DSv4HybridSelfAttention.forward()` 的完整执行流

这一节按代码真正执行的顺序展开，而不是按模块分类。入口位于
[deepseek_v4_hybrid_attention.py:222](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L222)。传入的
`hidden_states` 已经是 attention 前 mHC 聚合出的一条 stream，形状为 `[T,1,H]`，
而不是四流的 `[T,1,4H]`。

```text
HyperConnectionTransformerLayer._forward_attention
└── DSv4HybridSelfAttention.forward(hidden=[T,1,H])
    │
    ├─ ① 选择本 microbatch 的 CP group
    │    CP>1 时检查 THD + contiguous layout，并交换左边界 hidden
    │
    ├─ ② get_query_key_value_tensors()
    │    ├─ 获取本层 Rotary/YaRN cache
    │    ├─ hidden -> q_down -> q_norm -> q_up -> per-head q_rms
    │    ├─ hidden -> kv_proj -> kv_norm
    │    └─ 对 Q 和共享 KV 的 PE 部分应用本层 RoPE
    │
    ├─ ③ CompressedSparseAttention.forward(Q, K, V, x=hidden, qr=q_latent)
    │    ├─ THD/THD-CP/SBHD layout 分派
    │    ├─ 可选 Compressor 生成 compressed KV
    │    ├─ 构造 window indices
    │    ├─ 可选 Indexer 生成 compressed top-k indices
    │    └─ native 或 fused sparse attention
    │
    ├─ ④ 对 core attention 输出的 PE 部分做 inverse RoPE
    ├─ ⑤ grouped wo_a 低秩收缩
    ├─ ⑥ row-parallel output projection
    └─ 返回 (output=[T,1,H], bias)
         ↓
       attention mHC post：重新写回四条 residual streams
```

#### 步骤 ①：选择 CP group 和交换窗口边界

forward 先从 `packed_seq_params` 选择当前 microbatch 的 static/dynamic CP group。若
`CP>1`，DSv4 只接受 THD packed、contiguous partition；随后
`exchange_cp_boundary_hidden()` 从相邻 CP rank 取得窗口或压缩所需的左边界 token。
入口见
[deepseek_v4_hybrid_attention.py:256](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L256)。
CP=1 时这一阶段没有通信，`boundary_hidden=None`。

#### 步骤 ②：一次性产生 core 所需的五类输入

`get_query_key_value_tensors()` 位于
[deepseek_v4_hybrid_attention.py:583](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L583)，返回的不只是 Q/K/V：

```text
query         [T, num_heads, v_head_dim]  # 真正参与 attention 的多头 Q
key=value     [T, 1, 1, v_head_dim]       # 单个共享 MQA KV head
q_compressed  [T, q_lora_rank]            # qr，供 CSA Indexer 再投影
kv_compressed [T, H]                      # 实际就是原 hidden，传给 Compressor
boundary_kv                              # 仅 CP>1
```

这里的两条计算链并行来自同一个 hidden：

```text
Q 路径：
hidden [T,1,H]
  -> TELinear q_down [T,q_lora_rank]
  -> TE RMSNorm
  -> TEColumnParallelLinear q_up [T,num_heads*v_head_dim]
  -> reshape [T,num_heads,v_head_dim]
  -> torch.compile per-head RMS
  -> split(NoPE, PE) -> PE 部分 RoPE -> Q

KV 路径：
hidden [T,1,H]
  -> TEColumnParallelLinear kv_proj [T,v_head_dim]
  -> TE RMSNorm
  -> split(NoPE, PE) -> PE 部分 RoPE
  -> unsqueeze 为一个共享 KV head，且 key=value
```

Q/K 的 RoPE apply 是 MCore MLA RoPE kernel 或 PyTorch/MCore fallback；位置表则来自该层
自己持有的普通 Rotary（W）或 YaRN（C/H）缓存。当前独立基线
`apply_rope_fusion=False`，所以走
[deepseek_v4_hybrid_attention.py:808](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L808)
的 split + `apply_rotary_pos_emb()` 路径。开启 fusion 后改走
`fused_mla_rope_inplace()`，但不会把 q_down/q_up/KV projection 一起融合。

`q_compressed` 与原始 `hidden_states` 还会同时传进 core：前者供 Indexer 的 Q 分支，
后者供 Compressor 和 Indexer 的 K/权重分支。因此只看 Q/K/V 三个张量会漏掉 DSv4
sparse attention 的一半输入。

#### 步骤 ③：进入 `CompressedSparseAttention`

外层在
[deepseek_v4_hybrid_attention.py:307](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L307)
调用：

```python
core_attn_out = self.core_attention(
    query, key, value, attention_mask,
    packed_seq_params=packed_seq_params,
    x=hidden_states,
    qr=q_compressed,
    boundary_hidden=boundary_hidden,
    boundary_kv=boundary_kv,
)
```

`CompressedSparseAttention.forward()` 首先按 layout 分派，当前 packed 基线走
`_forward_thd()`，dynamic CP 走 `_forward_thd_cp()`，入口见
[csa.py:1807](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1807)。CP=1 的 THD 主干按顺序是：

1. 把单头 `key` 压成原始 KV 序列 `kv_thd`；
2. 若 `ratio>1`，`Compressor(hidden)` 按每个 packed segment 产生 compressed KV；
3. 在每个 segment 内把 original KV 与 compressed KV 拼成 `kv_full_thd`；
4. 为每个 query 构造 causal sliding-window indices；
5. ratio=4 时让 `CSAIndexer(x.detach(), qr.detach())` 预测 compressed top-k；
6. 把 window indices 与 compressed indices 合并；
7. 执行 native 或 fused sparse attention，得到 `[T,num_heads,v_head_dim]`；
8. 训练且启用 indexer loss 时，用 `DSAIndexerLossAutoScaler` 把辅助梯度挂到输出图上。

这段顺序可以直接对照
[csa.py:2559](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2559)：per-segment compression 在
2606 附近，KV 拼接在 2622 附近，窗口索引在 2628 附近，native/fused/indexer 分派在
2640 附近。

W/C/H 的区别只发生在这条公共主干的若干开关上：

| 层 | Compressor | Indexer | core 实际看到的 KV |
| --- | --- | --- | --- |
| W，ratio=0 | 无 | 无 | causal window original KV |
| C，ratio=4 | 有 | 有 | window original KV + indexer 选中的 compressed KV |
| H，ratio=128 | 有 | 无 | window original KV + 全部可见 compressed KV |

`apply_dsa_kernel_fusion=False` 时，最后一步调用 reference
`unfused_compressed_sparse_attn()`；打开后，按是否有 Indexer、是否训练分派到
FlashMLA/cuDNN DSA 组合路径。第 8.5～8.8 节是在这一步内部进一步展开 fused 与
unfused 的差异，并不是另一条独立 attention forward。

#### 步骤 ④～⑥：inverse RoPE、grouped output 和返回

core 输出回到 `DSv4HybridSelfAttention.forward()` 后还不能直接接 `W_o`：

1. reshape 为 `[T,1,num_heads,v_head_dim]`；
2. 只对每个 head 最后的 `qk_pos_emb_head_dim` 做 inverse RoPE；
3. 按 `o_groups` 把 heads 分组；
4. 每组用 `wo_a` 从组内宽维低秩收缩到 `o_lora_rank`；
5. 拼接所有组，再通过 `TERowParallelLinear` 投影回 `H`；
6. 返回 `(output, bias)` 给 `HyperConnectionTransformerLayer`，由 attention mHC post
   与旧的四流 residual 混合。

对应源码依次是：inverse RoPE
[deepseek_v4_hybrid_attention.py:345](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L345)，grouped `einsum`
[deepseek_v4_hybrid_attention.py:461](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L461)，output projection
[deepseek_v4_hybrid_attention.py:471](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L471)，mHC 写回
[transformer_layer.py:2023](./third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py#L2023)。

这一整条链中，CP=1 且 TP=1 时 Attention 本身没有跨卡通信；CP>1 增加 boundary
exchange 和 compressed/global KV 相关通信。TE Linear 名称保留了 column/row parallel
语义，但当前 DSv4 Hybrid 明确禁止 TP>1，因此 q_up、kv_proj 和 output projection
在本实现中都不会产生真实 TP collective。

### 7.2 Q 路径

以 CP=1、`T=256` 为例，真实形状是：

```text
hidden                 (256,1,1024)
q_down                 (256,1,256)
q RMSNorm              (256,256)
q_up                   (256,4096)
reshape                (256,16,256)
per-head RMS + RoPE    (256,16,256)
```

Q 是典型低秩路径：`1024 -> q_lora_rank 256 -> 16*256`。`q_down` 是重复参数的 `TELinear`，`q_up` 是 `TEColumnParallelLinear`。当前 DSv4 Hybrid 在构造和 config 校验中都强制 `TP=1`，所以没有 TP collective；这里的 column/duplicated 类型只保留了 MCore 的并行布局语义，不能理解成当前实现已经支持 `TP>1`。设计上复制 `q_down` 是为了让所有 head 分片共享完整低维 Q latent，避免在 q_down 与 q_up 之间额外 all-gather。

### 7.3 KV 路径不是 16 个 KV heads

KV 形状是：

```text
hidden -> linear_kv_proj: 1024 -> 256
kv RMSNorm
key = value = kv.reshape(T,1,1,256)
```

即 16 个 query heads 共享一个 KV head，是 MQA。代码中 `kv_compressed` 这个历史变量名容易误导；它在这里仍是 hidden-state 输入，真正的 compressed KV 是后面的 CSA `Compressor` 产生的。源码注释也专门说明了这一点：

- [deepseek_v4_hybrid_attention.py:658](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L658)
- Q/KV 投影主体：[deepseek_v4_hybrid_attention.py:583](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L583)

本次 `apply_rope_fusion=False`，所以走 MCore/PyTorch RoPE；Bridge 的 DSv4 provider 会先默认 `apply_rope_fusion=True`，独立项目为了 H200 依赖安全再覆盖为 false。这里同样应区分“Bridge 建议值”和“本次实测值”。

### 7.4 Attention 输出为什么还要 inverse RoPE

CSA core 返回 `(T,1,16*256)` 后，DSv4 先把每个 head 的 rotary 部分做 inverse RoPE，再做 grouped output projection。这与普通 MHA 直接接 `W_o` 不同。

当前 16 个 heads 被分成 4 组。每组先通过一个独立低秩参数收缩到 `o_lora_rank=256`：

```python
core_attn_out = torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)
```

四组拼回 `4*256=1024`，再经过 `TERowParallelLinear(1024 -> 1024)`。源码在 [deepseek_v4_hybrid_attention.py:461](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L461)。这减少了直接对 `16*256=4096` 做完整输出投影的代价。

## 8. Attention 重点二：CSA 到底算了什么

核心类是 [csa.py:1459](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1459)。两层故意采用不同 ratio，以覆盖两个分支。

### 8.1 Layer 0：ratio=0，只有滑窗

ratio=0 时根本不构造 `Compressor` 和 `CSAIndexer`，每个 query 只看最近 64 个原始 KV，再加一个每头可学习 attention sink。

CP=1 实测：

```text
Q                 (256,16,256)
K=V               (256,1,256)
core output       (256,1,4096)
core event time   1.981 ms
full attention    7.284 ms
```

### 8.2 Layer 1：ratio=4，滑窗 + 压缩检索

这一层有两套 compressor：

1. attention compressor：把 hidden 压成 head dim 256 的 compressed KV；
2. indexer compressor：把 hidden 压成 dim 32 的检索 K，并做 Hadamard rotation。

CP=1 下，两条长度 128 的 sequence 各得到 `128/4=32` 个 compressed rows，总计：

```text
attention compressed KV    (64,1,256)
indexer compressed K       (64,1,32)
```

Compressor 的核心逻辑在 [csa.py:796](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L796)：

1. `linear_wkv` 产生候选 value；
2. `linear_wgate` 产生 gate score；
3. 加 FP32 的 APE；
4. ratio=4 使用 `coff=2` 的相邻 group 重叠窗口；
5. gate 在 FP32 softmax；
6. 对候选加权求和；
7. RMSNorm + RoPE。

即使未来启用 FP8，compressor 的两条 GEMM 也被 `get_fp8_disabled_context` 明确保持 BF16；APE 保持 FP32。这是模型数值契约，不应为追求“全 FP8”而强行改掉。

### 8.3 Lightning indexer

Indexer 对每个 query 产生 4 个、每个 32 维的检索 head：

```text
Q index: q_lora 256 -> 4*32
K index: 自己的 compressor -> compressed rows * 32
rotation: fast_hadamard_transform
score: relu(Q_i @ K) * learned_weight_i，再对 4 heads 求和
top-k: 选最多 32 个可见 compressed blocks
```

源码是 [csa.py:1202](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1202)，Hadamard 调用在 [dsa.py:30](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py#L30)。Profiler 实测出现：

```text
HadamardTransformFn                                    count=2
fast_hadamard_transform_kernel<..., BFloat16>          count=2
```

这证明当前没有用 Python mock 跳过 indexer rotation。

### 8.4 最终 sparse attention 的 key 集合

每个 query 最终最多使用：

```text
64 个 local sliding-window KV + 32 个 indexer 选中的 compressed KV
```

当前 unfused 实现位于 [csa.py:536](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L536)，明确执行：

```text
按 topk indices gather KV
-> FP32 Q·K einsum
-> causal/invalid mask
-> 与 attention sink 一起做稳定 softmax
-> attention weights·V einsum
-> 转回 BF16
```

源码自己也注明：该 reference 实现的性能和显存占用不适合真实大模型。这就是为什么本次能证明 forward 语义跑通，却不能把它当成官方 V4-Flash 的性能结果。

### 8.5 Fused 生产路径是什么

若 `apply_dsa_kernel_fusion=True`，MCore 的 [dsa_kernels.py](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa_kernels.py) 会组合：

- sparse-attention forward：DeepSeek `flash_mla.flash_mla_sparse_fwd`；
- backward 与 indexer score：cuDNN Frontend `cudnn.DSA` 的 CuTeDSL kernel；
- top-k：TRT-LLM radix top-k。

Bridge 在 [deepseek_v4_bridge.py:105](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L105) 中只对 Blackwell 默认开启整套 fused DSv4 kernel，并检查 `cudnn.DSA` 与 `flash_mla` 是否可导入。H200 上必须先确认 FlashMLA 分支、head dim、top-k alignment 和 backward kernel 都支持 SM90。本项目完成这些逐层检查后才在 standalone override 中显式设为 true；这不是修改 Bridge 的默认判断。

另一个容易误判的开关是 `attention_backend=AttnBackend.flash`：Dynamic 示例确实设置了它，但 DSv4 experimental module spec 已经把 core 替换成 `CompressedSparseAttention`。最终是否调用 FlashMLA 仍由 `apply_dsa_kernel_fusion` 决定；generic attention backend 不能把当前 reference CSA 自动变成 FlashAttention。

### 8.6 Fused 版本究竟“fuse”了什么

先澄清一个容易造成误解的名字：`apply_dsa_kernel_fusion=True` **不是把整个 DSv4 attention 合成一个 CUDA kernel**。Q/KV projection、两个 compressor、RoPE/inverse RoPE 和 grouped output projection 仍是独立模块。它主要替换的是 CSA 内部从 indexer score/top-k 到 sparse attention forward/backward 的计算链，并用一个自定义 `autograd.Function` 协调 indexer loss 和 attention backward。

MCore 的真实分支在 [csa.py:1859](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1859)：

```text
apply_dsa_kernel_fusion = false
  -> reference unfused training/inference

apply_dsa_kernel_fusion = true
  + 有 indexer + training + grad enabled
  -> FusedIndexerSparseAttnFunc（fused training）

apply_dsa_kernel_fusion = true
  + 有 indexer + inference/no-grad
  -> cuDNN DSA indexer + top-k -> FlashMLA（fused inference）

apply_dsa_kernel_fusion = true
  + 无 indexer（ratio=0 等）
  -> FlashMLA forward + cuDNN DSA backward
```

Dynamic CP 在 [csa.py:2279](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2279) 也有同样的 fused/unfused 分支；它不会因为开启 fused DSA 就消除 CP 通信。

#### 8.6.1 Unfused 路径

Unfused 版本基本依靠 PyTorch 原生算子和普通 autograd：

```text
indexer:
  einsum(Q_index, K_index)
  -> ReLU
  -> 乘 learned head weights
  -> 跨 index heads 求和
  -> torch.topk

sparse attention:
  按 top-k indices 将 KV gather 成 [rows, topk, head_dim]
  -> FP32 einsum(Q, K)
  -> mask + attention sink + softmax
  -> FP32 einsum(P, V)
  -> BF16 output

backward:
  PyTorch autograd 逐算子回放/求导
```

Indexer 入口中的 `FusedDSAIndexerLoss` 这个名字有迷惑性：当前 fallback 内部仍会走 native `einsum`/`torch.topk`，THD 版本还会按 segment 循环。最终 sparse attention 的 reference 实现在 [csa.py:536](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L536)。它最大的代价是物化 `kv_gathered`、`scores` 和 `attn_weights`；长序列、大 top-k 时会同时增加 HBM 读写、kernel launch 和 activation 显存。

#### 8.6.2 Fused inference 路径

Fused inference 并不计算 indexer loss，也没有 backward：

```text
cuDNN Frontend DSA indexer_forward_wrapper
  -> CuTeDSL 计算 QK / ReLU / head-weight reduce / causal mask
  -> cudnn.DSA.indexer_top_k_wrapper radix top-k
  -> 合并 compressed indices 与 sliding-window indices
  -> FlashMLA flash_mla_sparse_fwd
  -> output + LSE
```

MCore adapter 在 [dsa_kernels.py:471](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa_kernels.py#L471) 组合 indexer 与 radix top-k，在 [dsa_kernels.py:82](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa_kernels.py#L82) 适配 FlashMLA。FlashMLA 不再先构造完整 `kv_gathered` 和 attention matrix 再调多个 PyTorch kernel，而是在专用 sparse-attention kernel 中按 indices 读 KV，完成 score、online softmax/attention sink 和 value accumulation。

#### 8.6.3 Fused training 路径

Fused training 的核心是 [FusedIndexerSparseAttnFunc](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa_kernels.py#L984)。它不仅换了 kernel，还重新设计了 forward/backward 之间保存什么：

```text
forward:
  1. cuDNN DSA indexer score + radix top-k
  2. 合并 indexer top-k 与 sliding-window indices
  3. FlashMLA sparse forward
       -> output, LSE, indexer-boundary LSE
  4. 按 sparse/dense indexer-loss 配置重算 target/predict score
  5. cuDNN DSA indexer backward 在 forward 阶段预计算 indexer gradients
  6. 保存 Q/KV、indices、output、LSE 和预计算的 indexer gradients

backward:
  1. cudnn.DSA.sparse_attention_backward_wrapper
       -> dQ, dKV, d(attention sink)
  2. 用真正的 grad_loss 缩放预计算 indexer gradients
       -> dQ_index, dK_index, dWeight
```

“在 forward 中预计算 indexer backward”看起来反直觉，原因是 indexer 的 KL loss 与 sparse-attention 共享 top-k 和 score/LSE 中间信息。自定义 autograd 将这条数值链集中管理，真正 backward 到来时只需使用 loss scale。注意：开启 dense indexer loss 时仍需要 dense indexer score/相关重算，所以“fused”不等于每个中间 tensor 都消失；最明确的节省来自 sparse attention 不再物化 reference 版的大 gather/attention 中间量，以及不再让 PyTorch 为整条细粒度算子链保存 autograd graph。

#### 8.6.4 逐项对比

| 环节 | Unfused | Fused | 是否真的被 fused DSA 替换 |
|---|---|---|---|
| Q/KV/indexer projection | TE `ColumnParallelLinear` 等 | 相同 | 否 |
| Attention/indexer compressor | TE GEMM + overlap/softmax/reduce/norm | 相同 | 否 |
| Indexer score | PyTorch `einsum`/ReLU/reduce | cuDNN Frontend DSA CuTeDSL | 是 |
| Top-k | `torch.topk` | `cudnn.DSA.indexer_top_k_wrapper` radix top-k | 是 |
| Sparse forward | gather + 两个 FP32 `einsum` + 分立 softmax | FlashMLA sparse prefill kernel | 是 |
| Sparse backward | PyTorch autograd | cuDNN Frontend DSA CuTeDSL wrapper | 是 |
| Indexer loss/backward | native score/loss + autograd | score recompute + DSA indexer backward；自定义 autograd 管理 | 是，但仍取决于 sparse/dense loss |
| Inverse RoPE/grouped output | 独立算子 | 相同 | 否 |
| CP boundary P2P/AllGather | 保留 | 保留 | 否 |
| CP logical-to-physical layout | CuTeDSL lowering | 同一套 CuTeDSL lowering | 否，两条路都需要 |

因此，fused 版本的本质是“**专用 kernel pipeline + 中间量复用 + 自定义 backward**”，不是一枚包含整个 attention layer 的超大 kernel。

### 8.7 Fused DSA 是不是 B 卡专用？H200 能不能用？

结论需要分成三层：

1. **Bridge 默认策略：H200 不开。** [deepseek_v4_bridge.py:105](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L105) 的 `deepseek_v4_supports_blackwell_fused_kernels()` 要求 compute capability major `>= 10`；H200 是 SM90，major 为 9。随后 `use_dsa_kernel_fusion = use_blackwell_fused_kernels and dependencies_available`，所以 AutoBridge 在 H200 上一定生成 `apply_dsa_kernel_fusion=False`。Bridge 的 H100/H200 recipe 也显式写了 false。
2. **底层 kernel 能力：不是统一的 Blackwell-only。** MCore 的 FlashMLA adapter 明确区分 SM90 top-k alignment=128 和 SM100 alignment=64，见 [dsa_kernels.py:64](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa_kernels.py#L64)。当前安装的 cuDNN Frontend 1.25.0 也会在 H200 上 dispatch 到 SM90 版 indexer forward、top-k、indexer backward 和 sparse-attention backward。FlashMLA 的 sparse prefill 亦有 SM90 实现。所以“Bridge 默认不开”不能推导成“H200 硬件无法执行”。
3. **当前 standalone 环境：已经端到端验证。** 实测 GPU 为
   `NVIDIA H200, capability=(9,0)`。直接 kernel test 完成 FlashMLA sparse forward
   `out=[16,64,512]`，以及同一输出/LSE 对应的 cuDNN DSA backward
   `dq=[16,64,512], dkv=[64,512], d_sink=[64]`，张量均 finite。随后 CP=1 和
   Dynamic CP SFT 均完成非零 indexer loss 下的 forward、backward、梯度同步和 Adam 更新。

完整 SFT 的关键成功日志是：

```text
SFT_SMOKE_SUCCESS cp=1 layers=2 steps=1 fused_dsa=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1 fused_dsa=True
```

因此现在最准确的结论是：**H200 能跑当前 MCore DSv4 fused training pipeline，但 Bridge
不会在 H200 上自动开启；本项目是在验证依赖与 shape 后通过 provider override 明确开启。**
这里还有两个不可省略的工程条件：

- CUDA 12.8 无法编译当前 FlashMLA `nv_dev` 的 Blackwell translation units。本项目给
  `third_party/FlashMLA` 应用了
  [`flashmla-sm90-prefill-only.patch`](./patches/flashmla-sm90-prefill-only.patch)，只构建
  SFT 需要的 SM90 sparse prefill；这个安装物不包含通用 decode/dense API。
- PyPI 的 `nvidia-cudnn-frontend==1.25.0` 缺少 Dynamic CP 使用的
  `q_causal_offsets` 参数。必须安装 MCore `uv.lock` 固定的
  `0a14b7181d129d30e7bad34b8c3ed0a0c995e23d` 源码提交。版本字符串相同并不代表 API
  内容相同。

SFT 比 inference 更严格，所以本文的支持结论来自 FlashMLA forward、cuDNN DSA
attention backward、非零 indexer loss/backward、THD 和 Dynamic CP layout 的组合测试，
不是由某一枚 SM90 kernel 能编译推断出来的。完整安装命令见
[`INSTALL_COMMANDS.md` 第 12 节](./INSTALL_COMMANDS.md#12-为-h200-安装-fused-dsa本机实测成功)。

### 8.8 为什么没有使用 `cudnn.CSA.csa_compressor_forward_wrapper`

这里必须把两个名字相近、支持范围不同的 fusion 分开：

```text
当前已经在 H200 跑通的 DSA fusion
  -> indexer score / top-k
  -> FlashMLA sparse-attention forward
  -> cuDNN DSA indexer/attention backward

更新的 CSA Compressor fusion
  -> 只替换 Compressor 内的 gather + APE + overlap
     + FP32 softmax + weighted reduce
  -> 当前只验证并强制允许 SM100/B200
```

因此，**H200 能运行 fused DSA，不等于也能运行 fused CSA Compressor**。

当前 MCore 固定提交仍在
[csa.py:1048](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1048)
用 PyTorch 实现 Compressor：先做 `linear_wkv/linear_wgate`，再建立 gather indices、加入
APE、执行 overlap transform、FP32 softmax 和 weighted sum。之后才在
[csa.py:1859](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1859)
根据 `apply_dsa_kernel_fusion` 选择 indexer/sparse-attention 路径。换句话说，当前开关的
实际范围是“fuse DSA”，并没有覆盖位于它之前的 CSA Compressor。

本项目锁定的 cuDNN Frontend 提交
`0a14b7181d129d30e7bad34b8c3ed0a0c995e23d` 只有 `cudnn.DSA`，还没有
`cudnn.CSA` namespace。`csa_compressor_forward_wrapper/backward_wrapper` 来自之后加入的
实验 API；本机另一份较新 cuDNN Frontend 源码为 `3e226069`（2026-08-19），才包含该
模块。即使换成新源码，其
[api.py:322](../../../cudnn-frontend/python/cudnn/csa/compressor/api.py#L322)
也会检查：

```python
capability = torch.cuda.get_device_capability(target)
if capability != (10, 0):
    raise RuntimeError(...)
```

H200 是 SM90 `(9, 0)`，所以当前公开接口会直接拒绝；B200 是 SM100 `(10, 0)`，才落在
已验证支持面。文档说明 kernel 本身没有明显的架构专属特性，将来可以扩展验证范围，
但现在不能仅删除这个检查就把它当成 H200 已支持。

此外，该 API 不是一个完整的 `nn.Module`：forward/backward wrapper 不会自动建立
PyTorch autograd，MCore 还需要自定义 `autograd.Function` 将两者接线；RMSNorm、RoPE、
Hadamard rotation 和前面的两个 projection GEMM 仍在 kernel 外。它也只支持 THD packed
pooling，而当前 contiguous/Dynamic CP 会先生成 `hidden_compact + compressed_group_ids`
这种 pre-grouped 输入，见
[csa.py:2345](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2345)，
尚不能直接套用只接收 `cu_seqlens/cu_seqlens_comp` 的 wrapper。

所以本项目当前的选择是：

| 部分 | H200 当前实现 | 原因 |
|---|---|---|
| DSA indexer / sparse attention | fused | cuDNN DSA 和 FlashMLA 都有已验证 SM90 路径 |
| CSA Compressor gated pooling | MCore native PyTorch | 新的 cuDNN CSA API 目前只放行 SM100，且尚缺 MCore autograd/CP adapter |

以后在 B200 上接入时，较合理的做法是新增独立的
`apply_csa_compressor_fusion` 开关，而不是扩大 `apply_dsa_kernel_fusion` 的隐含语义；THD、
SBHD、pre-grouped CP、确定性 backward 等不满足条件的场景仍应保留 native fallback。

## 9. Dynamic CP：真实多了哪些步骤

Dynamic case 的调度是：

```text
seq512 -> CP4
seq256 -> CP2
seq128 -> CP1
seq64  -> CP1
```

Rank 0 负责 `seq512` 的 CP4 group，所以它只保留 128 个 contiguous rows。实测：

```text
input/local embedding       (1,128) -> (128,1,1024)
boundary_hidden             (64,1,1024)
KV projection input         (192,1024) = boundary 64 + local 128
boundary_kv                 (64,1,256)
local Q/K/V                 Q(128,16,256), KV(128,1,256)
```

DSv4 要求 contiguous CP 的原因非常具体：滑窗 query 需要左邻 rank 的末尾 token，而 compressed blocks 又要有稳定的全局 sequence row 映射。核心路径在 [csa.py:2279](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L2279)：

1. 从 `PackedSeqParams.cp_group` 取本 microbatch 的 runtime CP group；
2. 向左邻 rank 接收 64 个 boundary rows，同时把自己的尾部发给右邻；
3. 构造 compressor 所需的 local + boundary compact input；
4. ratio=4 层对 indexer compressed K 做一次 CP AllGather；
5. 对 attention compressed KV 再做一次 CP AllGather；
6. 拼接 `boundary KV + local KV + rank-major compressed KV`；
7. 用 CuTeDSL 把 logical indices 降成物理 row indices；
8. 调 sparse attention。

边界交换用 `torch.distributed.batch_isend_irecv`，源码在 [csa_cp_utils.py:123](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py#L123)。Backward 会把 boundary gradient 反向发送给真正拥有这些 rows 的 rank。

CuTeDSL kernel 在 [csa_cp_layout_kernels.py](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa_cp_layout_kernels.py)。Profiler 实际看到了：

```text
dsv4_cp_build_attention_indices_kernel...             count=2, 28.447 us
dsv4_cp_compressor_input_compact_fwd_kernel...         count=1,  2.208 us
```

`build_attention_indices` count=2 是因为 ratio=0 和 ratio=4 两层都要生成本 rank 的 window/final index layout；compact compressor input 只出现在 ratio=4 层。

注意：provider 的静态 `context_parallel_size=1` 并不表示 Dynamic CP 没生效。这个示例像 verl 一样先预创建动态 DP×CP groups，再通过 `PackedSeqParams.local_cp_size/cp_group` 把 runtime group 传入 attention。MCore attention 在 [deepseek_v4_hybrid_attention.py:222](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L222) 临时切换到该 group，forward 结束再恢复静态 group。

## 10. MoE 重点：router、三次 A2A 与 shared expert

### 10.1 本 tiny 模型两层都是 Hash-MoE

Transformers 的 DSv4 config 默认前三层是 hash layers；tiny 模型只有两层，所以 Bridge 最终得到 `moe_n_hash_layers=2`。GPTModel 必须把原始 `input_ids` 送到每层 MoE router。

Hash-MoE 不是“不算 gate”：

1. router 仍用 FP32 logits 和 `sqrtsoftplus` 计算权重；
2. 但 expert id 不由 logits top-k 决定，而是查 `tid2eid[token_id]`；
3. 再从 score 中取出这些 experts 的权重并归一化。

源码在 [router.py:727](./third_party/Megatron-LM/megatron/core/transformer/moe/router.py#L727)。

因为这次不加载 HF 权重，`tid2eid` 是 MCore 为可运行性生成的 round-robin placeholder，见 [router.py:185](./third_party/Megatron-LM/megatron/core/transformer/moe/router.py#L185)。官方 checkpoint 中训练好的 `tid2eid` 由 Bridge 映射加载，见 [deepseek_v4_bridge.py:709](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L709)。因此当前 routing 能验证结构和通信，不能代表真实模型的 token-to-expert 语义。

实际 router 输出：

```text
CP=1 rank0:      probs (256,8) fp32, routing_map (256,8) bool
Dynamic rank0:   probs (128,8) fp32, routing_map (128,8) bool
```

### 10.2 EP=8 后每张卡发生什么

8 个 experts、EP=8、expert-TP=1，因此每个 rank 只有一个 local expert。MCore 的 `MoEAlltoAllTokenDispatcher` 流程是：

```text
router map
  -> 统计每个 expert 的 token 数
  -> AllGather token counts，算 variable-size splits
  -> permute token rows
  -> A2A #1: dispatch hidden rows
  -> A2A #2: dispatch routing probabilities
  -> local expert fc1/SwiGLU/fc2
  -> A2A #3: combine expert outputs
  -> unpermute 回原 token 顺序
```

对应源码：

- 总编排：[moe_layer.py:668](./third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py#L668)
- token-count metadata：[token_dispatcher.py:500](./third_party/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py#L500)
- 两次 dispatch A2A：[token_dispatcher.py:687](./third_party/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py#L687)
- combine A2A：[token_dispatcher.py:841](./third_party/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py#L841)
- local expert：[experts.py:1239](./third_party/Megatron-LM/megatron/core/transformer/moe/experts.py#L1239)

CP=1 时 rank0 local expert 收到 `(1047,1024)`。全局共有 `256 tokens/rank * 8 ranks * topk4 = 8192` 个 expert assignments，均匀时每个 expert 约 1024；1047 很接近，但这只是这批随机 token + placeholder hash table 的结果。Dynamic rank0 收到 `(465,1024)`，因为动态调度后各 rank 的输入样本不同。

### 10.3 Shared expert

所有 token 还会经过一条完整 shared SwiGLU MLP，结果直接加到 routed-expert 输出。本次 `moe_shared_expert_overlap=False`，所以它在主 stream 上单独计算。Bridge 生产默认是 true：MCore 可以把 shared expert 的 FC1/FC2 与 EP A2A 交错发射，源码在 [shared_experts.py:189](./third_party/Megatron-LM/megatron/core/transformer/moe/shared_experts.py#L189)。

本次还关闭了 grouped GEMM 和 fused permute。由于每 rank 恰好只有一个 local expert，grouped GEMM 的收益不会像“一卡多个 experts”时那么大；shared expert overlap 和通信/计算重叠通常更值得先测。

## 11. 输出层与 loss

四流收缩和 final RMSNorm 后：

```text
CP=1:     hidden (256,1,1024)
           logits (256,1,12800) bf16
           per-token loss (1,256) fp32，全部 finite

Dynamic:  hidden (128,1,1024)
           logits (128,1,12800) bf16
           per-token loss (1,128) fp32，全部 finite
```

输出模块虽然叫 `LinearCrossEntropyModule`，本次 `cross_entropy_loss_fusion=False`，所以实际先通过它继承的 MCore `ColumnParallelLinear` 物化 logits，再调用 vocab-parallel cross entropy：

- [linear_cross_entropy.py:11](./third_party/Megatron-LM/megatron/core/transformer/linear_cross_entropy.py#L11)
- [cross_entropy.py:112](./third_party/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py#L112)
- [gpt_model.py:770](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L770)

Vocab parallel CE 会依次对 logits max、target logit、sum-exp 做三次 TP AllReduce。Profiler 确实记录到 3 个 `nccl:all_reduce` API event，但 TP=1，所以没有多卡数据交换。Bridge 默认 `cross_entropy_loss_fusion=True` 和 TE implementation；对官方大词表，开启兼容的 linear/CE fusion 可以明显减少完整 logits 的峰值显存。

## 12. Forward 通信总表

下面只算 forward，不包含 backward 的 DDP gradient AllReduce、boundary gradient 回传和 optimizer 通信。

| 位置 | CP=1 实测 | Dynamic CP rank0 实测 | 实现 |
|---|---:|---:|---|
| Embedding TP reduce | 0（TP=1） | 0 | MCore TP mapping / NCCL |
| CP 左边界 P2P | 0 | 每层 1 组，共 2 个 `nccl:coalesced`；底层 SendRecv kernel 与 A2A 聚合统计，不能单独用总 count 归因 | `batch_isend_irecv` / NCCL |
| CSA compressed CP AllGather | 0 | ratio=4 层 2 次 | MCore TP mapping / NCCL |
| MoE count AllGather | 每层 1 次，共 2 | 每层 1 次，共 2 | MCore / NCCL |
| MoE All-to-All | 每层 3 次，共 6 | 每层 3 次，共 6 | MCore / NCCL |
| TP vocab CE AllReduce | API count=3，group size 1 | API count=3，group size 1 | PyTorch distributed / NCCL |
| PP Send/Recv | 0（PP=1） | 0（PP=1） | — |

Profiler 聚合结果：

```text
CP=1:
  NCCL AllGather count=2, device total ~=144.480 us
  NCCL AllToAll count=6, SendRecv device total ~=100.543 us

Dynamic CP rank0:
  NCCL coalesced boundary exchanges count=2, device total ~=164.928 us
  NCCL AllGather count=4, device total ~=83.776 us
  NCCL AllToAll count=6, device total ~=65.406 us
```

Dynamic 的 4 次 AllGather = 2 次 MoE metadata + ratio=4 CSA 的 2 次 compressed buffers。不要横向比较上面微秒数：两次 rank0 token 数不同，且它们是带 profiler 的单次观测；事件类型和 count 比绝对时间更可信。

## 13. 实际时间与显存观察

### 13.1 CP=1，rank0，256 tokens

| 模块 | CUDA event 时间 |
|---|---:|
| embedding | 0.649 ms |
| layer0 window-only core | 1.981 ms |
| layer0 full attention | 7.284 ms |
| layer0 MoE | 6.218 ms |
| layer0 total | 18.445 ms |
| layer1 ratio4 core | 12.652 ms |
| layer1 full attention | 17.872 ms |
| layer1 MoE | 4.693 ms |
| layer1 total | 26.919 ms |
| profiled peak allocated memory | 725,789,696 bytes，约 692 MiB |

ratio=4 core 比 window-only 明显更重，因为多了两套 compressor、indexer、top-k 和更宽的 gather/score 流程。这一趋势可信，但单个毫秒值不应当当 benchmark。

### 13.2 Dynamic CP，rank0，CP4 local 128 tokens

| 模块 | CUDA event 时间 |
|---|---:|
| embedding | 0.620 ms |
| layer0 window-only core | 1.403 ms |
| layer0 full attention | 7.222 ms |
| layer0 MoE | 6.457 ms |
| layer0 total | 18.692 ms |
| layer1 ratio4 core | 7.843 ms |
| layer1 full attention | 13.286 ms |
| layer1 MoE | 4.421 ms |
| layer1 total | 22.152 ms |
| profiled peak allocated memory | 578,725,888 bytes，约 552 MiB |

Dynamic 本地 token 减半，但增加了 boundary/AllGather/layout 开销，因此也不能用上述表直接推导 CP scaling efficiency。

### 13.3 为什么 hook 时间只能看趋势

CUDA kernel 是异步发射的，hook 中 event 能较好覆盖同一 stream 上的工作，但 NCCL stream、shared stream、allocator 释放和 profiler instrumentation 会改变调度。某些 module 的 `memory_delta` 甚至是负数，因为其生命周期内其他 activation 被释放。要做正式性能分析，应移除 hooks，多 warmup，用 Nsight Systems 看 timeline，再用 Nsight Compute 看 sparse attention/GEMM kernel。

## 14. H200 上如何做性能优化

建议按下面顺序，而不是一次打开所有开关。

### 14.1 第一优先级：替换 reference sparse attention

reference 模式的最大结构性问题是 [csa.py:536](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L536) 的 gather/einsum 路径。长上下文时，它会构造大 gather tensor，显存和带宽效率都差。

本项目现在已完成 H200 上匹配当前 MCore commit 的 FlashMLA SM90 forward、cuDNN DSA
backward、非零 indexer loss，以及 packed variable-length Dynamic CP 的完整 optimizer
step，因此可直接用 `run_fused_cp1.sh` 和 `run_fused_dynamic_cp.sh` 作为后续优化基线。
但当前 smoke test 没有做 reference/fused 数值对齐和正式吞吐 benchmark，也没有覆盖官方
层数与超长上下文；“能跑通”不应写成“性能已经最优”。下一阶段应依次补：固定随机输入的
forward/gradient tolerance、长序列峰值显存、warmup 后吞吐、Nsight Systems timeline。

Bridge 的默认架构判断仍保持原样。standalone override 是实验配置，不应反向解释成
Bridge 官方承诺 H200 默认支持。

### 14.2 RoPE 与 mHC

- 若当前 MCore 的 fused MLA RoPE extension 在 H200 可用，可独立打开 `apply_rope_fusion`；先检查 forward/inverse RoPE 与 packed CP position id。
- mHC 在 H200 上继续用 native `torch.compile` 是合理选择。第一个真实 step 前必须 warmup **grad-enabled** graph；`no_grad` warmup 会编译另一份图。
- mHC 保存四流 activation，内存压力明显。长上下文可按 MCore 支持配置 selective recompute，尤其 `recompute_modules` 中的 `mhc`；代价是 backward 重算 Sinkhorn/aggregate。

### 14.3 MoE 通信与计算

- 先打开 `moe_shared_expert_overlap`，观察 shared FC1/FC2 是否成功隐藏在两次 dispatch/一次 combine A2A 后面。
- `moe_permute_fusion` 可减少排序与搬运 kernel，但要确认对应扩展版本。
- 当前每 rank 一个 expert，grouped GEMM 收益可能有限；官方 256 experts 或不同 EP 布局下，一卡多 expert 时再重点测试 grouped GEMM。
- EP group 应尽量落在高速 NVLink/NVSwitch 域内。当前每层固定 3 次 A2A，拓扑比单个 GEMM 微调更重要。
- 正式训练一定要加载 checkpoint 的 `tid2eid`；placeholder hash table 即使负载看似平衡，也不具有模型语义。

### 14.4 Dynamic CP

- `max_seqlen_per_dp_cp_rank` 应根据显存和 kernel tile 调整；它决定 64/128/256/512 等序列被分到 CP1/2/4 的阈值。
- contiguous layout 是当前 DSv4 CP 的硬约束，不能换回 zig-zag 而仍假定 window boundary 正确。
- 每个 CP layer 都有 64-row boundary P2P；ratio=4 还多两个 compressed AllGather。短序列不应强行上大 CP，否则通信会盖过计算节省。
- 变长数据应让 scheduler 尽量平衡每 rank 本地 token，而不是只平衡 sample 数。

### 14.5 Output 与精度

- 官方词表很大，优先验证 TE/linear cross-entropy fusion，避免完整 logits 长时间驻留。
- 当前整体是 BF16，不是 FP8/MXFP4 compute。即使将来使用 FP8，compressor gate/value 和部分 indexer projection 按源码仍应保持 BF16，APE/mHC mapping/router 保持 FP32。
- 开 FP8/MXFP4 时分清“checkpoint weight storage/transfer dtype”和“forward compute recipe”；它们不是同一件事。

## 15. Bridge 如何决定这些模块，但不参与 forward

Bridge 的 DSv4 注册和 provider 翻译在 [deepseek_v4_bridge.py:349](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L349)。它设置：

- `experimental_attention_variant="dsv4_hybrid"`；
- MLA/Q-LoRA/grouped output 几何；
- per-layer compression ratio 与 indexer；
- mHC；
- MoE dispatcher/router/hash layers；
- 生产 fused 开关。

### 15.1 从 Bridge 到 DSv4 核心 layer spec 的完整调用栈

这条链比较难找，因为它跨了三个阶段，而且同一个函数在 Bridge 中还被起了别名：

```text
阶段 A：Bridge 只把“构建函数”存进 provider，还没有生成 ModuleSpec

AutoBridge.from_hf_config(...)
  -> DeepSeekV4Bridge.provider_bridge(...)
       provider.experimental_attention_variant = "dsv4_hybrid"
       provider.transformer_layer_spec = _get_exp_attn_spec
                                         │
                                         └─ 实际就是：
                                            get_transformer_block_with_
                                            experimental_attention_variant_spec

阶段 B：provide_distributed_model() 延迟调用该函数，生成整块 spec

provider.provide_distributed_model(...)
  -> Bridge get_model(...)
  -> _create_model(...)
  -> GPTModelProvider.provide(...)
       transformer_layer_spec = self.transformer_layer_spec
       transformer_layer_spec(self, vp_stage=...)
  -> get_transformer_block_with_experimental_attention_variant_spec(config)
       -> _get_backend_spec_provider(config)
            -> TESpecProvider
       -> get_transformer_layer_with_experimental_attention_variant_spec(config, backend)
            -> get_experimental_attention_variant_module_spec(config, backend)
                 -> config.experimental_attention_variant == "dsv4_hybrid"
                 -> get_dsv4_hybrid_module_spec_for_backend(config, backend)
                      └─ 这里才定义 DSv4 attention 的核心嵌套 ModuleSpec
            -> 将 attention spec + MoE spec + mHC
               包进每层 HyperConnectionTransformerLayer spec
       -> 按 PP/VP stage 截取本 rank 的 layer specs
       -> 返回 TransformerBlockSubmodules

阶段 C：MCore 根据 ModuleSpec 递归实例化真正的 nn.Module

GPTModel(..., transformer_layer_spec=TransformerBlockSubmodules)
  -> TransformerBlock
  -> build_module(...)
       -> HyperConnectionTransformerLayer
       -> DSv4HybridSelfAttention
       -> CompressedSparseAttention / Compressor / CSAIndexer
       -> MoELayer / TopKRouter / experts
```

对应源码位置：

| 调用节点 | 代码位置 | 此时拿到的东西 |
|---|---|---|
| Bridge 导入并重命名 | [deepseek_v4_bridge.py:67](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L67) | Python 函数别名 `_get_exp_attn_spec` |
| 函数保存进 provider | [deepseek_v4_bridge.py:403](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L403) | 尚未调用的 callable |
| `provide()` 延迟调用 | [gpt_provider.py:254](./third_party/Megatron-Bridge/src/megatron/bridge/models/gpt_provider.py#L254) | `TransformerBlockSubmodules` |
| block spec 入口 | [experimental_attention_variant_module_specs.py:350](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L350) | 整个 block 的 spec |
| 逐层组合 attention/MLP/mHC | [experimental_attention_variant_module_specs.py:227](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L227) | `list[ModuleSpec]` |
| 根据 variant 分发 | [experimental_attention_variant_module_specs.py:203](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L203) | DSv4/DSA/GatedDeltaNet 三选一 |
| **DSv4 核心 attention spec** | [experimental_attention_variant_module_specs.py:145](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L145) | `DSv4HybridSelfAttention` 的嵌套 `ModuleSpec` |
| 递归实例化 spec | [spec_utils.py:74](./third_party/Megatron-LM/megatron/core/transformer/spec_utils.py#L74) | 真正的 `nn.Module` |

因此打印 provider 时看到：

```text
<function get_transformer_block_with_experimental_attention_variant_spec at 0x...>
```

是正常的：provider 此时保存的是一张“如何生成施工图的函数”，不是已经构造好的 layer，也不是 GPU 上的 module。直到 `GPTModelProvider.provide()` 才调用它。

### 15.2 `get_dsv4_hybrid_module_spec_for_backend()` 到底决定了什么

真正的 DSv4 attention 施工图在 [experimental_attention_variant_module_specs.py:145](./third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py#L145)。它组装出的嵌套关系是：

```text
ModuleSpec(DSv4HybridSelfAttention)
├── linear_q_down_proj = backend.linear()
├── linear_q_up_proj   = backend.column_parallel_linear()
├── linear_kv_proj     = backend.column_parallel_linear()
├── q_layernorm / kv_layernorm
├── core_attention = ModuleSpec(CompressedSparseAttention)
│   ├── compressor = ModuleSpec(Compressor)
│   │   ├── linear_wkv
│   │   ├── linear_wgate
│   │   └── RMSNorm
│   └── indexer = ModuleSpec(CSAIndexer)
│       ├── linear_wq_b
│       ├── linear_weights_proj
│       └── 自己的 Compressor
└── linear_proj = backend.row_parallel_linear()
```

这里的 `backend` 来自 `TESpecProvider`，所以当前具体解析为 `TELinear`、`TEColumnParallelLinear`、`TERowParallelLinear` 和 TE norm。也就是说，这个函数同时决定了：

- attention 顶层类是 `DSv4HybridSelfAttention`；
- core 算法类是 `CompressedSparseAttention`；
- compressor/indexer 的组合关系；
- 每个 projection 采用哪种 TP 语义和哪个 backend；
- Q/KV norm 不能和 projection 融合，因为 DSA indexer 需要 normalized Q；
- attention mask 是 causal；
- input layernorm 不与 attention 融合。

不过它提供的是一张可复用的结构图，并没有在这里按层直接删除 compressor/indexer。真正实例化 `CompressedSparseAttention` 时，构造函数再读取该层的 `compress_ratio`：

```text
ratio=0   -> 不实例化 compressor，也不实例化 indexer
ratio=4   -> 实例化 attention compressor + CSAIndexer(+ indexer compressor)
ratio=128 -> 只实例化 attention compressor，不实例化 indexer
```

条件构造在 [csa.py:1519](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1519)。这解释了为什么所有层共享同一种 DSv4 `ModuleSpec`，最终打印出来的 layer 0 和 layer 1 模块树却不同。

以后想快速定位这条链，可以直接依次搜索：

```bash
rg -n "transformer_layer_spec = _get_exp_attn_spec" third_party/Megatron-Bridge/src
rg -n "experimental_attention_variant == \"dsv4_hybrid\"" third_party/Megatron-LM/megatron
rg -n "def get_dsv4_hybrid_module_spec_for_backend" third_party/Megatron-LM/megatron
```

从 `module(...)` 开始执行后，Bridge 已经退出 forward 调用栈。若以后看到一个性能问题，先判断它属于：

- 配置/模块选择错误：看 Bridge/provider；
- 算法与张量布局：看 MCore DSv4/CSA/MoE；
- GEMM/norm：看 Transformer Engine；
- CP layout：看 MCore CuTeDSL wrapper；
- 通信：看 MCore dispatcher 与 NCCL timeline。

## 16. 复现实验

诊断脚本是 [trace_dsv4_forward.py](./trace_dsv4_forward.py)，它只在 rank0 注册只读 hooks 和 profiler，但所有 8 ranks 都参加真实 collective。

```bash
# CP=1；两条 128-token packed sequences
MODE=cp1 NUM_LAYERS=2 bash run_forward_trace.sh

# Dynamic CP；rank0 会进入 seq512 的 CP4 group
MODE=dynamic NUM_LAYERS=2 bash run_forward_trace.sh

# 直接验证 H200 上 FlashMLA forward + cuDNN DSA backward
/mnt/shared-storage-user/huanghaian/miniconda3/envs/dsv4_sft_bridge/bin/python \
  verify_fused_dsa_sm90.py

# 完整 fused SFT：非零 indexer loss，含 backward 和 optimizer step
bash run_fused_cp1.sh
bash run_fused_dynamic_cp.sh
```

成功标志：

```text
TRACE_FORWARD_RESULT ... finite=True
TRACE_FORWARD_SUCCESS mode=cp1
TRACE_FORWARD_SUCCESS mode=dynamic
FUSED_DSA_SM90_VERIFY_SUCCESS
SFT_SMOKE_SUCCESS cp=1 layers=2 steps=1 fused_dsa=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1 fused_dsa=True
```

脚本会打印：

- provider 的最终值，而不是只打印 HF config；
- 模块 Python 类型、直属参数数目和参数形状；
- 每个核心模块的输入/输出 shape、CUDA event 时间和 allocated-memory delta；
- profiler 中的通信事件；
- CuTeDSL、Hadamard、softmax、CUTLASS/CompiledFxGraph 等重点 kernel。

## 17. 推荐的阅读顺序

如果只想抓住 forward，不建议从 Bridge weight mapping 开始。按下面顺序读最快：

先读本文的 **4.3 全模型主线**，再读 **7.1 单个 Attention 完整执行流**；遇到
Compressor/Indexer 分支时跳到第 8 节，遇到通信时再查第 9 和第 12 节。对应源码建议
按下面顺序跟：

1. [gpt_model.py:509](./third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py#L509)：总入口；
2. [transformer_block.py:824](./third_party/Megatron-LM/megatron/core/transformer/transformer_block.py#L824)：mHC expand，然后从 895 附近进入层循环、从 952 附近 contract；
3. [transformer_layer.py:1933](./third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py#L1933)：每层 attention + MoE 外围；
4. [deepseek_v4_hybrid_attention.py:222](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py#L222)：Q/KV、CP boundary、inverse RoPE、grouped output；
5. [csa.py:1807](./third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py#L1807)：core layout 分派；THD 主干从 2559 附近继续看 window/compressor/indexer/sparse attention；
6. [moe_layer.py:668](./third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py#L668) 与 [token_dispatcher.py:375](./third_party/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py#L375)：MoE 和三次 A2A；
7. 最后再看 [deepseek_v4_bridge.py:390](./third_party/Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L390)：这些 module 为什么被选中。

一句话建立心智模型：**Bridge 决定“搭什么模型”，MCore 决定“forward 怎么走”，TE/FlashMLA/CuTeDSL/FHT 决定“某一段在 GPU 上怎么跑”，NCCL 决定“分片之间怎么交换数据”。**
