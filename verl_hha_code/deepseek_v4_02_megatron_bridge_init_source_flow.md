# DeepSeek-V4 学习 02：Megatron-Bridge 初始化、MCore 建模与权重导入源码串讲

> 目标：只解释下面五个调用如何把 DeepSeek-V4 的 HF checkpoint 变成分布式 Megatron 模型，不展开 GRPO、rollout 和优化器。
>
> 基线：`verl v0.9.0`（tag `483b8a0`）示例固定使用 Megatron-Bridge `c7774d4` 和 Megatron-Core/Megatron-LM `fd1121b8`。

```python
bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
provider.apply_overrides_and_finalize(dtype=torch.bfloat16, overrides=provider_overrides)
module = provider.provide_distributed_model(...)
bridge.load_hf_weights(module, model_path)
```

## 1. 先建立正确的心智模型

这五步不是“Bridge 创建了一个自己的模型”，而是：

```text
HF config + checkpoint 文件
          │
          │ [Megatron-Bridge] 识别模型类型、翻译配置
          ▼
MLAModelProvider
  = 一份可延迟 finalize 的 MCore TransformerConfig
  + 一个能够创建 GPTModel 的 factory
          │
          │ [verl] 覆盖本次运行的 TP/PP/EP/ETP/CP、BF16 等
          │ [Bridge -> MCore] finalize 并校验配置
          ▼
最终 provider
          │
          │ [Bridge] 编排 PP/VPP、GPU、BF16、DDP 包装
          │ [MCore] 根据 ModuleSpec 实例化真正的模块
          ▼
每个 rank 上的局部分片 module: list[GPTModel wrapper]
          │
          │ [Bridge] 懒加载 HF tensor、反量化、改布局、按 TP/EP/PP 分发
          ▼
装入 checkpoint 权重的 BF16 Megatron 分布式模型
```

四个工程的职责要分清：

| 标记 | 工程 | 在这条链里的职责 |
|---|---|---|
| `[verl]` | verl | 初始化并行组；准备运行时 overrides；调用 Bridge；决定 actor 是否包 DDP；加载后做可选 patch |
| `[Bridge]` | Megatron-Bridge | HF 架构识别；HF config → provider；模型创建编排；HF/Megatron 参数名和布局转换 |
| `[MCore]` | Megatron-Core（Megatron-LM 中的 `megatron/core`） | 真正的 `GPTModel`、Transformer block、DSv4 attention、CSA、mHC、MoE、并行层及通信实现 |
| `[TE]` | Transformer Engine | 被 MCore ModuleSpec 选中的线性层、RMSNorm、Grouped GEMM 等实现及部分运行时 kernel |

一句话回答“只看 Bridge 够不够”：**看懂配置和权重转换主要看 Bridge；看懂最终 module 长什么样、有哪些层和算子，必须继续进入 MCore；要确认最终跑到哪个 CUDA kernel，还要结合 TE/FlashMLA 以及一次实际 forward。**

## 2. 版本基线：不要混着读当前 main

`examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh` 在 v0.9.0 中明确写了：

```text
Megatron-Bridge c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
Megatron-LM     fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
```

对应源码：

- [Megatron-Bridge c7774d4](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11)
- [Megatron-LM fd1121b8](https://github.com/NVIDIA/Megatron-LM/tree/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1)

截至本文整理时，你的本地 sibling 仓库是：

```text
../Megatron-Bridge  main b864189731ae15c5f891f5d543f638d2213348cc
../Megatron-LM      main 4b18b260f012c8de51f729fb09771f99266bc675
```

后者当前 main 中搜索不到 `dsv4_hybrid`，因此这两个 HEAD **不能作为 v0.9.0 DSv4 路径的配套代码直接运行**。本文用上述两个精确 commit 解读，但文件路径仍采用仓库内路径，方便对照：

```bash
git -C ../Megatron-Bridge rev-parse HEAD
git -C ../Megatron-LM rev-parse HEAD
rg -n 'dsv4_hybrid' ../Megatron-LM/megatron/core
```

## 3. 调用发生之前：verl 先初始化并行世界

源码入口：

```text
[verl] verl/workers/engine/megatron/transformer_impl.py
       MegatronEngine.__init__ / _build_tf_config / _build_megatron_module
```

在创建 provider 以前，verl 已调用 MCore 的 `mpu.initialize_model_parallel(...)`，传入：

```text
TP、PP、VPP、CP、EP、ETP
```

所以后面 `provide_distributed_model()` 看到的是已经建立好的 NCCL process groups。Bridge 自身也能在独立脚本中初始化分布式环境，但在 verl 路径里它不会重新创建一套并行组，而是通过：

```python
ProcessGroupCollection.use_mpu_process_groups()
```

取得 verl/MCore 已初始化的 TP、PP、DP、CP、EP 等 group。

这点很重要：**模型分片的拓扑由 verl 提前确定，Bridge 读取并使用它，MCore 模块按它分配本 rank 的层和参数。**

---

## 4. 第一步：`AutoBridge.from_hf_pretrained(...)`

### 4.1 `[verl]` 实际只是重新导出第三方 `AutoBridge`

v0.9.0 的：

```text
verl/models/mcore/bridge.py
```

核心就是：

```python
from megatron.bridge import AutoBridge
```

所以 `AutoBridge.from_hf_pretrained` 的主体不是 verl 实现，而在：

```text
[Bridge] src/megatron/bridge/models/conversion/auto_bridge.py
```

verl 这个文件还带有旧版本 Bridge 的 value model、freeze router 兼容 fallback；这些是 verl 的兼容代码，但与 DSv4 架构识别无关。

### 4.2 它不会先创建一份完整 HF GPU 模型

`from_hf_pretrained()` 主要做两件事：

1. 用 `safe_load_config_with_retry()` 读取和校验 HF config。
2. 创建 Bridge 自己的 `PreTrainedCausalLM` 包装对象。

这里的 `PreTrainedCausalLM` 位于：

```text
[Bridge] src/megatron/bridge/models/hf_pretrained/causal_lm.py
[Bridge] src/megatron/bridge/models/hf_pretrained/base.py
[Bridge] src/megatron/bridge/models/hf_pretrained/state.py
```

它不是 `transformers.AutoModelForCausalLM.from_pretrained()`。它对 config 和 checkpoint state 都是懒加载的：

- `.config` 第一次访问时才加载 HF config；
- `.state` 对 safetensors 建立 `SafeTensorsStateSource`；
- 读取索引/header 得到有哪些 key；
- 真正转换某个参数时，才从对应 shard 取出 tensor，通常先在 CPU 上出现。

因此第一步结束后的对象更像：

```text
AutoBridge(
  hf_pretrained = lazy HF config + lazy safetensors state source
)
```

此时没有 MCore `GPTModel`，也没有为了导入权重而额外常驻一份完整 HF GPU 模型。

### 4.3 如何识别 DeepSeek-V4

`AutoBridge` 从 HF config 的 `architectures`/`auto_map` 得到源架构名：

```text
DeepseekV4ForCausalLM
```

然后查询 Bridge 注册表。DSv4 的注册发生在：

```text
[Bridge] src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py
```

```python
@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(MegatronModelBridge):
    ...
```

这张注册表给后续三件事定了方向：

- 源 checkpoint 按 `DeepSeekV4Bridge` 的规则解释；
- 目标模型类是 MCore `GPTModel`；
- 中间 provider 类型是 `MLAModelProvider`。

---

## 5. 第二步：`bridge.to_megatron_provider(load_weights=False)`

### 5.1 provider 到底是什么

类型关系位于：

```text
[Bridge] src/megatron/bridge/models/mla_provider.py
[Bridge] src/megatron/bridge/models/gpt_provider.py
[Bridge] src/megatron/bridge/models/transformer_config.py
```

简化后是：

```python
class MLAModelProvider(MLATransformerConfig, GPTModelProvider):
    pass

class GPTModelProvider(TransformerConfig, ModelProviderMixin):
    def provide(...):
        return MCoreGPTModel(...)
```

所以 provider 同时具有两种身份：

1. **配置对象**：包含 MCore `TransformerConfig`/`MLATransformerConfig` 的全部字段；
2. **模型工厂**：其 `provide()` 能调用 MCore `GPTModel(...)`。

它目前仍不是 `nn.Module`，不包含参数，也不占用模型显存。

### 5.2 第一层翻译：通用 HF config → MCore 字段

`MegatronModelBridge.provider_bridge()` 先调用通用 `CONFIG_MAPPING`。典型映射包括：

| HF config | provider/MCore config |
|---|---|
| `num_hidden_layers` | `num_layers` |
| `hidden_size` | `hidden_size` |
| `intermediate_size` | `ffn_hidden_size` |
| `num_attention_heads` | `num_attention_heads` |
| `num_key_value_heads` | `num_query_groups` |
| `vocab_size` | `vocab_size` |
| `max_position_embeddings` | `seq_length` |
| `rms_norm_eps` | `layernorm_epsilon` |
| `n_routed_experts` | `num_moe_experts` |
| `num_experts_per_tok` | `moe_router_topk` |
| `q_lora_rank` | `q_lora_rank` |
| `kv_lora_rank` | `kv_lora_rank` |
| `qk_nope_head_dim` | `qk_head_dim` |
| `qk_rope_head_dim` | `qk_pos_emb_head_dim` |
| `num_nextn_predict_layers` | `mtp_num_layers` |

它还根据 HF dtype 填 `params_dtype/fp16/bf16`，翻译 activation，并处理 RoPE/YaRN 的通用部分。

### 5.3 第二层翻译：DSv4 专属语义

接着进入 `DeepSeekV4Bridge.provider_bridge()`。这一步不是简单重命名，而是给 MCore 填入“HF config 本身不能直接表达的实现选择”。核心设置如下。

#### Attention/MLA

```python
provider.experimental_attention_variant = "dsv4_hybrid"
provider.multi_latent_attention = True
provider.transformer_layer_spec = _get_exp_attn_spec
provider.qk_layernorm = True
provider.normalization = "RMSNorm"
provider.add_bias_linear = False
```

并设置 DSv4 的 MLA 几何，例如 `q_lora_rank`、`v_head_dim`、`qk_pos_emb_head_dim`、`o_groups`、`o_lora_rank`。

最关键的是：

```python
provider.transformer_layer_spec =
    get_transformer_block_with_experimental_attention_variant_spec
```

这条 callable 是后面创建最终 module tree 的入口。Bridge 并不自己写 DSv4 forward，而是让 provider 在建模时向 MCore 请求 DSv4 的 `ModuleSpec`。

#### CSA 与 DSA indexer

Bridge 把 HF 的 `layer_types/compress_rates` 或旧版 `compress_ratios` 展平为：

```text
provider.csa_compress_ratios: 每个主层和 MTP 层各一个 ratio
provider.csa_window_size
provider.dsa_indexer_n_heads
provider.dsa_indexer_head_dim
provider.dsa_indexer_topk
```

ratio 的含义是：

| ratio | 层行为 |
|---:|---|
| `0` | sliding-window attention，不创建 compressor/indexer |
| `4` | compressed sparse attention，创建 compressor 和 learned indexer |
| `128` | heavily compressed attention，创建 compressor，直接使用高度压缩表示 |

#### mHC / Hyper-Connections

```python
provider.enable_hyper_connections = True
provider.num_residual_streams = hf_config.hc_mult       # Flash 通常为 4
provider.mhc_sinkhorn_iterations = hf_config.hc_sinkhorn_iters
```

Bridge 会按 GPU capability 决定 Blackwell 专用融合实现的默认值。v0.9.0 的 H200 示例又显式设置 `use_fused_mhc=False`；H200 不是 Blackwell，因此不要把 mHC 结构与 Blackwell fused mHC kernel 混为一谈。

#### MoE

关键项包括：

```text
gated_linear_unit=True
moe_grouped_gemm=True
moe_token_dispatcher_type=alltoall
moe_router_topk=6
moe_router_score_function=sqrtsoftplus
moe_n_hash_layers=3
activation_func_clamp_value=10.0
moe_layer_freq=[1] * num_hidden_layers
moe_shared_expert_intermediate_size=...
```

DSv4 的前几个 hash-routing 层还拥有 `tid2eid` buffer，即 token id 到 expert id 的查表。

#### MTP

Bridge 先忠实读取：

```python
provider.mtp_num_layers = hf_config.num_nextn_predict_layers
```

是否真正保留，后面由 verl 的训练配置决定。

### 5.4 `load_weights=False` 的准确含义

`to_megatron_provider(load_weights=False)` 的含义是：

- 不把 HF 权重加载注册成 provider 的 pre-wrap hook；
- 不把 `perform_initialization` 改成 `False`；
- 只返回已经翻译过的 provider；
- 由 verl 在 module 构建完成后显式调用 `bridge.load_hf_weights(...)`。

它**不等于**“MCore 建模时完全不初始化参数”。默认建模路径仍可能先生成随机/占位初值，随后被 checkpoint 权重覆盖。`load_weights=True` 才是 Bridge 原生的“建模前关闭随机初始化，并在 DDP 包装前自动加载”组合路径。

verl 选择拆成两步，便于自己插入 value model、PEFT、DDP、distributed checkpoint 等编排。

---

## 6. 第三步：`apply_overrides_and_finalize(...)`

### 6.1 为什么一定要 finalize

Bridge 的 `TransformerConfig` 刻意覆盖了 `__post_init__`：

```python
def __post_init__(self):
    pass
```

这使得 provider 构造后还能安全修改 TP/PP/EP/CP 等字段。`finalize()` 才会调用真正的：

```python
MCoreMLATransformerConfig.__post_init__(self)
```

因此配置生效顺序是：

```text
HF 架构参数
  → Bridge 通用映射
  → Bridge DSv4 专属设置
  → verl 本次运行 overrides
  → MCore __post_init__ 校验和派生字段计算
```

后写入的 verl override 优先级最高，但最终必须通过 MCore 的约束检查。

### 6.2 `[verl]` 写入了什么

v0.9.0 的 `_build_tf_config()` 准备的核心 overrides 是：

```text
tensor_model_parallel_size
pipeline_model_parallel_size
expert_model_parallel_size
expert_tensor_parallel_size
virtual_pipeline_model_parallel_size
context_parallel_size
sequence_parallel
overlap_p2p_comm
batch_p2p_comm=False
variable_seq_lengths=True
attention_backend=AttnBackend.flash
moe_token_dispatcher_type=alltoall
moe_router_load_balancing_type=none
```

然后再合并用户的 `override_transformer_config`。

这属于 **配置覆盖**，不是 monkey patch：verl 没有重写 provider 类或 MCore 构造函数。

### 6.3 DSv4 的两个额外处理

#### 默认关闭训练端 MTP

如果 HF checkpoint 带 MTP，但 verl 配置没有启用 MTP：

```python
provider_overrides["mtp_num_layers"] = 0
provider_overrides["csa_compress_ratios"] =
    provider.csa_compress_ratios[: provider.num_layers]
```

第二行必须同步裁掉 MTP 对应的 ratio，否则 ratio 数量与最终层数不一致。

#### PP layout

v0.9.0 示例是：

```text
Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5L
```

可理解为 PP=8 时：

- 第一 stage：embedding `E` + 6 个 decoder `t`；
- 中间 stage：各自 5 或 6 个 decoder；
- 最后一 stage：decoder + loss/head `L`；
- 总计 43 个 decoder layer。

DSv4 的 hash-routing 层要求与 embedding 一起位于第一 stage，因此 Bridge 在 PP>1 且用户没给 layout 时也会尝试自动生成一份；示例显式给出，所以显式值优先。

### 6.4 finalize 做什么

Bridge 的 `apply_overrides_and_finalize()`：

1. 设置 `params_dtype=torch.bfloat16`，同步 `bf16=True/fp16=False`；
2. 检查每个 override 确实是 provider 字段；
3. 必要时生成 DSv4 pipeline layout；
4. 调用 provider 的 `finalize()`；
5. 最终进入 MCore `MLATransformerConfig.__post_init__()`。

MCore 在这里验证并行和 DSv4 约束、计算派生维度、解析 pipeline layout。配置不合法时通常在这里就失败，不必等到 forward。

---

## 7. 第四步：`provider.provide_distributed_model(...)`

这是“配置对象”真正变成“GPU 分片模型”的一步。

### 7.1 `[verl]` 外层编排

调用位于：

```text
[verl] verl/utils/megatron_utils.py::make_megatron_module
```

verl 先准备：

- 可选 PEFT hook；
- value head/freeze router hook；
- DDP config；
- actor：`wrap_with_ddp=True`；
- ref/forward-only：`wrap_with_ddp=False`。

随后调用：

```python
provider.provide_distributed_model(
    wrap_with_ddp=...,
    ddp_config=...,
    fp16=provider.fp16,
    bf16=provider.bf16,
    use_megatron_fsdp=...,
)
```

### 7.2 `[Bridge]` 如何组织每个 rank 的模型

核心在：

```text
[Bridge] src/megatron/bridge/models/model_provider.py
         ModelProviderMixin.provide_distributed_model
         get_model
         _create_model
```

流程是：

1. 取得已有 `ProcessGroupCollection`；
2. 根据当前 PP rank / VPP chunk 算 `pre_process` 和 `post_process`；
3. 对每个本地 VPP chunk 调用 `provider.provide(...)`；
4. 执行 pre-wrap hooks；
5. 默认把 raw model 移到 `torch.cuda.current_device()`；
6. BF16 时包装 `Float16Module`；
7. actor 再包装 MCore DDP，ref 不包；
8. 始终返回 list，以支持 VPP 多 chunk。

所以典型返回值不是裸 `GPTModel`：

```text
actor:
list[
  DistributedDataParallel(
    Float16Module(
      GPTModel(...)
    )
  )
]

ref / forward_only:
list[
  Float16Module(
    GPTModel(...)
  )
]
```

具体 wrapper 会随 Megatron FSDP、layer-wise DDP 等配置变化。

### 7.3 它是不是 GPU 上的模型

默认 verl 路径下，答案是 **是**：

```python
model_module.cuda(torch.cuda.current_device())
```

发生在 Bridge `get_model()` 中，之后才做 BF16/DDP 包装。例外是显式使用：

- `use_cpu_initialization=True`；
- `init_model_with_meta_device=True`；
- 某些 FSDP2 延迟 materialize 路径。

而且它只是当前 rank 的局部模型：

- PP 决定本 rank 拥有哪些 transformer layers；
- TP 决定线性层权重如何切；
- EP 决定本 rank 拥有哪些 experts；
- VPP 决定 list 里有几个 chunk。

“完整模型”是所有相关 rank 上这些局部 module 的集合，不会在单卡上出现一份完整 DSv4。

### 7.4 `[Bridge → MCore]` `provider.provide()` 的交界线

`GPTModelProvider.provide()` 先执行：

```python
transformer_layer_spec = provider.transformer_layer_spec(provider, vp_stage=...)
```

DSv4 中这正是前面写入的 MCore `_get_exp_attn_spec`。它返回当前 PP/VPP chunk 的 `TransformerBlockSubmodules`，然后 Bridge 调用：

```python
MCoreGPTModel(
    provider,
    transformer_layer_spec=transformer_layer_spec,
    pre_process=...,
    post_process=...,
    pg_collection=...,
    mtp_block_spec=...,
)
```

**这一行以后，真正的网络结构由 MCore 创建。**

### 7.5 `[MCore]` ModuleSpec 如何变成真实模块

主要源码：

```text
megatron/core/models/gpt/gpt_model.py
megatron/core/transformer/transformer_block.py
megatron/core/transformer/spec_utils.py
megatron/core/models/gpt/experimental_attention_variant_module_specs.py
```

调用关系：

```text
GPTModel.__init__
  ├─ 根据 pre_process 创建 embedding
  ├─ TransformerBlock(config, submodules=block_spec)
  │    └─ 对本 rank 的每个 layer_spec 调 build_module(...)
  │         └─ 递归实例化 layer、attention、MLP、norm、linear
  ├─ 根据 layout 创建可选 MTP block
  └─ 根据 post_process 创建 output layer
```

Bridge 提供的是一张 `ModuleSpec` 配方，`spec_utils.build_module()` 才把配方里的 class 实例化为 `nn.Module`。

### 7.6 DSv4 module 大致长什么样

以下是一个“拥有 decoder layer 的 rank”上的近似结构；embedding、final norm、MTP、output layer 只在对应 PP stage 上存在：

```text
GPTModel                                      [MCore]
├─ embedding? LanguageModelEmbedding          [MCore]
├─ decoder: TransformerBlock                  [MCore]
│  ├─ layers: ModuleList
│  │  └─ HyperConnectionTransformerLayer      [MCore]
│  │     ├─ input_layernorm                    [TE RMSNorm]
│  │     ├─ self_attention:
│  │     │  DSv4HybridSelfAttention            [MCore]
│  │     │  ├─ linear_q_down_proj              [TE Linear, duplicated]
│  │     │  ├─ q_layernorm                     [TE RMSNorm]
│  │     │  ├─ linear_q_up_proj                [TE ColumnParallelLinear]
│  │     │  ├─ linear_kv_proj                  [TE ColumnParallelLinear]
│  │     │  ├─ kv_layernorm                    [TE RMSNorm]
│  │     │  ├─ core_attention:
│  │     │  │  CompressedSparseAttention       [MCore]
│  │     │  │  ├─ compressor? Compressor       [MCore + TE 子层]
│  │     │  │  └─ indexer? CSAIndexer          [MCore + TE 子层]
│  │     │  ├─ linear_o_group_proj             [MCore Parameter]
│  │     │  └─ linear_proj                     [TE RowParallelLinear]
│  │     ├─ self_attention_hyper_connection    [MCore]
│  │     ├─ pre_mlp_layernorm                  [TE RMSNorm]
│  │     ├─ mlp: MoELayer                      [MCore]
│  │     │  ├─ router                          [MCore TopK/hash router]
│  │     │  │  └─ tid2eid?                     [MCore persistent buffer]
│  │     │  ├─ token_dispatcher                [MCore all-to-all]
│  │     │  ├─ experts                         [TE GroupedMLP]
│  │     │  └─ shared_experts                  [MCore/TE]
│  │     └─ mlp_hyper_connection               [MCore]
│  └─ final_layernorm?                         [TE RMSNorm]
├─ mtp? MultiTokenPredictionBlock              [MCore]
└─ output_layer?                               [MCore/TE parallel linear]
```

MCore 的 experimental spec 当前强制：

```python
config.transformer_impl == "transformer_engine"
```

并使用 `TESpecProvider` 选线性层和 norm，所以只安装 Bridge 与 verl 不够；必须有相匹配的 MCore 和 Transformer Engine。

### 7.7 “算子”能静态看出多少

不用 forward，可以从 provider + ModuleSpec + `print(module)` 确认：

- Python module class；
- 当前 rank 有哪些层；
- linear 是 column/row/duplicated；
- 是否有 compressor/indexer；
- 是否使用 HyperConnection layer；
- MoE router、dispatcher、GroupedMLP 的实现类型；
- 参数 shape、dtype、device 和 requires_grad。

但不能仅靠打印最终确认：

- TE 内部最终选了哪个 GEMM kernel；
- sparse attention 是否走某个 fused DSA/FlashMLA kernel；
- kernel 是否因 H200、shape、dtype、sequence layout 而 fallback；
- forward 中真正触发的通信和 fused path。

`attention_backend=flash` 也不表示整个 `DSv4HybridSelfAttention` 被替换成一个普通 FlashAttention op。DSv4 的 CSA/compressor/indexer 仍是专用结构，运行时再根据配置和硬件进入具体 kernel。

---

## 8. 第五步：`bridge.load_hf_weights(module, model_path)`

module 已经在各 rank 上创建好，Bridge 现在把 checkpoint tensor 逐个变换并装进去。

### 8.1 外层入口

`AutoBridge.load_hf_weights()` 会：

1. 为 `model_path` 创建懒加载 `PreTrainedCausalLM`；
2. 再次根据架构取得 `DeepSeekV4Bridge`；
3. 调用 `DeepSeekV4Bridge.load_weights_hf_to_megatron(...)`。

核心源码：

```text
[Bridge] src/megatron/bridge/models/conversion/auto_bridge.py
[Bridge] src/megatron/bridge/models/conversion/model_bridge.py
[Bridge] src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py
[Bridge] src/megatron/bridge/models/conversion/param_mapping.py
[Bridge] src/megatron/bridge/models/conversion/quantization_utils.py
```

### 8.2 先把 wrapper 解开

传入的可能是：

```text
DDP(Float16Module(GPTModel))
```

Bridge 用 `unwrap_model()` 找到底层 `GPTModel`，但参数对象本身不重新创建。DDP bucket 已经建立也没关系，因为后面只是对既有参数做原地 `copy_`，形状和 parameter identity 不变。

### 8.3 为全模型构建 conversion tasks

`build_conversion_tasks()` 的工作不是马上加载 tensor，而是先生成确定性的任务列表：

1. 调用 DSv4 `mapping_registry()` 获得全部参数规则；
2. 收集所有 PP rank 的全局 Megatron 参数名并排序；
3. 遍历本 rank 每个 VPP chunk 的 `named_parameters()`；
4. 额外遍历 persistent buffers，因此 `tid2eid` 也能进入任务；
5. 把本地 layer number 翻译成全局 layer number；
6. 用 wildcard registry 查到相应的 HF key 和 mapping 类型；
7. 本 rank 不拥有的参数生成 `module=None` 占位 task，使各 PP rank 的全局任务索引一致；HF 导入时该 rank 直接跳过占位项，导出或需要 PP 通信时则可据此保持一致顺序。

这就是 PP 的关键：不是把所有层权重先放到每个 rank，而是全体 rank 对同一全局顺序达成一致，只有拥有目标层的 rank 才真正 `copy_`。

### 8.4 DSv4 mapping registry 映射哪些东西

它覆盖：

- embedding、LM head、final norm；
- mHC 全局和逐层参数；
- MLA 的 `wq_a/wq_b/wkv/wo_a/wo_b`；
- attention sink；
- compressor 和 indexer；
- router weight、bias 和 `tid2eid`；
- routed experts；
- shared experts；
- MTP 对应结构。

例如：

```text
HF layers.*.attn.wq_b.weight
  → Megatron decoder.layers.*.self_attention.linear_q_up_proj.weight

HF layers.*.ffn.experts.*.w1.weight  (gate)
HF layers.*.ffn.experts.*.w3.weight  (up)
  → Megatron mlp.experts.linear_fc1.weight*

HF layers.*.ffn.experts.*.w2.weight  (down)
  → Megatron mlp.experts.linear_fc2.weight*
```

### 8.5 每个 task 的执行顺序

`load_weights_hf_to_megatron()` 对每个目标参数做：

```text
懒读取 HF tensor
  → DSv4 maybe_modify_loaded_hf_weight()
  → mapping.hf_to_megatron()
  → 检查转换结果和目标参数 shape
  → target_parameter.copy_(converted_shard)
```

### 8.6 FP8/MXFP4 是如何导入的

DSv4 override：

```python
def maybe_modify_loaded_hf_weight(...):
    return maybe_dequantize_hf_quantized_weight(...)
```

Bridge 根据 checkpoint tensor dtype 自动判断：

| checkpoint weight | sibling scale | 导入动作 |
|---|---|---|
| `torch.float8_e4m3fn` | `*.scale` | 扩展 block scale，反量化为 BF16 |
| packed `torch.int8`（每字节两个 E2M1 nibble） | `*.scale` | 解包 MXFP4，扩展 E8M0 scale，反量化为 BF16 |
| 普通 dtype | 可无 | 按需直接使用/转换 |

因此“支持 FP8/MXFP4 weight transfer”在这一导入方向上的准确含义是：**能读取量化存储格式并在复制进训练模型之前恢复为 BF16**。它不表示 actor 的 MCore 参数会以 MXFP4 形式参与训练；训练侧是否启用 FP8/量化参数是另一套 provider/TE 配置。

### 8.7 mapping 如何处理 TP/EP

`AutoMapping` 会查看目标 MCore module class 的注册类型：

- column-parallel：沿输出维切分/scatter 到 TP rank；
- row-parallel：沿输入维切分/scatter 到 TP rank；
- replicated：向并行 rank 广播完整 tensor。

`GatedMLPMapping` 不是简单 concat 全 tensor。它会分别处理 HF 的 gate `w1` 与 up `w3`，按目标 FC1 布局切分后组合，保证每个 rank 收到正确局部块。

Expert 参数还结合：

- EP：当前 rank 只保留其负责的 expert；
- ETP：expert 内部再做 tensor parallel；
- expert 名字中的 global/local expert index 转换。

最终 mapping 产出的 tensor 会移到目标参数所在 device/dtype，然后在 `torch.no_grad()` 语义下原地复制。

### 8.8 权重加载后的状态

这一步结束后：

```text
module 是真实 GPU module
+ 参数已经按 PP/TP/EP/ETP 分片
+ checkpoint FP8/MXFP4 已按导入规则反量化
+ 本地目标参数已写入 BF16 权重
+ actor 可有 DDP wrapper，ref 没有
```

后续如果 verl 开启 `param_offload/all_offload`，它可能再把参数移到 CPU；那是 engine 初始化后续的内存管理，不改变 `provide_distributed_model()` 默认先在 GPU 上创建模型这一事实。

---

## 9. verl 到底 patch 了哪些，哪些没有 patch

### 9.1 这五步中的 verl 自有代码

| 行为 | 归属 | 性质 |
|---|---|---|
| 初始化 TP/PP/EP/ETP/CP groups | verl 调 MCore | 编排 |
| 组装 provider overrides | verl | 配置覆盖 |
| MTP 未启用时将 `mtp_num_layers=0` 并裁 ratio | verl | DSv4 配置适配 |
| 创建 DDP config、决定 actor/ref 是否包 DDP | verl | 编排 |
| 显式在建模后调用 `load_hf_weights` | verl 调 Bridge | 调用顺序 |
| `verl/models/mcore/bridge.py` 的旧版兼容 fallback | verl | 兼容实现 |

### 9.2 这五步不是 verl 实现的部分

| 行为 | 真正实现方 |
|---|---|
| `AutoBridge.from_hf_pretrained` | Bridge |
| DSv4 HF config → `MLAModelProvider` | Bridge |
| DSv4 `mapping_registry`、FP8/MXFP4 反量化 | Bridge |
| provider finalize 的最终约束 | MCore（Bridge 延迟调用） |
| `GPTModel`、TransformerBlock、DSv4HybridSelfAttention | MCore |
| CSA/compressor/indexer、mHC、MoE router/dispatcher | MCore |
| TE linear/RMSNorm/GroupedMLP | Transformer Engine，经 MCore ModuleSpec 选择 |

### 9.3 建模/加载完成后才发生的 verl patch

v0.9.0 `_maybe_enable_fused_kernels()` 中：

```python
from verl.models.mcore.model_forward_fused import patch_fused_forward
for model in self.module:
    patch_fused_forward(model)
```

这是 verl 的真实 monkey patch，但它发生在 module 创建和权重导入之后，主要改 forward 路径，不负责定义 DSv4 的基本层结构，也不负责 HF 权重映射。

另外 routing replay 开启时，verl/配套代码会为 MoE router 加重放相关行为；这同样是可选训练语义，不是 `DeepSeekV4Bridge.provider_bridge()` 的基本建模职责。

CP contiguous layout 与后续 CP fixes 主要作用于输入布局、scheduler 和 attention forward，不应混进“provider 如何创建 module”这条主链中。它们会通过 override 改配置，但具体效果在 MCore forward/通信路径体现。

---

## 10. 不执行 forward，能看到什么

在 8 卡 H200 上可以只初始化 process groups、provider 和 module，然后打印，不调用 forward。建议分三层看，而不是只打印最外层 wrapper。

### 10.1 先打印 provider

```python
print(type(provider))
for key in (
    "experimental_attention_variant",
    "transformer_layer_spec",
    "num_layers",
    "csa_compress_ratios",
    "enable_hyper_connections",
    "num_residual_streams",
    "num_moe_experts",
    "moe_n_hash_layers",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
):
    print(key, getattr(provider, key, None))
```

这一步确认“配置将创建什么”，不分配模型参数。

### 10.2 打印 ModuleSpec

```python
spec = provider.transformer_layer_spec(provider, vp_stage=None)
print(spec)
```

在已经初始化 PP group 的 rank 上，它会显示当前 PP stage 对应的 layer specs，可以静态确认 MCore/TE class 选择。

### 10.3 解 wrapper 后打印真实模型

```python
from megatron.core.utils import unwrap_model

raw_chunks = unwrap_model(module)
for chunk_id, raw in enumerate(raw_chunks):
    print(f"rank={torch.distributed.get_rank()} chunk={chunk_id}")
    print(raw)
    for name, submodule in raw.named_modules():
        print(name, type(submodule).__module__, type(submodule).__name__)
```

还可以只看参数元数据，避免打印 tensor 内容：

```python
for name, param in raw.named_parameters():
    print(name, tuple(param.shape), param.dtype, param.device, param.requires_grad)
```

注意：PP=8 时每张卡只会打印自己的局部 stage；把 8 个 rank 的日志按 rank 收集后才是完整结构。`print(module)` 不会触发 forward，但完整 DSv4 仍会真实分配大量参数显存。若只想看结构，可研究 Bridge 的 meta-device 路径；不过 DSv4/TE 的所有模块是否都能稳定 meta 初始化，应以精确依赖版本实测。

---

## 11. 推荐的源码阅读顺序

按下面顺序读，调用边界最清楚：

1. `[verl]` `verl/workers/engine/megatron/transformer_impl.py`
   - `_build_tf_config()`
   - `_build_megatron_module()`
2. `[verl]` `verl/utils/megatron_utils.py::make_megatron_module()`
3. `[Bridge]` `models/conversion/auto_bridge.py`
   - `from_hf_pretrained()`
   - `to_megatron_provider()`
   - `load_hf_weights()`
4. `[Bridge]` `models/deepseek/deepseek_v4_bridge.py`
   - 注册 decorator
   - `provider_bridge()`
   - `mapping_registry()`
   - `maybe_modify_loaded_hf_weight()`
5. `[Bridge]` `models/model_provider.py`
   - `apply_overrides_and_finalize()`
   - `provide_distributed_model()` / `get_model()` / `_create_model()`
6. `[Bridge]` `models/gpt_provider.py::provide()`
7. `[MCore]` `models/gpt/experimental_attention_variant_module_specs.py`
8. `[MCore]` `models/gpt/gpt_model.py`、`transformer_block.py`、`spec_utils.py`
9. `[MCore]` `transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py`
10. `[MCore]` `transformer/experimental_attention_variant/csa.py`
11. `[MCore]` `transformer/hyper_connection.py`、`transformer_layer.py`
12. `[Bridge]` 回到 `conversion/model_bridge.py`、`param_mapping.py`、`quantization_utils.py` 看权重导入。

## 12. 最终压缩成五句话

1. `from_hf_pretrained` 创建的是懒 HF config/checkpoint 访问器，并据 `DeepseekV4ForCausalLM` 选中 `DeepSeekV4Bridge`，不是 HF GPU 模型。
2. `to_megatron_provider(False)` 把 HF 架构翻译成 `MLAModelProvider`，并注入 DSv4 的 ModuleSpec、CSA、mHC、MoE 和 MTP 语义；provider 仍不是模型。
3. `apply_overrides_and_finalize` 让 verl 覆盖本次并行/精度/训练策略，再让 MCore 完成校验和派生配置。
4. `provide_distributed_model` 由 Bridge 编排，由 MCore 根据 ModuleSpec 创建本 rank 的真正 `GPTModel`，默认搬到 GPU，并按 actor/ref 需要做 BF16/DDP 包装。
5. `load_hf_weights` 由 Bridge 懒读取 checkpoint，把 FP8/MXFP4 反量化为 BF16，按映射规则改布局并沿 TP/EP/PP 分发，最后原地写入 MCore 参数。

这也解释了为什么研究 DSv4 初始化不能只看 Bridge：**Bridge 决定“怎么翻译、怎么组装、怎么搬权重”，MCore 决定“最终造出哪些模块以及 forward 怎么执行”。**
