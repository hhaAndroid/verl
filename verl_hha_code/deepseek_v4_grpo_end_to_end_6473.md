# DeepSeek-V4 GRPO 端到端流程：PR #6473 源码导读

> 研究范围：verl v0.9.0 中与 DeepSeek-V4-Flash GRPO 端到端训练有关的第一组改动，重点是 PR [#6473](https://github.com/verl-project/verl/pull/6473)。
>
> 核对基线：PR #6473 的 squash commit `cf8005fdadabe99b3620a4b86a960a914d948b37`；v0.9.0 tag `483b8a009ba3a97563edee3a19887e4862b8094a`。
>
> 本文不是启动脚本的逐行翻译，而是回答：一个 prompt 如何经过 vLLM 采样、奖励和 GRPO、Megatron actor 更新，再把最新的 FP8/MXFP4 权重安全地送回 vLLM，形成下一轮 on-policy 闭环。
>
> 可以初步参考 [https://verl.readthedocs.io/en/latest/advance/deepseek_v4_integration.html](https://verl.readthedocs.io/en/latest/advance/deepseek_v4_integration.html)

## 1. 先给结论

PR #6473 真正打通的是下面这个循环：

```text
DeepSeek-V4-Flash checkpoint
        │
        ├─ Megatron-Bridge 导入、反量化 ──> BF16 Megatron actor（可选 ref）
        │                                      │
        └─ vLLM 原生量化加载 ──────────────> FP8/MXFP4 rollout
                                               │
prompt ──> vLLM 生成 n 个 response + logprob + MoE routes (R3)
                                               │
         reward ──> GRPO group advantage ──> Megatron 重算 old logprob
                                               │
                         Megatron actor 做 PPO/GRPO 更新
                                               │
      Megatron-Bridge 导出 HF/checkpoint 格式的量化权重与 scale
                                               │
       bucketed CUDA IPC ──> vLLM prepare/load/finalize 热更新
                                               │
                              下一轮使用新策略继续 rollout
```

这里最难的不是 GRPO 公式，而是两个跨实现边界：

1. **训练/推理权重边界**：Megatron 内部是训练布局，vLLM 内部是推理 kernel 布局；DeepSeek-V4 又混用了 FP8 与 MXFP4，不能简单按名字 `copy_`。
2. **MoE 路由边界**：vLLM rollout 与 Megatron 重算 log probability 时可能选到不同 expert。R3 把 rollout 实际走过的路由带回训练端重放，降低 rollout/training mismatch。

Megatron-Bridge 同时承担三种职责：

- 把 Hugging Face DeepSeek-V4 配置转换成 Megatron provider/config；
- 把 checkpoint 权重导入并分发到 TP/PP/EP ranks；
- 把训练后的 Megatron shards 汇集、改名并导出成 rollout 或持久化 checkpoint 可消费的格式。



## 2. verl 如何通过 Megatron-Bridge 初始化 DeepSeek-V4

这一章只讨论初始化，不讨论 rollout、GRPO、router replay 的运行过程和 actor→vLLM 权重热更新。

### 2.1 先标清代码归属

后面的流程使用以下标签：


| 标签                     | 代码归属                                     | 主要职责                                                                          |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `[verl]`               | verl 仓库                                  | worker 编排、读取训练配置、选择 engine、覆盖并行参数、调用第三方库、增加兼容 patch                           |
| `[Transformers/HF]`    | Hugging Face Transformers 和模型 checkpoint | 提供 `config.json`、tokenizer、HF 参数命名和 checkpoint tensors                        |
| `[Megatron-Bridge]`    | NVIDIA Megatron-Bridge                   | 自动识别模型 Bridge、HF→Megatron 配置翻译、参数映射、量化权重反量化和 distributed weight loading       |
| `[Megatron-Core]`      | NVIDIA Megatron-LM/Megatron-Core         | 真正的分布式 `GPTModel`、TP/PP/EP/CP 通信组、Transformer layers、MoE、DDP 和 optimizer 基础设施 |
| `[Transformer Engine]` | NVIDIA Transformer Engine                | FlashAttention、部分 fused kernel、FP8/THD 等底层算子                                  |
| `[vLLM]`               | vLLM                                     | 初始化 actor 时只在 HF config fallback 中被借用；它不是 Megatron actor 的建模后端                |


最容易混淆的一点是：

> verl 没有自己实现一套 DeepSeek-V4 Megatron 模型。DeepSeek-V4 的模型语义和 HF↔Megatron 转换主要由 Megatron-Bridge 实现；真正执行训练的模型类来自 Megatron-Core；verl 把它们接入自己的 worker/trainer，并在不兼容处做 override 或 patch。

相关源码位置：


| 归属                  | 文件/入口                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `[verl]`            | `verl/workers/engine_workers.py::ActorRolloutRefWorker.init_model`                       |
| `[verl]`            | `verl/workers/engine/megatron/transformer_impl.py::MegatronEngine`                       |
| `[verl]`            | `verl/utils/megatron_utils.py::make_megatron_module`                                     |
| `[verl]`            | `verl/models/mcore/bridge.py`，Megatron-Bridge 的薄封装与版本兼容 fallback                         |
| `[Megatron-Bridge]` | `megatron/bridge/models/conversion/auto_bridge.py::AutoBridge`                           |
| `[Megatron-Bridge]` | `megatron/bridge/models/deepseek/deepseek_v4_bridge.py::DeepSeekV4Bridge`                |
| `[Megatron-Bridge]` | `megatron/bridge/models/conversion/model_bridge.py::load_weights_hf_to_megatron`         |
| `[Megatron-Core]`   | `megatron.core.models.gpt.GPTModel` 以及 DeepSeek-V4 experimental attention/MoE layer spec |




### 2.2 最短调用链

整个初始化可以先记成下面几行：

```python
# [verl] 选择并创建 Megatron engine
engine = EngineRegistry.new(model_type="language_model", backend="megatron", ...)

# [verl] 调用；[Megatron-Bridge] 实现 AutoBridge
bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)

# [Megatron-Bridge] 把 HF config 翻译成 Megatron provider
provider = bridge.to_megatron_provider(load_weights=False)

# [verl] 写入本次训练的 TP/PP/EP/CP、BF16、FlashAttention 等配置
provider.apply_overrides_and_finalize(dtype=torch.bfloat16, overrides=provider_overrides)

# [verl] 编排调用；[Megatron-Bridge/Megatron-Core] 创建真正的 distributed GPTModel
module = provider.provide_distributed_model(...) # gpu 分片模型

# [verl] 发起调用；[Megatron-Bridge] 映射、反量化、切分并复制 HF 权重
bridge.load_hf_weights(module, model_path)

# [verl] 编排；底层使用 [Megatron-Core] optimizer
optimizer = build_optimizer(module)
```

对象关系如下：


| 对象                 | 归属                  | 是不是模型 | 作用                                               |
| ------------------ | ------------------- | ----- | ------------------------------------------------ |
| `HFModelConfig`    | `[verl]`            | 否     | 统一保存 local path、tokenizer、HF config 和 verl 的模型选项 |
| HF `config.json`   | `[Transformers/HF]` | 否     | 描述 DeepSeek-V4 的原始结构                             |
| `AutoBridge`       | `[Megatron-Bridge]` | 否     | 自动选择适合当前 HF architecture 的具体 Bridge              |
| `DeepSeekV4Bridge` | `[Megatron-Bridge]` | 否     | DSv4 专属配置翻译与 weight mapping                      |
| `MLAModelProvider` | `[Megatron-Bridge]` | 否     | Megatron 模型工厂，同时承载 TransformerConfig             |
| `GPTModel` chunks  | `[Megatron-Core]`   | 是     | 每个 PP/VP rank 真正参与训练的模型                          |




### 2.3 第一步：verl 准备 HFModelConfig

入口是：

```text
[verl] ActorRolloutRefWorker.init_model()
  └─ omega_conf_to_dataclass(config.model)
       └─ [verl] HFModelConfig.__post_init__()
```

`HFModelConfig` 做的事情包括：

1. `[verl]` 用 `copy_to_local()` 把模型或 HDFS 路径准备成本机可访问的 `local_path`；
2. `[verl]` 调用 tokenizer/processor helper；底层读取的是 `[Transformers/HF]` 资产；
3. `[verl]` 调用 `AutoConfig.from_pretrained()` 读取 `[Transformers/HF] config.json`；
4. 保存 `hf_config.model_type`、`architectures`、token IDs、MTP 配置等信息。

正常情况下这里得到：

```text
hf_config.model_type   = "deepseek_v4"
hf_config.architectures = ["DeepseekV4ForCausalLM"]
```

这里已经有第一个 verl 兼容处理：如果当前 Transformers 恰好因为没有注册 `deepseek_v4` 而抛出对应的 `KeyError`，verl 才退回：

```python
# [verl compatibility fallback]
from vllm.transformers_utils.config import get_config
hf_config = get_config(...)
```

这是 **verl 的 config-loader fallback**，不是 Megatron-Bridge 的能力，也不是让 actor 使用 vLLM 模型。verl 只借 vLLM 的 config registry 把配置对象读出来；后面 actor 仍然是 Megatron-Core。

这个 fallback 是 fail-closed 的：只有精确匹配 `KeyError("deepseek_v4")` 才触发，其他 `ValueError` 不会被吞掉。

### 2.4 第二步：verl 选择 Megatron engine

`ActorRolloutRefWorker` 为 actor 构造：

```python
# [verl]
actor_training_config = TrainingWorkerConfig(
    model_type="language_model",
    model_config=model_config,
    engine_config=actor_config.engine,
    optimizer_config=actor_config.optim,
)

actor = TrainingWorker(actor_training_config)
actor.reset()
```

`TrainingWorker` 再调用：

```python
# [verl]
EngineRegistry.new(
    model_type="language_model",
    backend="megatron",
)
```

注册表选择 `[verl] MegatronEngineWithLMHead`。这个类是 verl 对 engine 生命周期的封装，但内部模型仍是 `[Megatron-Core] GPTModel`。

`reset()` 最终进入：

```text
[verl] MegatronEngine.initialize()
  ├─ _build_tf_config() # transformers
  ├─ _build_megatron_module()
  ├─ _maybe_enable_fused_kernels()
  ├─ _build_optimizer()
  └─ _build_lr_scheduler()
```



### 2.5 第三步：Megatron-Core 初始化模型并行通信组

在创建模型前，`[verl] MegatronEngine._init_device_mesh()` 调用：

```python
# API 属于 [Megatron-Core]，参数来自 [verl] 配置
mpu.initialize_model_parallel(
    tensor_model_parallel_size=ACTOR_TP,
    pipeline_model_parallel_size=ACTOR_PP,
    expert_model_parallel_size=ACTOR_EP,
    expert_tensor_parallel_size=ACTOR_ETP,
    context_parallel_size=ACTOR_CP,
)
```

职责要拆开看：

- TP/PP/EP/CP 数值由 `[verl]` Hydra 配置和示例脚本提供；
- process groups、rank 计算和 collective groups 由 `[Megatron-Core]` 实现；
- 后面 Bridge 根据这些 group，决定每个 rank 应该得到哪个参数 shard。



### 2.6 第四步：AutoBridge 自动找到 DeepSeekV4Bridge

verl 在 `_build_tf_config()` 中写的是：

```python
# [verl] 导入一个薄封装
from verl.models.mcore.bridge import AutoBridge

# 真正的 AutoBridge 类来自 [Megatron-Bridge]
bridge = AutoBridge.from_hf_pretrained(
    model_config.local_path,
    trust_remote_code=model_config.trust_remote_code,
)
```

`verl/models/mcore/bridge.py` 本身没有重新实现 AutoBridge；主要逻辑就是：

```python
# [verl thin wrapper]
from megatron.bridge import AutoBridge
```

并针对不同 Megatron-Bridge 版本补少量 `make_value_model`、`freeze_moe_router` fallback。对普通 DeepSeek-V4 actor，这些 fallback 不是模型主体。

`[Megatron-Bridge] AutoBridge.from_hf_pretrained()` 会：

1. 读取 HF config；
2. 验证它是不是支持的 CausalLM architecture；
3. 建立一个 checkpoint state source；
4. 根据 `architectures`/`model_type` 从 Bridge registry 选择具体 Bridge。

Megatron-Bridge 中 DeepSeek-V4 的注册是：

```python
# [Megatron-Bridge]
@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(MegatronModelBridge):
    ...
```

因此分派结果是：

```text
[Transformers/HF]
architectures=DeepseekV4ForCausalLM
model_type=deepseek_v4
          │
          ▼
[Megatron-Bridge]
DeepSeekV4Bridge
          │
          ├─ target:   [Megatron-Core] GPTModel
          └─ provider: [Megatron-Bridge] MLAModelProvider
```

如果安装的 Megatron-Bridge 版本没有这个注册类，verl 无法只靠自身代码补出完整 DeepSeek-V4 模型，因此 #6473 明确依赖支持 DSv4 的 Bridge commit。

### 2.7 第五步：DeepSeekV4Bridge 把 HF config 翻译成 provider

verl 调用：

```python
# 调用点属于 [verl]；实现属于 [Megatron-Bridge]
provider = bridge.to_megatron_provider(load_weights=False)
```

`provider` 不是模型，而是一份可创建模型的配置化工厂。`load_weights=False` 的意义是：先只翻译结构，稍后等 verl 完成并行/DDP/PEFT 编排后，再显式加载权重。

`[Megatron-Bridge] DeepSeekV4Bridge.provider_bridge()` 先继承通用 HF→GPT 字段，再写入 DSv4 专属语义：


| HF DeepSeek-V4 字段                         | Megatron provider 字段          | 归属                     |
| ----------------------------------------- | ----------------------------- | ---------------------- |
| `head_dim`                                | `v_head_dim`                  | `[Megatron-Bridge]` 翻译 |
| `qk_rope_head_dim`                        | `qk_pos_emb_head_dim`         | `[Megatron-Bridge]` 翻译 |
| `q_lora_rank`                             | `q_lora_rank`                 | `[Megatron-Bridge]` 翻译 |
| `o_lora_rank`                             | `o_lora_rank`                 | `[Megatron-Bridge]` 翻译 |
| `sliding_window`                          | `csa_window_size`             | `[Megatron-Bridge]` 翻译 |
| `index_n_heads/index_head_dim/index_topk` | DSA indexer geometry          | `[Megatron-Bridge]` 翻译 |
| `hc_mult/hc_sinkhorn_iters`               | mHC residual streams/Sinkhorn | `[Megatron-Bridge]` 翻译 |
| `num_experts_per_tok`                     | `moe_router_topk`             | `[Megatron-Bridge]` 翻译 |
| `num_nextn_predict_layers`                | `mtp_num_layers`              | `[Megatron-Bridge]` 翻译 |


还会设置：

```python
# [Megatron-Bridge]
provider.experimental_attention_variant = "dsv4_hybrid"
provider.multi_latent_attention = True
provider.transformer_layer_spec = dsv4_experimental_layer_spec

provider.enable_hyper_connections = True
provider.moe_grouped_gemm = True
provider.moe_n_hash_layers = 3
provider.moe_router_topk = 6
provider.activation_func_clamp_value = 10.0
```

而这些 layer spec 最终创建的 attention、MoE、router 和 tensor-parallel modules 来自 `[Megatron-Core]`。

所以这一层的边界是：

```text
[Megatron-Bridge] 决定“DeepSeek-V4 应怎样配置 Megatron”
[Megatron-Core]   实现“配置指定的分布式层怎样计算”
```



### 2.8 第六步：verl 覆盖本次训练的并行和运行参数

Bridge 从 HF config 知道模型结构，但不知道本次 verl job 想用多少 TP/PP/EP/CP，也不知道 verl 的 packed sequence、pipeline 通信和 MoE dispatcher 策略。模型/optimizer offload 不在这里设置，而是在模型和 optimizer 创建完成后由 `[verl]` 单独执行。

#### 2.8.1 verl 固定放入 `provider_overrides` 的字段

`[verl] MegatronEngine._build_tf_config()` 首先构造：

```python
# [verl integration overrides]
virtual_pp = self.engine_config.virtual_pipeline_model_parallel_size

provider_overrides = {
    # 本次 job 的并行拓扑
    "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
    "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
    "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
    "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
    "virtual_pipeline_model_parallel_size": virtual_pp,
    "context_parallel_size": self.engine_config.context_parallel_size,
    "sequence_parallel": self.engine_config.sequence_parallel,

    # pipeline 通信方式
    "overlap_p2p_comm": virtual_pp is not None and virtual_pp > 1,
    "batch_p2p_comm": False,

    # verl 的变长/packed sequence 路径
    "variable_seq_lengths": True,

    # MCore attention backend
    "attention_backend": AttnBackend.flash,

    # MoE token dispatch 和负载均衡策略
    "moe_token_dispatcher_type": "alltoall",
    "moe_router_load_balancing_type": "none",
}
```

这些是 `[verl]` 显式选择的值，而不是 Bridge 从 DeepSeek-V4 HF config 翻译出来的架构参数。特别是 `attention_backend=flash` 是传给 `[Megatron-Core]` 的 backend 枚举；它不表示 Python module tree 中一定存在一个名字就叫 `FlashAttention` 的模块。真正调用哪个 Transformer Engine/CUDA kernel 还要在构建和 forward 时决定。

#### 2.8.2 配置文件还可以在最后覆盖它们

`[verl]` 随后把 `engine_config.override_transformer_config` 合并进同一个字典：

```python
# [verl]
for key, value in override_transformer_config.items():
    provider_overrides[key] = value
```

它是后写入的，所以优先级高于上面的固定值。v0.9.0 默认 YAML 中包含：

```yaml
# [verl config]
override_transformer_config:
  recompute_granularity: null
  recompute_modules: ["core_attn"]
  recompute_method: null
  recompute_num_layers: null
  attention_backend: flash
```

`mapping_string_to_attn_backend()` 会先把字符串 `flash` 转成 MCore 的 `AttnBackend.flash`。用户还可以通过这个字典覆盖其他 provider 字段；但新版 `[Megatron-Bridge]` 会用 `hasattr(provider, name)` 检查字段，写入不存在的字段会直接抛 `AttributeError`。

如果启用 `[verl] dynamic_context_parallel`，这个待合并字典还会加入：

```python
# [verl Dynamic-CP integration overrides]
{
    "calculate_per_token_loss": True,
    "context_parallel_size": self.engine_config.context_parallel_size,
    "dynamic_context_parallel": False,
    "max_seqlen_per_dp_cp_rank": self.engine_config.max_seqlen_per_dp_cp_rank,
}
```

这里的 `dynamic_context_parallel=False` 不是关闭 verl Dynamic CP。verl 会在进入 MCore pipeline schedule 之前完成 DP×CP 动态调度，因此这里关闭的是 **MCore 内部再次调度 Dynamic CP**，避免两边重复调度。DeepSeek-V4 在 CP/Dynamic-CP 下追加的两个专属字段见下一节。

#### 2.8.3 `dtype` 不是字典项，但也会覆盖 provider

最终调用 `[Megatron-Bridge]` API：

```python
provider.apply_overrides_and_finalize(
    dtype=self.param_dtype,  # DSv4 示例通常是 torch.bfloat16
    overrides=provider_overrides,
)
```

当 `dtype=torch.bfloat16` 时，`[Megatron-Bridge]` 会先隐式写入三个字段：

```python
# [Megatron-Bridge]
provider.params_dtype = torch.bfloat16
provider.fp16 = False
provider.bf16 = True
```

随后 Bridge 才逐项执行 `setattr(provider, name, value)`，最后调用 `provider.finalize()`。所以完整顺序是：

```text
[Megatron-Bridge] 设置 params_dtype / fp16 / bf16
        ↓
[Megatron-Bridge] 应用由 verl 组装的 provider_overrides
        ↓
[Megatron-Bridge] DSv4 且 PP>1 时，按需自动生成 pipeline layout
        ↓
[Megatron-Bridge/Megatron-Core] provider.finalize()：派生配置并校验约束
```

这里必须区分：

- override 字典的选择和数值属于 `[verl]`；
- `dtype` 参数来自 `[verl]`，但把它展开成 `params_dtype/fp16/bf16` 的实现属于 `[Megatron-Bridge]`；
- `apply_overrides_and_finalize()` 的校验和派生逻辑属于 `[Megatron-Bridge]`；
- provider 内部的 TransformerConfig 约束主要来自 `[Megatron-Core]`。

因此，`finalize()` 之后看到的 provider 字段不一定全是 verl 显式覆盖的。例如 DSv4 的自动 `pipeline_model_parallel_layout` 和根据 hybrid pattern 推导出的字段，属于 `[Megatron-Bridge/Megatron-Core]` 的派生结果。

对于旧 Megatron-Bridge 没有 `apply_overrides_and_finalize()` 的情况，verl 会 fallback 到逐字段 `setattr()` 再 `provider.finalize()`。这是 `[verl]` 的第三方版本兼容 shim，不是 DeepSeek-V4 算法逻辑。

### 2.9 第七步：verl 对 MTP 和 CP 做 DSv4 配置修正

这是配置修改，不是 monkey patch。

#### MTP 修正

如果 verl 配置关闭 MTP：

```python
# [verl DSv4 config fix]
provider_overrides["mtp_num_layers"] = 0
provider_overrides["csa_compress_ratios"] = csa_compress_ratios[:provider.num_layers]
```

原因是 `[Megatron-Bridge]` 会如实从 HF config 读出 MTP 层；但 GRPO actor/ref 可能明确不要 MTP。只关闭 loss 而不改变 provider 层数，会让主层数、CSA metadata 和 weight mappings 不一致。

`HFModelConfig` 目前也会在 `model_config.mtp.enable=False` 时，把 `hf_config.num_nextn_predict_layers` 等字段归零。这仍属于 `[verl]` 的统一配置规范化。

#### CP 修正

当 DSv4 使用 CP>1，或者启用 verl Dynamic CP 时，verl 会把下面两个字段也加入 `provider_overrides`：

```python
# [verl DSv4 CP config fix，主要由后续 #7221/#7297 补齐]
provider_overrides["cp_partition_mode"] = "contiguous"
provider_overrides["sequence_packing_scheduler"] = "default_dynamic_cp"
```

这是 verl 告诉 Megatron-Core 应采用哪种布局；真正的 CP partition/scheduler 执行仍在 Megatron-Core/Transformer Engine。

### 2.10 第八步：创建真正的 distributed GPTModel

`[verl] MegatronEngine._build_megatron_module()` 调用 `[verl] make_megatron_module()`，后者最终执行：

```python
# 调用编排属于 [verl]
# provider/provide_distributed_model 属于 [Megatron-Bridge]
# 创建出的 GPTModel、DDP 和 layers 属于 [Megatron-Core]
model = provider.provide_distributed_model(
    wrap_with_ddp=True,
    ddp_config=ddp_config,
    fp16=False,
    bf16=True,
)
```

这一步才真正分配 Megatron parameters：

- `[Megatron-Core]` PP rank 只创建自己负责的 decoder layers；
- `[Megatron-Core]` TP rank 创建相应 tensor shard；
- `[Megatron-Core]` EP rank 创建自己负责的 experts；
- `[Megatron-Core]` 第一 PP stage 创建 embedding；
- `[Megatron-Core]` 最后一 PP stage 创建 output layer；
- `[Megatron-Core]` 根据 verl 传入的 DDP config 包装模型。

verl 在模型创建前还可以向 provider 注册 PEFT/value-head/freeze-router hook；这些 hook 的注册时机由 `[verl]` 控制，provider 的 pre-wrap hook 机制来自 `[Megatron-Bridge]`。

到这里为止只有正确的分布式结构，HF checkpoint 权重还没有被显式装入。

### 2.11 第九步：Megatron-Bridge 加载、反量化并切分 HF 权重

verl 显式调用：

```python
# 调用点属于 [verl]
# load_hf_weights 实现属于 [Megatron-Bridge]
self.bridge.load_hf_weights(module, model_config.local_path)
```

`[Megatron-Bridge] DeepSeekV4Bridge.mapping_registry()` 定义名称和结构映射。例如：


| HF checkpoint               | Megatron-Core parameter            | 转换实现                                |
| --------------------------- | ---------------------------------- | ----------------------------------- |
| `embed.weight`              | `embedding.word_embeddings.weight` | `[Megatron-Bridge] AutoMapping`     |
| `head.weight`               | `output_layer.weight`              | `[Megatron-Bridge] AutoMapping`     |
| `layers.*.attn.wq_a.weight` | `linear_q_down_proj.weight`        | `[Megatron-Bridge] AutoMapping`     |
| `layers.*.attn.wq_b.weight` | `linear_q_up_proj.weight`          | `[Megatron-Bridge] AutoMapping`     |
| `layers.*.attn.wkv.weight`  | `linear_kv_proj.weight`            | `[Megatron-Bridge] AutoMapping`     |
| `layers.*.ffn.gate.weight`  | `mlp.router.weight`                | `[Megatron-Bridge] AutoMapping`     |
| `layers.*.ffn.gate.tid2eid` | `mlp.router.tid2eid`               | `[Megatron-Bridge] AutoMapping`     |
| expert `w1 + w3`            | `experts.linear_fc1`               | `[Megatron-Bridge] GatedMLPMapping` |
| expert `w2`                 | `experts.linear_fc2`               | `[Megatron-Bridge] AutoMapping`     |


DeepSeek-V4-Flash checkpoint 含有量化权重：

```text
dense/attention 等：E4M3 FP8 weight + block scale
routed experts：     packed MXFP4 weight + E8M0 scale
```

导入时的反量化入口是：

```python
# [Megatron-Bridge DeepSeekV4Bridge]
def maybe_modify_loaded_hf_weight(hf_param, hf_state_dict):
    return quantization_utils.maybe_dequantize_hf_quantized_weight(
        hf_param, hf_state_dict
    )
```

这段反量化 **不是 verl patch**，而是 Megatron-Bridge 的 DSv4 原生转换逻辑。它把 checkpoint weight 与对应 scale 组合，恢复为训练使用的 BF16 tensor。

之后 `[Megatron-Bridge] load_weights_hf_to_megatron()` 对每个 conversion task 执行：

```text
读取 [HF] source tensor/scale
  → [Megatron-Bridge] DSv4 反量化
  → [Megatron-Bridge] 参数合并/转置/重排
  → [Megatron-Bridge + Megatron-Core groups] TP/EP shard 分发
  → copy 到当前 rank 的 [Megatron-Core] parameter
```

PP rank 不拥有某层时，对应 task 的 `megatron_module` 为空，会跳过；TP/EP rank 只收到自己的 local shard。

所以 actor 最终不是在 packed MXFP4 parameter 上直接做 optimizer update，而是获得按 Megatron TP/PP/EP 布局分布的 BF16 可训练参数。

### 2.12 第十步：verl 安装训练侧 patch，然后创建 optimizer

权重加载完成后，verl 还会根据配置安装若干训练 patch。必须区分哪些是 DSv4 必需、哪些是可选功能、哪些只是版本兼容。


| verl 行为                                               | 类型                          | 何时启用                                      | 是否 DSv4 基础初始化必需         |
| ----------------------------------------------------- | --------------------------- | ----------------------------------------- | ----------------------- |
| Transformers 不识别 `deepseek_v4` 时改用 vLLM config loader | config fallback             | 精确的 missing model type                    | 取决于 Transformers 版本     |
| 将 MTP layer count 归零并裁剪 CSA metadata                  | config override             | `mtp.enable=False`                        | 是，关闭 MTP 时需要            |
| 强制 DSv4 CP 使用 contiguous layout                       | config override             | CP>1                                      | CP=1 不需要                |
| `patch_fused_forward()`                               | native hook 或 monkey patch  | `use_fused_kernels=True` 且 remove-padding | 示例启用，但不是 Bridge 建模所必需   |
| `apply_router_replay_patch()`                         | monkey patch                | R2/R3 开启                                  | 普通初始化不需要；#6473 示例 R3 需要 |
| DSv4 THD shard 至少 pad 到 CSA window                    | forward preprocessing fix   | fused/THD forward                         | 发生在 forward，不是建模阶段      |
| recomputation backward patch                          | generic compatibility patch | CUDA engine 初始化                           | 不是 DSv4 专属              |
| Bridge `apply_overrides_and_finalize` 缺失时 fallback    | version shim                | 较旧 Bridge                                 | 不是 DSv4 算法逻辑            |




#### fused forward patch

`[verl] MegatronEngine._maybe_enable_fused_kernels()` 调用：

```python
# [verl]
patch_fused_forward(megatron_gpt_model)
```

新 Megatron-Core 若支持 `output_processor` hook，verl 优先使用第三方提供的原生 hook，不替换 `forward`；旧版本没有这个 hook 时，verl 才保存 `forward_backup` 并 monkey-patch `GPTModel.forward`。

这个 patch 的目的主要是适配 verl PPO 所需的：

- packed/remove-padding 输入；
- temperature；
- fused linear cross entropy；
- 直接返回 token logprob 与 entropy；
- 把 `input_ids` 继续传给 DeepSeek-V4 hash router；
- 接受 DSv4 decoder 返回 tuple 的情况。

因此它是 `[verl] PPO training interface patch`，不是 Megatron-Bridge 用来创建 DeepSeek-V4 结构的代码。

#### router replay patch

当 `router_replay.mode != disabled` 时，verl 在 Megatron engine 构造阶段调用：

```python
# [verl monkey patch]
apply_router_replay_patch()
```

对支持原生 learned-router replay 的新 Megatron-Core，verl 主要扩展 DeepSeek-V4 hash router 的 route record，并修正重复 expert ID 时 dispatcher 的 token count；对旧 Megatron-Core，它还会 patch `TransformerConfig`/`TopKRouter` 以补齐 replay 能力。

这属于 R2/R3 对齐功能。禁用 replay 后，DeepSeek-V4 仍然可以完成 Bridge 初始化和普通 forward。

#### DSA/THD padding fix

verl 的 fused forward preprocessing 检查：

```python
if config.experimental_attention_variant == "dsv4_hybrid":
    min_local_rows = config.csa_window_size
```

然后在 THD packed buffer 太短时补齐。这是 `[verl] forward data-layout fix`；真正的 DSA/CSA kernel 仍在 Megatron-Core/Transformer Engine。它在第一次 forward 才生效，不改变 checkpoint→parameter 的初始化流程。

### 2.13 actor 与 ref 是否走不同的 Bridge

没有。actor 和 ref 都使用相同的：

```text
[verl] TrainingWorker
  → [verl] MegatronEngine
  → [Megatron-Bridge] AutoBridge / DeepSeekV4Bridge
  → [Megatron-Core] GPTModel
```

差别是 verl 给它们的运行配置不同：


| actor                  | ref                       |
| ---------------------- | ------------------------- |
| `forward_only=False`   | `forward_only=True`       |
| 创建 optimizer/scheduler | 不创建 optimizer/scheduler   |
| 参数可训练                  | 参数冻结，只计算 logprob          |
| 可按配置启用 MTP             | verl 强制 ref MTP=false     |
| 初始化后准备反向传播             | 初始化后可直接按 offload 配置迁到 CPU |


但默认 #6473 示例没有开启 KL，所以通常只初始化 actor，不创建 ref。只有 `actor.use_kl_loss=True` 或 `algorithm.use_kl_in_reward=True` 时，verl 才把 role 设成包含 ref，并让 `ActorRolloutRefWorker.init_model()` 先建 ref、再建 actor。

### 2.14 初始化过程的最终心智模型

按归属重画一次：

```text
[verl]
ActorRolloutRefWorker.init_model
  │
  ├─ [Transformers/HF] 读取 config/tokenizer/checkpoint schema
  │
  ├─ [verl] EngineRegistry 选择 MegatronEngine
  │
  ├─ [Megatron-Core] 初始化 TP/PP/EP/CP groups
  │
  ├─ [Megatron-Bridge] AutoBridge 识别 DeepSeekV4Bridge
  │      └─ HF config → MLAModelProvider
  │
  ├─ [verl] 覆盖并行度、BF16、MTP、CP、router replay 等 job 配置
  │
  ├─ [Megatron-Bridge] provider.provide_distributed_model
  │      └─ [Megatron-Core] 创建 distributed GPTModel/DDP/layers
  │
  ├─ [Megatron-Bridge] load_hf_weights
  │      ├─ DSv4 parameter mapping
  │      ├─ FP8/MXFP4 → BF16
  │      └─ TP/PP/EP shard 分发与 copy
  │
  ├─ [verl] 安装配置所需的 fused/replay/compatibility patch
  │
  └─ [verl + Megatron-Core] 创建 optimizer/scheduler
```

一句话总结：

> Megatron-Bridge 负责“理解 DeepSeek-V4 并翻译”；Megatron-Core 负责“创建和运行分布式模型”；verl 负责“决定本次 RL 任务怎样组织它们”，并在配置、接口或版本不匹配的边缘位置做 patch。



## 3. “四个 PR”应怎样划分

v0.9.0 release note 把几项工作写在同一条 DeepSeek-V4 条目里，但职责不同：


| PR                                                      | 作用                                                                      | 本文覆盖程度                        |
| ------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------- |
| [#6473](https://github.com/verl-project/verl/pull/6473) | 首次打通 Megatron-Bridge actor/ref、vLLM rollout、GRPO、R3，以及 FP8/MXFP4 在线权重更新 | 主体                            |
| [#7224](https://github.com/verl-project/verl/pull/7224) | 继续修正并拆分 vLLM 的 DeepSeek-V4 FP8/MXFP4 linear/MoE refit                   | 解释 v0.9.0 最终代码为何与 #6473 文件名不同 |
| [#7221](https://github.com/verl-project/verl/pull/7221) | 增加 DeepSeek-V4 所需的 contiguous context-parallel layout                   | 只划清边界                         |
| [#7297](https://github.com/verl-project/verl/pull/7297) | 补齐 CP 配置与执行修复，使长上下文实际可运行                                                | 只划清边界                         |


所以，“#6473 是四个 PR”不太准确；更准确的说法是：release note 的这一项由四个 PR 共同完成，#6473 是端到端主干，其余三个是在 v0.9.0 发布前补齐量化 refit 与长上下文 CP。

## 4. 两个时间截面：不要混着读



### 4.1 #6473 合入时

新增示例：

```text
examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh
```

原始默认值大致是：


| 项目               | #6473 合入时                                       |
| ---------------- | ----------------------------------------------- |
| 集群               | 11 nodes × 8 GPUs = 88 GPUs                     |
| actor 并行         | TP=1, PP=11, EP=8, CP=1                         |
| actor DP         | `88 / (1 × 11 × 8 × 1) = 1`                     |
| rollout          | vLLM TP=8                                       |
| batch            | prompt batch=8，每 prompt 采样 `n=2`                |
| 长度               | prompt 2048，response 10240                      |
| rollout KV cache | FP8                                             |
| router replay    | R3                                              |
| 权重 bucket        | 512 MiB                                         |
| trainer          | `trainer.use_v1=False`，走 legacy `RayPPOTrainer` |
| checkpoint       | 每 5 step 保存一次                                   |


actor 的 pipeline layout 是：

```text
Et*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3L
```

可以把 `E`、`t`、`L` 粗略理解为 embedding、transformer layer 和 loss/output stage。DeepSeek-V4 的 hash-routed MoE 层与 embedding 的 stage 位置有关，因此这里没有完全依赖自动切 PP，而是显式指定布局。

### 4.2 v0.9.0 tag 中的最终示例

发布版本已经有明显变化：


| 项目         | v0.9.0 tag                                                 |
| ---------- | ---------------------------------------------------------- |
| 集群变量       | `NNODES=8`、每节点 8 GPU，即 64 GPUs；脚本头部“11 nodes”注释已滞后         |
| actor 并行   | TP=1, PP=8, EP=8，默认仍为 CP=1                                 |
| rollout    | TP=1, DP=4, EP=4                                           |
| 采样数        | `n=8`                                                      |
| 数据         | 启用 DeepSeek-V4 continuous-token 处理和 `enable_thinking`      |
| trainer    | 未覆盖 `use_v1`，因此使用 v0.9.0 默认的 V1 sync trainer               |
| checkpoint | `save_freq=-1`，示例默认不保存                                     |
| CP         | 当 `ACTOR_CP>1` 时追加 contiguous layout、packing scheduler 等参数 |


因此学习源码时应区分：

- 想理解 PR 作者最初打通的调用路径，看 `cf8005fd` 的 legacy trainer；
- 想按 v0.9.0 真正运行，看 tag 下的 V1 sync trainer；
- 二者的算法闭环和 DeepSeek-V4 模型边界相同，调度器与 worker API 不同。



## 5. 进程和组件分工



### 5.1 控制面

启动命令是：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  model_engine=megatron \
  actor_rollout_ref.rollout.name=vllm \
  ...
```

`main_ppo.py` 负责：

1. Hydra 合并 yaml 与命令行配置；
2. `need_reference_policy()`、`need_critic()` 判定要不要 ref/critic；
3. 初始化 Ray；
4. 启动远程 TaskRunner；
5. TaskRunner 创建 worker groups、rollout servers、checkpoint/weight-sync manager、reward 组件，然后进入 `fit()`。



### 5.2 数据面


| 组件          | 实现                                       | 持有什么                         | 做什么                                                 |
| ----------- | ---------------------------------------- | ---------------------------- | --------------------------------------------------- |
| actor       | Megatron-Core，经 Megatron-Bridge 构造       | BF16 可训练参数、optimizer、grad    | 重算 logprob，执行 PPO/GRPO 更新                           |
| ref         | 同为 Megatron engine，可选                    | 冻结的参考参数                      | 只计算 `ref_log_probs`                                 |
| rollout     | vLLM async server                        | checkpoint 原生 FP8/MXFP4 推理布局 | autoregressive generation、rollout logprob、R3 routes |
| reward      | DAPO reward manager                      | 规则/任务评分函数                    | 把 response 转成 token-level score                     |
| controller  | legacy RayPPOTrainer 或 V1 PPOTrainerSync | batch metadata、UID、metrics   | 编排完整 step                                           |
| weight sync | checkpoint manager + bucketed transfer   | HF 命名的 tensor stream         | actor 更新后刷新 rollout                                 |


这些角色默认共用同一批 GPU，靠 sleep/offload 交替腾显存，而不是同时常驻完整 actor、ref 和 vLLM。

## 6. 冷启动：同一个 checkpoint 变成两种模型



### 6.1 配置加载的兼容处理

DeepSeek-V4 发布时，部分 Transformers 版本还没有注册 `deepseek_v4` model type。#6473 的处理原则是 fail narrowly：

- 只有遇到这个精确的 missing-model-type 情况，才借 vLLM 的 config loader 读取配置；
- 其他配置异常继续抛出，避免把真正错误吞掉。

rollout 侧还会规范化 HF overrides：禁用不需要的 MTP，并整理 YaRN RoPE 字段，使 vLLM 能接受模型配置。

### 6.2 actor/ref：HF config → Megatron provider

Megatron engine 的关键调用链是：

```text
MegatronEngine 初始化
  └─ AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
       └─ bridge.to_megatron_provider(load_weights=False)
            └─ 覆盖 TP / PP / EP / ETP / CP / pipeline layout
                 └─ provider finalize
                      └─ make_megatron_module(...)
                           └─ bridge.load_hf_weights(module, model_path)
```

verl 覆盖或补充的 Megatron 配置包括：

- variable sequence lengths；
- FlashAttention backend；
- MoE all-to-all dispatcher；
- router replay；
- fused DSA 与 recompute；
- TP/PP/EP/ETP/CP 以及显式 pipeline layout；
- 当 verl 关闭 MTP 时，把 provider 的 `mtp_num_layers` 置 0，并裁剪对应 CSA 元数据。

Megatron-Bridge 的 DeepSeek-V4 provider 负责把 HF config 中的 MLA、CSA/DSA、mHC、MoE、hash router、SwiGLU clamp、MTP 等字段映射为 Megatron-Core 的实验性 `dsv4_hybrid` attention/model 配置。

权重映射不只是改前缀。它涵盖：

- embedding、LM head、norm；
- attention 的 `wq`、`wkv`、`wo`；
- router gate、router bias、token-to-expert table；
- expert `w1/w3 → linear_fc1`，`w2 → linear_fc2`；
- shared experts；
- hyperconnection 与 MTP 参数。

checkpoint 中的量化权重在导入训练模型时会被反量化；所以 actor/ref 的训练计算使用 BF16 参数，而不是直接拿 MXFP4 参数做 optimizer update。

### 6.3 rollout：保留推理量化布局

vLLM 从原始 DeepSeek-V4-Flash checkpoint 启动，读取模型自带的 quantization config。即使命令行没有显式写 `rollout.quantization=fp8`，verl 也会从 HF config 识别它并在 vLLM 构建模型前安装量化兼容 patch。

这一步得到的是适合推理 kernel 的布局：

- dense、attention、shared expert 等主要使用 E4M3 FP8 权重与 UE8M0 scale；
- routed MoE experts 使用 packed MXFP4/E2M1，两个 4-bit value 装在一个 byte 中，并配套 block scale；
- vLLM 初次 load 后会把 checkpoint raw tensor 变换为 MXFP4/MegaMoE kernel layout，有些 parameter 会被替换，原始 loader metadata 也可能消失。

至此，同一份初始策略有两个物理表示：Megatron BF16 training layout 与 vLLM quantized inference layout。

### 6.4 首次权重同步为什么仍然需要

trainer 加载/恢复 checkpoint 后，在第一次 generation 前调用一次 `checkpoint_manager.update_weights()`：

- 保证 rollout 与刚刚恢复的 actor 完全一致；
- 统一走以后每个 step 都会使用的 Bridge export + vLLM refit 路径；
- 提前暴露 tensor name、shape、scale 或 packed-layout 问题。



## 7. 一个 GRPO step 的完整数据流

下面按 #6473 legacy 路径描述；v0.9.0 V1 对应关系见第 12 节。

```mermaid
sequenceDiagram
    participant T as RayPPOTrainer
    participant V as vLLM rollout
    participant R as DAPO reward
    participant M as Megatron actor
    participant F as Megatron ref(optional)
    participant W as Weight sync

    T->>V: prompt × rollout.n
    V-->>T: response, rollout_log_probs, routed_experts(R3)
    T->>V: sleep / release cache
    T->>R: prompt + response
    R-->>T: token_level_scores
    T->>M: recompute old_log_probs with replayed routes
    M-->>T: old_log_probs, entropy
    opt ref enabled
        T->>F: compute ref_log_probs
        F-->>T: ref_log_probs
    end
    T->>T: group reward → GRPO advantages
    T->>M: PPO clipped loss + optional KL → optimizer step
    T->>W: export updated actor weights
    W->>V: bucketed FP8/MXFP4 refit
    V-->>T: wake KV cache; ready for next rollout
```





### 7.1 取 prompt 并建立 group identity

trainer 从 DAPO-Math-17k 一类 parquet 数据集中取 `train_batch_size` 个 prompt，为每个原始 prompt 分配唯一 `uid`。

接着按 `rollout.n` interleave repeat。相同 `uid` 的多个 response 构成一个 GRPO group。后续即使为了平衡有效 token 数而打乱 batch 顺序，也仍然按 `uid` 聚合，不依赖样本相邻。

### 7.2 vLLM rollout

async rollout manager 把重复后的 prompt 发给 vLLM。每条结果至少带回：

- `response` token ids；
- response mask / EOS 信息；
- `rollout_log_probs`；
- R3 开启时的 `routed_experts`。

#6473 默认 `max_num_seqs=1`、`enforce_eager=True`，属于偏保守的正确性配置。生成结束后 trainer 让 rollout replicas sleep，释放 KV cache/weights 所占空间，以便同卡上的 Megatron actor 开始计算。

### 7.3 reward

DAPO reward manager 对每条 response 计算任务得分。示例还提供 overlong buffer/penalty 配置，但默认未启用。

若未启用 reward KL，则：

```text
token_level_rewards = token_level_scores
```

若启用 `algorithm.use_kl_in_reward=True`，还会用 ref logprob 对 token reward 加 KL penalty。

### 7.4 重算 old log probability

verl 默认不是直接把 `rollout_log_probs` 当 PPO anchor，而是让当前 actor 对采样结果再前向一次，得到 `old_log_probs`。

这样做有两个目的：

- PPO mini-batch 内有稳定的 proximal anchor；
- 可以度量 vLLM rollout policy 与 Megatron actor 的 logprob mismatch。

对 DeepSeek-V4 MoE，R3 会在这次 Megatron forward 中重放 rollout 的 expert IDs。否则即便权重相同，两个实现也可能因为数值差异走到不同 expert，最终 logprob 差异被放大。

### 7.5 GRPO advantage

对 response `i`，先把有效 response token reward 相加：

```math
R_i = \sum_t r_{i,t}
```

对同一 prompt/UID 的 group `G`：

```math
A_i = \frac{R_i - \operatorname{mean}(R_G)}{\operatorname{std}(R_G) + \epsilon}
```

再把同一个 scalar advantage 广播到该 response 的所有有效 token：

```math
A_{i,t} = A_i \cdot m_{i,t}
```

设置 `algorithm.norm_adv_by_std_in_grpo=False` 时只减均值，不除标准差，对应 Dr.GRPO 风格。若一个 group 只有一条样本，当前实现把 mean 设为 0、std 设为 1；正常 GRPO 应配置 `rollout.n > 1`。

GRPO 在此默认不需要 critic/value model，因此不会走 GAE 与 critic update。

### 7.6 actor loss 与 optimizer step

Megatron actor 重新前向，当前 token logprob 记为 `log_prob`，PPO ratio 为：

```math
\rho_{i,t} = \exp(\log \pi_\theta - \log \pi_{old})
```

基础 clipped objective 使用 `ρA` 与 `clip(ρ)A` 中更保守的一项；verl 默认实现还包含 dual-clip 对负 advantage 的下界处理。response mask 决定哪些 token 参与 loss aggregation。

如果开启 `actor.use_kl_loss=True`，actor loss 还会加入相对 reference policy 的 KL 项；否则 ref 不参与。

最后由 Megatron pipeline schedule 执行 forward/backward、梯度同步和 optimizer step。此时 actor 已变成策略 `π_(k+1)`，而 vLLM 还是 `π_k`，必须立即同步权重。

## 8. actor/ref 标题中的一个关键事实

PR 标题和说明写的是 “Megatron-Bridge actor/ref”，原始脚本也定义了整组 `REF_*` 并行参数，但默认配置同时满足：

```yaml
actor_rollout_ref.actor.use_kl_loss: false
algorithm.use_kl_in_reward: false
```

verl 的判定是：

```text
need_reference_policy = actor.use_kl_loss OR algorithm.use_kl_in_reward
```

因此，**#6473 默认示例不会实例化 reference policy**。脚本中的 REF 数组只是能力配置，传入配置不等于实际创建模型。v0.9.0 最终脚本干脆移除了这组默认 REF 参数。

若要真的跑 Megatron-Bridge ref，可增加例如：

```bash
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001
```

或者开启：

```bash
algorithm.use_kl_in_reward=True
```

非 LoRA 场景下，hybrid worker 会在同一个 actor/rollout/ref worker group 中先创建冻结 ref，再创建 actor；二者都经 Megatron-Bridge 构造。代价是显存、加载时间和一次额外 logprob forward，因而依赖 offload 更明显。

## 9. R3 router replay：为什么 DeepSeek-V4 要特殊处理



### 9.1 R2 与 R3 的区别


| 模式  | 在哪里记录 route               | 在哪里重放                                       |
| --- | ------------------------- | ------------------------------------------- |
| R2  | Megatron 计算 old logprob 时 | Megatron actor update                       |
| R3  | vLLM rollout 时            | Megatron old-logprob forward 和 actor update |


R2/R3 都不是“模型能运行”的必要条件，而是跨阶段对齐功能。#6473 示例默认用 R3。

### 9.2 为什么前三个 MoE 层会错位

DeepSeek-V4-Flash 前三个 routed layer 使用 hash router：expert ID 来自 `input_ids` 与 token-to-expert table，而不是 learned router logits。

旧的 router replay 只拦截 learned top-k router，因此会发生：

```text
vLLM routes:    [hash0, hash1, hash2, learned3, learned4, ...]
Megatron routes:[learned3, learned4, ...]
```

如果直接按 layer index 重放，从第一层起就全部错位。#6473 为此：

1. 把 `input_ids` 传入 DeepSeek-V4 decoder；
2. 除 learned top-k 外，也记录三层 hash-router 输出；
3. 保证 vLLM 与 Megatron 的 routed-layer 顺序一致。



### 9.3 causal mask 不是 response mask

autoregressive rollout 记录的是“用于预测下一个 token 的当前 row 所走的 route”。所以需要重放的是所有会影响 response logits 的 row，而不只是被标记为 response token 的 row；最后一个生成 token 对应的 model row 没有下一个被使用的 logit，通常不重放。

v0.9.0 代码会逐 sequence 对齐 route length；若 rollout 少了最后一行，则追加 ignored placeholder，使 jagged route tensor 与完整 `input_ids` 行数匹配，再用 causal replay mask 决定实际重放位置。

另一个细节是 top-k 结果可能包含重复 expert ID。此时 dispatcher 的 token count 必须从构造后的 routing map 求和，不能武断使用 `num_tokens × topk`。

### 9.4 route 数据在闭环中的位置

```text
vLLM final output
  └─ routed_experts
       └─ agent/rollout output
            └─ training batch
                 ├─ actor.compute_log_prob(): REPLAY_FORWARD
                 └─ actor.update_actor(): REPLAY_FORWARD / REPLAY_BACKWARD
```

若 R2 和 R3 同时向 batch 填 `routed_experts`，trainer 会显式报冲突，而不是猜测应使用哪一份。

## 10. 最核心的工程点：FP8/MXFP4 在线权重热更新



### 10.1 为什么普通 weight copy 不成立

actor update 后，Megatron 持有的是：

- 按 TP/PP/EP 切分的训练参数；
- 通常为 BF16 计算表示；
- Megatron-Core 命名和布局。

vLLM 运行时持有的是：

- HF/checkpoint 风格入口名经过 vLLM loader 映射后的参数；
- dense FP8 + UE8M0 scales；
- routed experts 的 packed MXFP4；
- `w1`/`w3` 已合并成 `w13`；
- 可能经过 kernel-specific permutation/packing 的对象。

因此 name、shape、dtype、分片和内存布局都可能不同。

### 10.2 Megatron 侧导出

actor worker 的 `update_weights()` 先调用：

```text
actor.engine.get_per_tensor_param()
```

Megatron engine 会：

1. 把 offloaded actor parameters 暂时召回 GPU；
2. 调用 `bridge.export_hf_weights(self.module)`；
3. 根据 Bridge conversion tasks 从 TP/PP/EP shards 组合出逻辑完整 tensor；
4. 转回 HF/checkpoint 命名；
5. 对 DeepSeek-V4 重新产生量化 weight/scale 对；
6. 以 generator 形式逐 tensor 输出，避免一次在 driver 堆完整 state dict。

这里导出的不是 vLLM 最终 kernel layout，而是 vLLM `load_weights()` 能识别的 checkpoint/raw layout。

### 10.3 bucketed CUDA IPC 传输

vLLM rollout 先远程启动 receiver，然后 actor 侧的 `BucketedWeightSender` 开始发送：

```text
Bridge tensor generator
  └─ 固定 512 MiB uint8 CUDA buffer
       ├─ tensor bytes 复制进 buffer
       ├─ ZMQ 发送 name/shape/dtype/offset/is_last 元数据
       └─ receiver 通过 CUDA IPC handle 重建 tensor views
```

要点：

- 大块 tensor 数据走 CUDA IPC，共享内存是无 IPC 平台的 fallback；
- ZMQ 主要传控制信息、metadata 与 ACK，不是把整个模型字节流经 socket 复制；
- 单个 tensor 大于 bucket 时，CUDA 路径可直接发送该 tensor 的 IPC handle；
- sender 等待每个 bucket 的 ACK，因此复用同一 buffer 时不会覆盖 receiver 尚未消费的数据。



### 10.4 prepare → load all buckets → finalize

这是 #6473 最重要的不变量：

```text
prepare once
  └─ load bucket 0
  └─ load bucket 1
  └─ ...
  └─ load final bucket
finalize once
```

不能对每个 bucket 都做 vLLM post-processing，因为 post-processing 是非幂等的：第一桶完成后，expert raw parameters 可能已经被替换成 packed kernel layout，第二桶再按 checkpoint shape 加载就会错。

prepare 阶段负责：

- 恢复 dense linear 所需的 parameter subclass、shape 与 `weight_loader`；
- 恢复 MXFP4 expert 的 raw `w13`、`w2` 和 scale 形状；
- 保存 finalize 所需的原布局状态。

每个 bucket 的 load 阶段负责：

- 让 vLLM 自己的 loader 做 TP slicing；
- 完成 HF name 到 vLLM name 的匹配；
- 把 `w1`、`w3` 合并到 `w13`；
- 保持 UE8M0 scale dtype/语义；
- 对 DeepSeek-V4 已由 Bridge 导出的 quantized expert tensor，避免无意义的 BF16→再次量化。

finalize 阶段负责：

- 仅在最后一个 bucket 之后重新构建 MXFP4/MegaMoE kernel layout；
- 恢复/安装 dense FP8 scale shards；
- 保证后续 generation 使用的是 vLLM kernel 期望的表示。

完成后 rollout 清 KV cache、记录最新 global step，并恢复 KV-cache memory；actor 若启用 param offload 则重新迁回 CPU。

### 10.5 #6473 与 #7224 的文件结构差异

#6473 初版把 DeepSeek-V4 特殊逻辑集中在：

```text
verl/utils/vllm/vllm_dsv4_fp8_utils.py
verl/utils/vllm/vllm_fp8_utils.py
verl/workers/rollout/vllm_rollout/utils.py
```

到 v0.9.0，#7224 将职责整理为：

```text
verl/utils/vllm/vllm_quant_utils.py  # 总入口和调度
verl/utils/vllm/vllm_fp8_utils.py    # FP8 linear/MoE stage + finalize
verl/utils/vllm/vllm_fp4_utils.py    # DeepSeek-V4 MXFP4 expert stage + finalize
```

所以在 v0.9.0 tag 中找不到 `vllm_dsv4_fp8_utils.py` 并不是功能消失，而是被后续 PR 拆分、修正了。#7224 还进一步关注 parameter address/layout 的稳定性，以免破坏已捕获的 CUDA graph。

## 11. 在线热更新与 checkpoint 保存是两条路径

这两件事都可能调用 Megatron-Bridge export，但消费者不同：


| 路径                           | 消费者                | 目标           | 时机                     |
| ---------------------------- | ------------------ | ------------ | ---------------------- |
| actor → rollout live sync    | 正在运行的 vLLM workers | 下一轮立即用新策略采样  | 初始化后、每次 actor update 后 |
| persistent checkpoint export | 文件系统/HF checkpoint | 恢复训练、离线部署或发布 | `save_freq` 命中时        |


#6473 明确依赖 Megatron-Bridge [PR #3969](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3969) / commit `[c7774d44](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11)`。这项 Bridge 改动主要保证量化 HF checkpoint 导出完整：

- 每个 quantized `.weight` 生成并导出匹配的 `.scale`；
- E4M3 与 MXFP4 分别按其来源 weight/scale 几何重新生成；
- 保持 E8M0 scale 的 shape/dtype；
- verl 关闭 MTP 时跳过 `mtp.*` source key；
- 让 safetensors/export key 处理适配源权重前缀。

仅验证 live weight sync 成功，不等价于 optimizer/RNG/trainer state checkpoint 能保存和恢复；反过来，HF checkpoint 文件能写出，也不等价于 vLLM 热更新后的 packed layout 正确。两条路径都应单独测试。

## 12. legacy #6473 与 v0.9.0 V1 的调用链映射



### 12.1 #6473 合入时的主路径

```text
verl.trainer.main_ppo.main
  └─ main_ppo_v0.TaskRunner.run
       ├─ 创建 ActorRolloutRefWorker
       ├─ worker.init_model(): ref(optional) → actor → rollout
       ├─ RayPPOTrainer.init_workers()
       └─ RayPPOTrainer.fit()
            ├─ _load_checkpoint()
            ├─ checkpoint_manager.update_weights()  # 初始同步
            ├─ async_rollout_manager.generate_sequences()
            ├─ checkpoint_manager.sleep_replicas()
            ├─ reward
            ├─ _compute_old_log_prob()
            ├─ _compute_ref_log_prob()              # optional
            ├─ compute_advantage(GRPO)
            ├─ _update_actor()
            ├─ _save_checkpoint()                   # optional
            └─ checkpoint_manager.update_weights()  # 更新 rollout
```



### 12.2 v0.9.0 tag 的实际默认路径

```text
verl.trainer.main_ppo.main
  └─ TaskRunnerV1.run
       ├─ transfer_queue.init()
       ├─ PPOTrainerSync.init()
       │    ├─ actor/ref worker groups
       │    ├─ LLMServerManager
       │    ├─ CheckpointEngineManager
       │    └─ load checkpoint + initial update_weights
       ├─ AgentLoopManagerTQ
       └─ PPOTrainerSync.fit()
            ├─ agent loop 向 TransferQueue 生产 rollout/reward 结果
            ├─ replay buffer sample
            ├─ sleep rollout replicas
            ├─ compute old/ref logprob
            ├─ GRPO advantage
            ├─ update actor
            └─ on_step_end(): update_weights
```

V1 把生成端与训练端用 TransferQueue/replay buffer 解耦，为 async trainer 复用；但 v0.9.0 DeepSeek-V4 示例选的是 `trainer.v1.trainer_mode=sync`，因此策略语义仍是同步闭环：一个 step 更新完成并同步后，下一批 rollout 才应使用新版本。

## 13. #6473 还补了哪些兼容性问题

除了主循环，还有几项容易被忽略但会直接造成启动或 forward 失败的修复：

- MTP 在 verl 侧关闭时，Bridge provider 也必须关闭，否则层数和 CSA metadata 对不上；
- fused DSA 的每个本地 THD shard 至少要容纳一个 CSA window，太短时需要 pad，计算后再 unpad；
- DeepSeek-V4 decoder 可能返回 tuple，fused forward 需要取实际 tensor output；
- remove-padding/THD 预处理测试覆盖了最小 shard padding；
- R3 的 hash route 必须把 `input_ids` 一路传到 decoder/router。

后续 #7221/#7297 处理的是 CP>1 的另一组问题。v0.9.0 脚本会追加：

```text
cp_partition_mode=contiguous
sequence_packing_scheduler=dp_balanced
max_seqlen_per_dp_cp_rank=(max_prompt + max_response) / CP
```

这不是 #6473 最初默认 CP=1 闭环的一部分，后面学习长上下文 CP 时应单独展开。

## 14. 建议的验证顺序

官方集群默认规模很大，不建议一上来就把所有功能同时打开。更容易定位问题的顺序是：

1. **关 replay，跑至少两个 optimizer step**。第二次 actor→vLLM 更新才能验证“vLLM 已 post-process 过一次后还能再次 refit”。
2. **对齐 logprob**。同时看 Pearson correlation、max/mean/std difference；只看到 `load_weights` 成功不代表 scale 或 packed layout 正确。
3. **打开 R3**。检查 route tensor 的 layer 数，尤其前三个 hash-routed layers；确认没有 R2/R3 冲突。
4. **单独打开 ref**。验证 `need_reference_policy=True`、ref worker 确实构建，并观察 ref logprob/KL 指标。
5. **打开 checkpoint 保存并恢复**。检查 model、optimizer、scheduler、RNG、trainer/global step，而不只看 HF weight 文件。
6. **最后扩大长度与 CP**。短序列 smoke test 覆盖不到 CSA window、THD packing、contiguous CP 和长上下文峰值显存问题。

常见症状与优先排查方向：


| 症状                                    | 优先排查                                                     |
| ------------------------------------- | -------------------------------------------------------- |
| expert shape 如 1024 vs 2048           | MXFP4 raw layout 未 prepare，或 w1/w3→w13 合并时机错误            |
| 权重加载无异常，但 rollout/actor logprob 严重不一致 | scale dtype/shape、FP4 packing、finalize 次数、route mismatch |
| 只有第二次更新失败                             | post-processing 被逐 bucket 执行，非幂等状态未恢复                    |
| route layer 数少 3                      | hash-routed layers 未记录或没传 `input_ids`                    |
| 开 CP 后构建/首个 attention forward 失败      | contiguous layout、packing scheduler、每 DP×CP rank 最大长度配置  |
| 标题说 ref，但日志中没有 ref                    | 两个 KL 开关都为 false，属于默认行为                                  |




## 15. 推荐的源码阅读顺序



### 第一遍：看闭环

1. `examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh`
2. `verl/trainer/main_ppo.py`
3. #6473 时：`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`
4. v0.9.0 时：`verl/trainer/ppo/v1/trainer_base.py` 与 `trainer_sync.py`
5. `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`



### 第二遍：看 Megatron actor/ref

1. `verl/workers/engine_workers.py::ActorRolloutRefWorker`
2. `verl/workers/engine/megatron/transformer_impl.py::MegatronEngine`
3. Megatron-Bridge：`models/deepseek/deepseek_v4_bridge.py`
4. `verl/models/mcore/model_forward_fused.py`



### 第三遍：看 R3

1. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
2. `verl/utils/megatron/router_replay_patch.py`
3. `verl/utils/megatron/router_replay_utils.py`
4. `verl/workers/engine/megatron/transformer_impl.py` 中 replay record/forward/backward 部分



### 第四遍：看量化权重同步

1. `verl/workers/engine/megatron/transformer_impl.py::get_per_tensor_param`
2. `verl/workers/engine_workers.py::update_weights`
3. `verl/workers/rollout/vllm_rollout/vllm_rollout.py::update_weights`
4. `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
5. `verl/workers/rollout/vllm_rollout/utils.py::update_weights_from_ipc`
6. v0.9.0 的 `vllm_quant_utils.py`、`vllm_fp8_utils.py`、`vllm_fp4_utils.py`



## 16. 依赖版本与复现实用提醒

#6473 示例锁定的关键依赖是：

```text
Megatron-Bridge c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
Megatron-LM     fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
```

不要随意拿最新 Megatron-Bridge/Megatron-LM 替换后，仍假定类名、provider 字段、vLLM loader 与 patch 全部相同。尤其 vLLM 的 MoE parameter ownership 与 post-load API 演进很快，verl v0.9.0 已包含多版本兼容代码。

本次研究为核对 Bridge 的 DeepSeek-V4 provider 与量化 export，临时读取了上述固定 Bridge commit；没有把第三方仓库复制进 verl 工作树。后续若要实际复现，建议在独立 dependency 目录或容器镜像中按 commit 安装，而不是把 clone 混进当前 Git 仓库。

## 17. 一句话心智模型

把整个系统看成一个“训练表示 ↔ 推理表示”的同步状态机：

```text
vLLM(π_k, quantized) rollout
  → route/logprob/reward
  → Megatron(BF16) 计算 GRPO 并得到 π_(k+1)
  → Bridge 恢复 checkpoint 语义
  → IPC 搬运
  → vLLM 恢复 kernel 语义
  → vLLM(π_(k+1), quantized) rollout
```

GRPO 决定“参数往哪里更新”；Megatron-Bridge 决定“同一参数在训练世界和 checkpoint 世界如何对应”；vLLM refit 决定“新参数怎样安全进入量化推理 kernel”；R3 决定“两个世界是否走过同一组 experts”。四者缺一个，DeepSeek-V4 的端到端 on-policy 闭环都可能只是表面跑通。

## 参考链接

- [verl v0.9.0 release notes](https://github.com/verl-project/verl/releases/tag/v0.9.0)
- [verl PR #6473](https://github.com/verl-project/verl/pull/6473)
- [verl PR #7224](https://github.com/verl-project/verl/pull/7224)
- [verl PR #7221](https://github.com/verl-project/verl/pull/7221)
- [verl PR #7297](https://github.com/verl-project/verl/pull/7297)
- [Megatron-Bridge PR #3969](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3969)
- [Megatron-Bridge pinned commit](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11)
- [Megatron-LM pinned commit](https://github.com/NVIDIA/Megatron-LM/commit/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1)

