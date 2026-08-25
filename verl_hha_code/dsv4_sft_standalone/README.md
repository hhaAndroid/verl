# 独立 DeepSeek-V4 Megatron-Bridge/MCore SFT smoke test

这个目录是一个**完全独立于 verl Python 包**的最小工程。训练脚本不会
`import verl`，也不会加载 DeepSeek 官方权重。

结合 8×H200 实际 forward hooks、profiler 通信事件和 Bridge/MCore 源码逐段对照的
专题报告见 [`DSV4_FORWARD_ANALYSIS.md`](./DSV4_FORWARD_ANALYSIS.md)。

## 它验证什么

执行链路是：

1. 在内存里创建一个可缩层的 `DeepseekV4Config`；
2. `[Megatron-Bridge] AutoBridge.from_hf_config()` 选择 DSv4 Bridge；
3. `[Megatron-Bridge] to_megatron_provider(load_weights=False)` 翻译配置；
4. `[本项目]` 覆盖 TP/PP/EP/CP 和算子开关；reference case 使用保守实现，fused
   case 显式开启 DSA fusion；CP=2 时按
   MCore DSv4 CP 单测设置 `sequence_packing_scheduler="dp_balanced"`；
5. `[Megatron-Bridge + Megatron-Core] provide_distributed_model()` 在 8 卡上随机初始化并用 MCore DDP 包装模型；
6. `[本项目 + Megatron-Core]` 生成合成 prompt/response、打包 THD batch，并调用 MCore 的 contiguous CP slice；
7. `[Megatron-Core]` forward、backward、梯度同步和 Adam 更新。

脚本故意不调用 `bridge.load_hf_weights()`。因此它验证的是“DSv4 模型结构和
SFT 训练链路能否跑通”，不是训练质量。

## 五个 case

```bash
bash run_cp1.sh
bash run_cp2.sh
bash run_dynamic_cp.sh
bash run_fused_cp1.sh
bash run_fused_dynamic_cp.sh
```

- CP=1：TP=1、PP=1、EP=8、CP=1；
- CP=2：TP=1、PP=1、EP=8、CP=2，并使用 DSv4 所要求的 contiguous THD layout、
  `dp_balanced` sequence-packing scheduler 配置、CuTeDSL layout kernel。
- Dynamic CP：静态 CP=1、DPxCP=8，但预创建 CP1/CP2/CP4/CP8 通信组；MCore
  `DefaultDynamicCPScheduler` 根据每条序列长度，在同一个 microbatch 内为不同样本选择
  不同的 runtime CP size。具体 rank 分配和源码归属见
  [`DYNAMIC_CP_EXAMPLE.md`](./DYNAMIC_CP_EXAMPLE.md)。
- Fused CP=1：显式设置 `apply_dsa_kernel_fusion=True`，执行 cuDNN DSA indexer、
  FlashMLA sparse forward、cuDNN DSA sparse-attention backward，并打开非零 indexer loss。
- Fused Dynamic CP：在 fused DSA 上叠加 CP1/CP2/CP4 动态调度，覆盖
  `q_causal_offsets`、THD/contiguous CP layout 和 backward。

这里没有单独提供“静态 CP=2 fused”脚本，因为 Dynamic case 的同一轮已经真实执行 CP2
和 CP4；如需固定 CP2，可执行 `bash run_cp2.sh --fused-dsa
--dsa-indexer-loss-coeff 0.001`。

默认 4 层、序列长度 128、两步。`NUM_LAYERS` 可以任意指定为大于等于 1 的整数；
第 0 层的 `compress_ratio=0`，后续层为 4，因此至少用 2 层才能同时覆盖 DSv4 的
压缩稀疏 attention 路径。快速排错示例：

```bash
NUM_LAYERS=1 TRAIN_STEPS=1 bash run_cp1.sh --print-structure
NUM_LAYERS=2 TRAIN_STEPS=1 bash run_cp2.sh
NUM_LAYERS=2 TRAIN_STEPS=1 bash run_dynamic_cp.sh
```

成功标志分别为 `SFT_SMOKE_SUCCESS cp=1`、`SFT_SMOKE_SUCCESS cp=2` 和
`DYNAMIC_CP_SFT_SUCCESS`；fused 日志还会明确打印 `fused_dsa=True`。

## 依赖边界和版本

Conda 环境名为 `dsv4_sft_bridge`，它由 `pt29_all_env` **克隆**得到；原环境不做
任何修改。目标源码固定为 verl v0.9.0 DSv4 示例所对应的提交：

- Megatron-Bridge: `c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11`
- Megatron-Core: `fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1`
- fast-hadamard-transform: `f134af63deb2df17e1171a9ec1ea4a7d8604d5ca`
- FlashMLA (`nv_dev`): `b7643bd54521f563b839b98289b5cd048c062ba2`
- FlashMLA CUTLASS submodule: `147f5673d0c1c3dcf66f78d677fd647e4a020219`
- cuDNN Frontend: `0a14b7181d129d30e7bad34b8c3ed0a0c995e23d`

源码放在本目录的 `third_party/`（已 gitignore），不是 verl 的 vendored 包。运行脚本
通过显式 `PYTHONPATH` 使用它们，不向 Conda 环境安装 `megatron-core` 或
`megatron-bridge` wheel，因此实际执行的源码 commit 很容易核对。

这个环境中与本工程直接相关的核心版本为：

- Python 3.12，PyTorch `2.9.1+cu128`；
- Transformers `5.8.1`；
- Transformer Engine `2.10.0`；
- nvidia-cudnn-cu12 `9.15.1.9`（安装 CuTeDSL 时保留，没有降级）；
- nvidia-cudnn-frontend `1.25.0`（使用上面的 MCore lockfile 源码提交）、
  nvidia-cutlass-dsl `4.5.0`；
- fast-hadamard-transform `1.0.4.post1`，使用上述固定提交和本机 CUDA 12.8 编译。
- flash-mla `1.0.0+b7643bd`，使用 CUDA 12.8 编译成 H200/SM90 sparse-prefill-only
  extension。

完整的逐条安装命令见 [`INSTALL_COMMANDS.md`](./INSTALL_COMMANDS.md)，其中包括 Conda
clone、精确 wheel hash、TE 组件来源、克隆环境内的 FA3 可恢复重命名、CuTeDSL、
Hadamard CUDA extension、H200 FlashMLA 和 cuDNN Frontend 源码版安装命令。

FlashMLA 当前 `nv_dev` 的 `FLASH_MLA_DISABLE_SM100=TRUE` 只去掉 SM100 gencode，仍会把
Blackwell translation units 交给 CUDA 12.8 编译。本项目只 patch 了
`third_party/FlashMLA`，将安装物裁剪为 MCore DSv4 **训练**实际需要的 SM90 sparse
prefill API；补丁保存在
[`patches/flashmla-sm90-prefill-only.patch`](./patches/flashmla-sm90-prefill-only.patch)。
它不提供 FlashMLA decode/dense API，不能当作通用推理 wheel。

CP=2 比 CP=1 多出的依赖不是 verl 引入的：MCore 的 DSv4 contiguous CP 实现会
直接调用 CuTeDSL 生成最终 attention indices，DSv4 的 indexer rotation 会调用
fast-hadamard-transform。两者缺失都会在真实 forward 中 fail closed。本工程没有用
单测里的 mock 替代它们，也没有 patch verl、Megatron-Bridge 或 Megatron-Core 源码；
唯一源码 patch 是上述独立 FlashMLA 的 CUDA 12.8 构建裁剪。

### 环境隔离说明

`dsv4_sft_bridge` 是从 `pt29_all_env` 克隆出的独立 Conda 环境。所有新增、禁用或编译
操作只发生在这个克隆环境；原始 `pt29_all_env` 没有被原地修改。环境安装遵守 verl
仓库指引：克隆 Conda 后，额外 Python 组件用克隆环境内的 `uv` 管理。

第三方源码都位于本工程 `third_party/`，没有写入 verl 源码目录：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git third_party/Megatron-LM
git -C third_party/Megatron-LM checkout --detach fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1

git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git third_party/Megatron-Bridge
git -C third_party/Megatron-Bridge checkout --detach c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
  third_party/fast-hadamard-transform
git -C third_party/fast-hadamard-transform checkout --detach \
  f134af63deb2df17e1171a9ec1ea4a7d8604d5ca

git clone --branch nv_dev --recurse-submodules \
  https://github.com/deepseek-ai/FlashMLA.git third_party/FlashMLA
git -C third_party/FlashMLA checkout --detach \
  b7643bd54521f563b839b98289b5cd048c062ba2

git clone https://github.com/NVIDIA/cudnn-frontend.git third_party/cudnn-frontend
git -C third_party/cudnn-frontend checkout --detach \
  0a14b7181d129d30e7bad34b8c3ed0a0c995e23d
```

## 哪些内容来自哪里

- `AutoBridge`、DSv4 provider 翻译、DDP config helper：Megatron-Bridge；
- `GPTModel`、`DSv4HybridAttention`、CSA、MoE、`PackedSeqParams`、contiguous CP
  slicing、DDP/optimizer：Megatron-Core；
- CP layout 的 CuTeDSL kernel：nvidia-cutlass-dsl，由 Megatron-Core 调用；
- DSv4 indexer 的 Hadamard rotation：fast-hadamard-transform，由 Megatron-Core 调用；
- fused DSA 的 indexer forward/backward 与 sparse-attention backward：cuDNN Frontend
  `cudnn.DSA`，由 Megatron-Core 调用；
- fused DSA 的 sparse-attention forward：FlashMLA SM90 sparse prefill，由
  Megatron-Core adapter 调用；
- 合成 SFT 数据、prompt loss mask、THD buffer 组装、训练入口：本项目；
- verl 代码：**没有导入**。静态示例只参考调用边界；动态示例按 verl 的
  TensorDict adapter 职责重写了不依赖 TensorDict 的最小调度 glue，而 scheduler、
  动态通信组、contiguous slicing 和 attention 切组均调用 MCore 正式实现。详细对应关系
  见 [`DYNAMIC_CP_EXAMPLE.md`](./DYNAMIC_CP_EXAMPLE.md)。

环境自检：

```bash
export PYTHONPATH="$PWD/third_party/Megatron-LM:$PWD/third_party/Megatron-Bridge/src"
/mnt/shared-storage-user/huanghaian/miniconda3/envs/dsv4_sft_bridge/bin/python check_environment.py --require-fused
/mnt/shared-storage-user/huanghaian/miniconda3/envs/dsv4_sft_bridge/bin/python verify_fused_dsa_sm90.py
```

## 本机 8×H200 验证结果

以下是直接执行脚本得到的关键日志。静态两个 case 都是 4 层、序列长度 128、2 个真实
optimizer step；动态 case 是 2 层、4 条变长序列、1 个真实 optimizer step；全部从
随机权重开始：

```text
# bash run_cp1.sh
MODEL_READY layers=4 cp=1 ep=8 local_params=95,730,557
STEP_OK step=0 loss=9.672159 grad_norm=8.058254 update=True
STEP_OK step=1 loss=9.626540 grad_norm=8.093449 update=True
SFT_SMOKE_SUCCESS cp=1 layers=4 steps=2

# bash run_cp2.sh
MODEL_READY layers=4 cp=2 ep=8 local_params=95,730,557
STEP_OK step=0 loss=9.636520 grad_norm=11.353281 update=True
STEP_OK step=1 loss=9.650787 grad_norm=11.400719 update=True
SFT_SMOKE_SUCCESS cp=2 layers=4 steps=2

# bash run_dynamic_cp.sh
DCP_PLAN microbatch=0 groups=[CP4: seq512, CP2: seq256, CP1: seq128, CP1: seq64]
DYNAMIC_MODEL_READY layers=2 static_cp=1 runtime_cp_sizes=[1,2,4] ep=8 local_params=60,371,633
DYNAMIC_STEP_OK step=0 loss=9.679252 global_response_tokens=646 grad_norm=7.973556 update=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1
```

loss 不应拿来比较这些 case 的数值等价性，因为 synthetic batch 和 DP/CP 调度并不相同；
这里的验收目标是每种路径都完成有限 loss、有限 grad norm 和参数更新。

fused case 的实测结果如下。它们使用 2 层、1 个 optimizer step，并将
`dsa_indexer_loss_coeff=0.001`，因此不仅经过 sparse-attention backward，也真实经过
indexer loss/backward：

```text
# bash run_fused_cp1.sh
MODEL_READY layers=2 cp=1 ep=8 fused_dsa=True indexer_loss_coeff=0.001 local_params=95,924,593
STEP_OK step=0 loss=9.679355 grad_norm=6.337935 update=True
SFT_SMOKE_SUCCESS cp=1 layers=2 steps=1 fused_dsa=True

# bash run_fused_dynamic_cp.sh
DCP_PLAN microbatch=0 groups=[CP4: seq512, CP2: seq256, CP1: seq128, CP1: seq64]
DYNAMIC_MODEL_READY layers=2 static_cp=1 runtime_cp_sizes=[1,2,4] ep=8 fused_dsa=True indexer_loss_coeff=0.001 local_params=95,924,593
DYNAMIC_STEP_OK step=0 loss=9.643462 global_response_tokens=646 grad_norm=7.047264 update=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1 fused_dsa=True
```

## Cursor / VS Code 源码跳转

运行脚本通过 `PYTHONPATH` 合并两个 `megatron` namespace package，但编辑器不会自动
读取 shell 脚本里的 `PYTHONPATH`。本目录提供了独立 workspace 配置：

```bash
cursor dsv4_sft_standalone.code-workspace
# 或
code dsv4_sft_standalone.code-workspace
```

打开后确认右下角 Python interpreter 是：

```text
/mnt/shared-storage-user/huanghaian/miniconda3/envs/dsv4_sft_bridge/bin/python
```

然后在 `from megatron.bridge import AutoBridge` 的 `AutoBridge` 上使用 `F12` 或
`Ctrl/Cmd + 鼠标左键`，应跳到
`third_party/Megatron-Bridge/src/megatron/bridge/__init__.py`；继续跳转定义会到实际的
`models/conversion/auto_bridge.py`。`megatron.core` 则会跳到
`third_party/Megatron-LM/megatron/core/`。

如果当前窗口已经打开了整个 verl 仓库，建议用上面的 `.code-workspace` 新开窗口。
也可以保留当前窗口，但需要执行一次 `Python: Restart Language Server`。对应的
`pyrightconfig.json` 同样放在本目录，供 Pyright CLI 或非 VS Code 编辑器使用。
