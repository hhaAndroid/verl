# 独立 DSv4 SFT 环境：完整有效安装命令

这份记录列出本机最终成功环境所对应的完整有效命令。下载超时、第一次 Conda
离线 clone 失败等无效尝试没有混入命令清单。

需要特别说明：Megatron-LM 和 Megatron-Bridge **没有**执行 `pip install` 或
`pip install -e`。二者固定源码 commit 后由 `PYTHONPATH` 加载。

以下命令不会修改源环境 `pt29_all_env`，但目标环境目录必须是一个尚不存在的新路径。

## 1. 路径变量

```bash
export CONDA_ROOT=/mnt/shared-storage-user/huanghaian/miniconda3
export SOURCE_ENV=${CONDA_ROOT}/envs/pt29_all_env
export TARGET_ENV=${CONDA_ROOT}/envs/dsv4_sft_bridge
export COMPAT_ENV=${CONDA_ROOT}/envs/slime_megatron
export PROJECT_DIR=/mnt/shared-storage-user/huanghaian/code/verl/verl_hha_code/dsv4_sft_standalone

export PYTHON=${TARGET_ENV}/bin/python
export UV=${TARGET_ENV}/bin/uv
export TARGET_SITE=${TARGET_ENV}/lib/python3.12/site-packages
export COMPAT_SITE=${COMPAT_ENV}/lib/python3.12/site-packages
```

先确认源环境存在、目标环境尚不存在：

```bash
test -x "${SOURCE_ENV}/bin/python"
test ! -e "${TARGET_ENV}"
```

如果 `TARGET_ENV` 已经存在，不要直接覆盖或删除；请换一个新的目标环境名。

## 2. 克隆 Conda 环境

```bash
CONDA_NO_PLUGINS=true "${CONDA_ROOT}/bin/conda" create \
  --yes \
  --prefix "${TARGET_ENV}" \
  --clone "${SOURCE_ENV}"
```

这个操作只向 `dsv4_sft_bridge` 写入内容。原始 `pt29_all_env` 不会被修改。

## 3. 只给目标环境安装 uv 0.12.5

```bash
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | \
  env UV_INSTALL_DIR="${TARGET_ENV}/bin" sh

"${UV}" --version
```

预期输出：

```text
uv 0.12.5 (x86_64-unknown-linux-gnu)
```

后面的 uv 命令都从 `/tmp` 执行，防止 uv 自动读取 verl 根目录的
`pyproject.toml`/workspace 配置：

```bash
cd /tmp
```

## 4. 安装固定 Transformers 5.8.1 wheel

该 wheel 来自 Megatron-Bridge lockfile 记录的精确文件：

```bash
curl -L --fail --retry 5 \
  https://files.pythonhosted.org/packages/fc/b1/8be7e7ef0b5200491312201918b6125ef9c9df9dd0f0240ccef9ac824e6b/transformers-5.8.1-py3-none-any.whl \
  -o /tmp/transformers-5.8.1-py3-none-any.whl

printf '%s  %s\n' \
  5340fb95962162cdfdae5cc91d7f8fedd92ed75216c1154c5e1f590fcf56dd0e \
  /tmp/transformers-5.8.1-py3-none-any.whl | sha256sum --check

"${UV}" pip install \
  --python "${PYTHON}" \
  --no-deps \
  /tmp/transformers-5.8.1-py3-none-any.whl
```

这里使用 `--no-deps`，防止解析器替换克隆环境中的 Torch/CUDA 包。

## 5. 从兼容环境复制 TE 和 Bridge 导入所需组件

`slime_megatron` 使用 Torch 2.9.1，并已经包含可用的 Transformer Engine 2.10.0。
本机最终成功环境使用的是以下逐项复制命令，而不是 editable install：

```bash
for item in \
  transformer_engine \
  transformer_engine-2.10.0.dist-info \
  transformer_engine_cu12-2.10.0.dist-info \
  transformer_engine_torch-2.10.0.dist-info \
  modelopt \
  nvidia_modelopt-0.42.0.dist-info \
  onnx \
  onnx-1.21.0.dist-info \
  onnxscript \
  onnxscript-0.6.2.dist-info \
  onnx_ir \
  onnx_ir-0.2.0.dist-info \
  pulp \
  pulp-3.3.0.dist-info \
  omegaconf \
  omegaconf-2.3.0.dist-info \
  hydra \
  hydra_core-1.3.2.dist-info \
  antlr4 \
  antlr4_python3_runtime-4.9.3.dist-info
do
  test -e "${COMPAT_SITE}/${item}"
  cp -a "${COMPAT_SITE}/${item}" "${TARGET_SITE}/"
done
```

`apache-tvm-ffi==0.1.11` 和 `torch-c-dlpack-ext==0.1.5` 已存在于
`pt29_all_env`，因此随 Conda clone 一起进入目标环境，不需要额外复制。

## 6. 只在克隆环境中禁用不完整的 FlashAttention 3 探测

源环境中留下了 FA3 beta 的包名和 metadata，但模块不完整，会让 TE/Transformers
误判 FA3 可用。本机使用的是可恢复的重命名，不删除文件，也不影响 FA2：

```bash
mv "${TARGET_SITE}/flash_attn_3" \
  "${TARGET_SITE}/flash_attn_3.disabled"

mv "${TARGET_SITE}/flash_attn_3-3.0.0b1.dist-info" \
  "${TARGET_SITE}/flash_attn_3-3.0.0b1.dist-info.disabled"

mv "${TARGET_SITE}/flash_attn_interface.py" \
  "${TARGET_SITE}/flash_attn_interface.py.disabled"
```

原始环境中的同名文件保持不变：

```bash
find "${SOURCE_ENV}/lib/python3.12/site-packages" -maxdepth 1 \
  \( -name 'flash_attn_3' \
     -o -name 'flash_attn_3-*.dist-info' \
     -o -name 'flash_attn_interface.py' \) \
  -print
```

## 7. 安装 DSv4 contiguous CP 所需 CuTeDSL

直接安装 `nvidia-cudnn-frontend[cutedsl]` 会尝试把现有
`nvidia-cudnn-cu12==9.15.1.9` 降到 9.10。因此最终采用 `--no-deps` 精确安装，保留
原 cuDNN。大 wheel 使用清华镜像：

```bash
cd /tmp

"${UV}" pip install \
  --python "${PYTHON}" \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps \
  nvidia-cutlass-dsl-libs-base==4.5.0 \
  nvidia-cutlass-dsl-libs-cu13==4.5.0

"${UV}" pip install \
  --python "${PYTHON}" \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps \
  cuda-bindings==13.3.1 \
  cuda-core==1.0.1 \
  cuda-pathfinder==1.6.1 \
  cuda-python==13.3.1 \
  nvidia-cudnn-frontend==1.25.0 \
  nvidia-cutlass-dsl==4.5.0
```

检查 cuDNN 没有被降级：

```bash
"${PYTHON}" -c \
  "from importlib.metadata import version; print(version('nvidia-cudnn-cu12'))"
```

预期输出：`9.15.1.9`。

## 8. 固定基础三份第三方源码

创建目录：

```bash
mkdir -p "${PROJECT_DIR}/third_party"
```

Megatron-LM：

```bash
git clone --filter=blob:none \
  https://github.com/NVIDIA/Megatron-LM.git \
  "${PROJECT_DIR}/third_party/Megatron-LM"

git -C "${PROJECT_DIR}/third_party/Megatron-LM" checkout --detach \
  fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
```

Megatron-Bridge 当时先固定在 `/tmp`，再 clone 到项目中：

```bash
git clone --filter=blob:none \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git \
  /tmp/verl-dsv4-megatron-bridge

git -C /tmp/verl-dsv4-megatron-bridge checkout --detach \
  c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11

git clone /tmp/verl-dsv4-megatron-bridge \
  "${PROJECT_DIR}/third_party/Megatron-Bridge"

git -C "${PROJECT_DIR}/third_party/Megatron-Bridge" checkout --detach \
  c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
```

从 GitHub 直接 clone 到项目目录再 checkout 同一 commit 也是等价的；上面保留的是
本机实际形成当前 `origin=/tmp/verl-dsv4-megatron-bridge` 的步骤。

fast-hadamard-transform：

```bash
git clone \
  https://github.com/Dao-AILab/fast-hadamard-transform.git \
  "${PROJECT_DIR}/third_party/fast-hadamard-transform"

git -C "${PROJECT_DIR}/third_party/fast-hadamard-transform" checkout --detach \
  f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
```

## 9. 为当前 Torch/CUDA 编译 Hadamard CUDA extension

```bash
cd /tmp

FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
NVCC_THREADS=8 \
MAX_JOBS=4 \
"${UV}" pip install \
  --python "${PYTHON}" \
  --no-build-isolation \
  --no-deps \
  "${PROJECT_DIR}/third_party/fast-hadamard-transform"
```

验证真实 H200 kernel：

```bash
"${PYTHON}" -c \
  "import torch; from fast_hadamard_transform import hadamard_transform; x=torch.randn(8,256,device='cuda',dtype=torch.bfloat16); y=hadamard_transform(x,scale=256**-0.5); print('FHT_OK',y.shape,y.dtype,torch.isfinite(y).all().item())"
```

## 10. 设置源码加载路径并自检

```bash
export PYTHONPATH="${PROJECT_DIR}/third_party/Megatron-LM:${PROJECT_DIR}/third_party/Megatron-Bridge/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
"${PYTHON}" check_environment.py
```

关键输出应包含：

```text
torch=2.9.1
transformers=5.8.1
transformer-engine=2.10.0
nvidia-cudnn-cu12=9.15.1.9
megatron-core-source=fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
megatron-bridge-source=c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
fast-hadamard-transform-source=f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
gpu_count=8
gpu0=NVIDIA H200
cutedsl_available=True
DSV4_IMPORTS_OK
```

## 11. 运行三个 SFT case

三个入口都已自行设置上述 `PYTHONPATH`：

```bash
cd "${PROJECT_DIR}"
bash run_cp1.sh
bash run_cp2.sh
bash run_dynamic_cp.sh
```

最终成功标志：

```text
SFT_SMOKE_SUCCESS cp=1 layers=4 steps=2
SFT_SMOKE_SUCCESS cp=2 layers=4 steps=2
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1
```

Dynamic CP case 的默认输入长度为 64、128、256、512，每 rank 上限为 128；因此一轮中
分别调度为 CP1、CP1、CP2、CP4。它要求恰好 8 个进程。详见
[`DYNAMIC_CP_EXAMPLE.md`](./DYNAMIC_CP_EXAMPLE.md)。

## 12. 为 H200 安装 fused DSA（本机实测成功）

这一节是在前 11 节的 reference 环境之上增量执行。MCore 的 fused DSv4 训练实际由
三部分配合完成：cuDNN Frontend DSA 做 indexer 和 attention backward，FlashMLA 做
sparse-attention forward，MCore 自定义 autograd 负责保存/复用中间量并把两者串起来。

### 12.1 固定并裁剪 FlashMLA `nv_dev`

MCore 的 `uv.lock` 对应 FlashMLA 提交为：

```bash
cd "${PROJECT_DIR}"

git clone --branch nv_dev --recurse-submodules \
  https://github.com/deepseek-ai/FlashMLA.git \
  third_party/FlashMLA

git -C third_party/FlashMLA checkout --detach \
  b7643bd54521f563b839b98289b5cd048c062ba2

git -C third_party/FlashMLA submodule update --init --recursive

git -C third_party/FlashMLA rev-parse HEAD
git -C third_party/FlashMLA/csrc/cutlass rev-parse HEAD
```

预期分别为：

```text
b7643bd54521f563b839b98289b5cd048c062ba2
147f5673d0c1c3dcf66f78d677fd647e4a020219
```

本机 CUDA compiler 是 12.8。当前 FlashMLA 的 `FLASH_MLA_DISABLE_SM100=TRUE` 只删除
SM100 gencode flag，却仍编译 SM100 source，因此原始源码会在 Blackwell translation
unit 上失败。应用本项目保存的构建补丁：

```bash
git -C third_party/FlashMLA apply \
  "${PROJECT_DIR}/patches/flashmla-sm90-prefill-only.patch"
```

补丁只修改 `third_party/FlashMLA` 的 build/API surface：排除 SM100、decode 和 dense
translation units，仅保留 MCore DSv4 SFT 使用的 SM90 sparse-prefill forward。它没有
修改 verl、Megatron-Bridge 或 Megatron-Core，也不是一个通用 FlashMLA 推理构建。

用克隆环境内的 uv 编译安装：

```bash
cd /tmp

CUDA_HOME=/usr/local/cuda-12.8 \
FLASH_MLA_DISABLE_SM100=TRUE \
NVCC_THREADS=8 \
MAX_JOBS=4 \
"${UV}" pip install \
  --python "${PYTHON}" \
  --no-build-isolation \
  --no-deps \
  "${PROJECT_DIR}/third_party/FlashMLA"
```

预期安装版本是 `flash-mla==1.0.0+b7643bd`。

### 12.2 用 MCore lockfile 的 cuDNN Frontend 源码替换 PyPI 包

PyPI 的 `nvidia-cudnn-frontend==1.25.0` 虽然版本字符串相同，但当前发布内容的
`DSA.indexer_forward_wrapper()` 没有 Dynamic CP 所需的 `q_causal_offsets` 参数，运行
会报 `unexpected keyword argument 'q_causal_offsets'`。MCore `uv.lock` 固定的提交已经
包含该接口：

```bash
cd "${PROJECT_DIR}"

git clone https://github.com/NVIDIA/cudnn-frontend.git \
  third_party/cudnn-frontend

git -C third_party/cudnn-frontend checkout --detach \
  0a14b7181d129d30e7bad34b8c3ed0a0c995e23d

cd /tmp

"${UV}" pip install \
  --python "${PYTHON}" \
  --reinstall \
  --no-build-isolation \
  --no-deps \
  "${PROJECT_DIR}/third_party/cudnn-frontend"
```

不能仅以 `version('nvidia-cudnn-frontend') == '1.25.0'` 判断是否正确，必须检查函数签名。
下面的项目自检会打印 `cudnn_dsa_q_causal_offsets=True`。

### 12.3 先验证 H200 forward/backward kernel 对

```bash
export PYTHONPATH="${PROJECT_DIR}/third_party/Megatron-LM:${PROJECT_DIR}/third_party/Megatron-Bridge/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
"${PYTHON}" check_environment.py --require-fused
"${PYTHON}" verify_fused_dsa_sm90.py
```

第二个命令直接执行与 MCore 相同的 kernel 配对，预期关键输出：

```text
device=NVIDIA H200 capability=(9, 0)
FLASHMLA_SM90_FWD_OK out=[16, 64, 512] lse=[16, 64]
CUDNN_DSA_SM90_BWD_OK dq=[16, 64, 512] dkv=[64, 512] d_sink=[64]
FUSED_DSA_SM90_VERIFY_SUCCESS
```

### 12.4 跑通完整 fused SFT

```bash
cd "${PROJECT_DIR}"
bash run_fused_cp1.sh
bash run_fused_dynamic_cp.sh
```

默认使用 2 层、1 个真实 optimizer step、`dsa_indexer_loss_coeff=0.001`。fused kernel
要求生产形状，本项目在 `--fused-dsa` 下将 attention geometry 恢复为 64 个 query heads、
head dim 512、indexer 64×128、top-k 512；hidden size、层数和 expert 数仍保持 smoke-test
规模。不能把 reference case 的 `head_dim=256/topk=32` 原样拿给 fused kernel。

本机 8×H200 实测关键日志：

```text
# CP=1
MODEL_READY layers=2 cp=1 ep=8 fused_dsa=True indexer_loss_coeff=0.001 local_params=95,924,593
STEP_OK step=0 loss=9.679355 grad_norm=6.337935 update=True
SFT_SMOKE_SUCCESS cp=1 layers=2 steps=1 fused_dsa=True

# Dynamic CP: 同一轮执行 CP4/CP2/CP1/CP1
DYNAMIC_MODEL_READY layers=2 static_cp=1 runtime_cp_sizes=[1,2,4] ep=8 fused_dsa=True indexer_loss_coeff=0.001 local_params=95,924,593
DYNAMIC_STEP_OK step=0 loss=9.643462 global_response_tokens=646 grad_norm=7.047264 update=True
DYNAMIC_CP_SFT_SUCCESS layers=2 steps=1 fused_dsa=True
```

## 安装边界总结

```text
pt29_all_env
  └─ 只读源环境，未修改

dsv4_sft_bridge
  ├─ Conda clone
  ├─ Transformers / TE / CuTeDSL / Hadamard runtime
  ├─ cuDNN Frontend（MCore lockfile 源码提交）
  ├─ FlashMLA（本地编译的 SM90 sparse-prefill-only extension）
  └─ 不安装 Megatron-LM、Megatron-Bridge、verl

dsv4_sft_standalone/third_party
  ├─ Megatron-LM 固定源码
  ├─ Megatron-Bridge 固定源码
  ├─ fast-hadamard-transform 固定源码
  ├─ FlashMLA 固定源码（只在这里应用构建裁剪 patch）
  └─ cuDNN Frontend 固定源码
```
