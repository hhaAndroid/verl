#!/usr/bin/env python3
import argparse
import inspect
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description="Check the standalone DSv4 environment.")
parser.add_argument(
    "--require-fused",
    action="store_true",
    help="Fail unless the FlashMLA/cuDNN DSA source and API requirements are present.",
)
args = parser.parse_args()

for package in (
    "torch",
    "transformers",
    "transformer-engine",
    "transformer-engine-torch",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "fast-hadamard-transform",
    "flash-mla",
):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=MISSING")

root = Path(__file__).resolve().parent
for name, path in (
    ("megatron-core-source", root / "third_party" / "Megatron-LM"),
    ("megatron-bridge-source", root / "third_party" / "Megatron-Bridge"),
    ("fast-hadamard-transform-source", root / "third_party" / "fast-hadamard-transform"),
    ("flash-mla-source", root / "third_party" / "FlashMLA"),
    ("flash-mla-cutlass-source", root / "third_party" / "FlashMLA" / "csrc" / "cutlass"),
    ("cudnn-frontend-source", root / "third_party" / "cudnn-frontend"),
):
    if not path.exists():
        print(f"{name}=MISSING")
        if args.require_fused and name in {
            "flash-mla-source",
            "flash-mla-cutlass-source",
            "cudnn-frontend-source",
        }:
            raise RuntimeError(f"Required fused source checkout is missing: {path}")
        continue
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    print(f"{name}={commit}")

print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()}")
    print(f"gpu0={torch.cuda.get_device_name(0)}")

from megatron.bridge import AutoBridge  # noqa: E402,F401
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402,F401
from megatron.core.transformer.experimental_attention_variant import (  # noqa: E402
    csa_cp_layout_kernels,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (  # noqa: E402
    DSv4HybridAttention,
)
from fast_hadamard_transform import hadamard_transform  # noqa: E402,F401
print(f"cutedsl_available={csa_cp_layout_kernels._CUTE_AVAILABLE}")
try:
    from cudnn import DSA  # noqa: E402
    from flash_mla import flash_mla_sparse_fwd  # noqa: E402,F401

    indexer_signature = inspect.signature(DSA.indexer_forward_wrapper)
    has_q_causal_offsets = "q_causal_offsets" in indexer_signature.parameters
    print("flash_mla_available=True")
    print(f"cudnn_dsa_q_causal_offsets={has_q_causal_offsets}")
    if args.require_fused and not has_q_causal_offsets:
        raise RuntimeError("cuDNN Frontend DSA is missing q_causal_offsets")
except ImportError as error:
    print(f"flash_mla_available=False ({error})")
    if args.require_fused:
        raise
print("DSV4_IMPORTS_OK")
