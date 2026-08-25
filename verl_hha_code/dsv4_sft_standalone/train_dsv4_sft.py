#!/usr/bin/env python3
"""Standalone DeepSeek-V4 random-init SFT smoke test on Megatron Bridge/Core.

This file intentionally does not import verl.  It uses Megatron Bridge to translate
an in-memory Hugging Face DeepSeek-V4 config, Megatron-Core to construct/train the
distributed model, and a small synthetic prompt/response batch to exercise SFT loss.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class PackedBatch:
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor
    packed_seq_params: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--context-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expert-model-parallel-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print-structure", action="store_true")
    parser.add_argument(
        "--fused-dsa",
        action="store_true",
        help="Use production DSv4 attention/indexer geometry and fused DSA kernels.",
    )
    parser.add_argument(
        "--dsa-indexer-loss-coeff",
        type=float,
        default=0.0,
        help="Auxiliary indexer KL coefficient; use a positive value to exercise indexer backward.",
    )
    return parser.parse_args()


def build_hf_config(args: argparse.Namespace):
    """Build only a DSv4 config; no HF checkpoint or tokenizer is accessed."""
    from transformers import DeepseekV4Config

    ratios = [0] + [4] * (args.num_layers - 1)
    # FlashMLA/cuDNN DSA kernels accept the production attention/indexer
    # geometry, not the aggressively shrunken reference-only geometry.  Keep
    # hidden/MoE/layer counts small while restoring those kernel-facing dims.
    if args.fused_dsa:
        attention_geometry = {
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "o_groups": 8,
            "sliding_window": 128,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
        }
    else:
        attention_geometry = {
            "head_dim": 256,
            "qk_rope_head_dim": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "o_groups": 4,
            "sliding_window": 64,
            "index_n_heads": 4,
            "index_head_dim": 32,
            "index_topk": 32,
        }

    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=4096,
        moe_intermediate_size=512,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=4,
        num_hidden_layers=args.num_layers,
        num_nextn_predict_layers=0,
        q_lora_rank=256,
        o_lora_rank=256,
        compress_ratios=ratios,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        rope_theta=10000,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        vocab_size=12800,
        dtype="bfloat16",
        tie_word_embeddings=False,
        **attention_geometry,
    )
    # Bridge versions around the DSv4 PR still consult torch_dtype.
    config.torch_dtype = torch.bfloat16
    return config


def build_model(args: argparse.Namespace):
    """Translate HF config with Bridge and create the real MCore GPU model."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.utils.config_utils import create_ddp_config
    from megatron.core.transformer.enums import AttnBackend
    
    # Preserve the optional remote-debug workflow without blocking normal runs.
    if os.environ.get("DSV4_DEBUGPY_CONNECT"):
        import debugpy

        debugpy.connect((os.environ.get("DSV4_DEBUGPY_HOST", "10.103.23.29"), 5680))

    hf_config = build_hf_config(args)
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Keep unrelated optional fusions conservative.  Fused DSA is enabled only
    # by the explicit CLI flag after restoring its kernel-facing geometry.
    overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "expert_tensor_parallel_size": 1,
        "context_parallel_size": args.context_parallel_size,
        # MCore's DSv4 contiguous-CP path consumes THD packed sequences and
        # requires the same scheduler flag used by its DSv4 CP unit tests.
        "sequence_packing_scheduler": (
            "dp_balanced" if args.context_parallel_size > 1 else None
        ),
        "sequence_parallel": False,
        "seq_length": args.sequence_length,
        "variable_seq_lengths": True,
        "cp_partition_mode": "contiguous",
        "attention_backend": AttnBackend.auto,
        "apply_dsa_kernel_fusion": args.fused_dsa,
        "dsa_indexer_loss_coeff": args.dsa_indexer_loss_coeff,
        "dsa_indexer_use_sparse_loss": args.fused_dsa,
        "use_fused_mhc": False,
        "moe_grouped_gemm": False,
        "moe_shared_expert_overlap": False,
        "moe_router_load_balancing_type": "none",
        "moe_router_enable_expert_bias": False,
        "moe_permute_fusion": False,
        "gradient_accumulation_fusion": False,
        "cross_entropy_loss_fusion": False,
        "apply_rope_fusion": False,
        "persist_layer_norm": False,
    }
    provider.apply_overrides_and_finalize(dtype=torch.bfloat16, overrides=overrides)

    ddp_config = create_ddp_config(
        wrap_with_ddp=True,
        use_distributed_optimizer=False,
        overrides={
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
            "check_for_nan_in_grad": True,
        },
    )
    model = provider.provide_distributed_model(
        ddp_config=ddp_config,
        wrap_with_ddp=True,
        bf16=True,
        fp16=False,
        data_parallel_random_init=False,
        use_cpu_initialization=False,
    )
    return bridge, provider, model


def build_optimizer(model: list[torch.nn.Module], args: argparse.Namespace):
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    config = OptimizerConfig(
        optimizer="adam",
        lr=args.learning_rate,
        min_lr=args.learning_rate,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
        clip_grad=1.0,
    )
    return get_megatron_optimizer(config, model)


def _synthetic_sequences(args: argparse.Namespace, step: int) -> tuple[list[torch.Tensor], ...]:
    """Create prompt/response next-token targets without a dataset dependency."""
    from megatron.core import parallel_state

    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1009 * dp_rank + step)

    tokens, labels, masks = [], [], []
    prompt_len = args.sequence_length // 3
    for _ in range(args.micro_batch_size):
        raw = torch.randint(4, 12800, (args.sequence_length + 1,), generator=generator)
        tokens.append(raw[:-1])
        labels.append(raw[1:])
        # A target at position i predicts raw[i+1]. Mask prompt-token targets.
        masks.append((torch.arange(args.sequence_length) + 1 >= prompt_len).float())
    return tokens, labels, masks


def pack_for_contiguous_cp(args: argparse.Namespace, step: int, device: torch.device) -> PackedBatch:
    """Pack equal/variable sequences into global THD rows, then take a contiguous CP slice.

    This is a small standalone input adapter, not an import or copy of verl.  The actual
    CP slicing operation is Megatron-Core's get_cp_slice_for_thd().
    """
    from megatron.core import parallel_state
    from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd
    from megatron.core.packed_seq_params import PackedSeqParams

    seq_tokens, seq_labels, seq_masks = _synthetic_sequences(args, step)
    cp_group = parallel_state.get_context_parallel_group()
    cp_size = cp_group.size()

    original_lengths = [x.numel() for x in seq_tokens]
    padded_lengths = [((length + cp_size - 1) // cp_size) * cp_size for length in original_lengths]
    cu_padded = [0]
    for length in padded_lengths:
        cu_padded.append(cu_padded[-1] + length)

    total = cu_padded[-1]
    batch = {
        "tokens": torch.zeros(total, dtype=torch.long, device=device),
        "labels": torch.zeros(total, dtype=torch.long, device=device),
        "loss_mask": torch.zeros(total, dtype=torch.float32, device=device),
        "position_ids": torch.zeros(total, dtype=torch.long, device=device),
        "padding_mask": torch.ones(total, dtype=torch.bool, device=device),
        "cu_seqlens_padded": torch.tensor(cu_padded, dtype=torch.int32, device=device),
    }
    for index, length in enumerate(original_lengths):
        start = cu_padded[index]
        end = start + length
        batch["tokens"][start:end] = seq_tokens[index].to(device)
        batch["labels"][start:end] = seq_labels[index].to(device)
        batch["loss_mask"][start:end] = seq_masks[index].to(device)
        batch["position_ids"][start:end] = torch.arange(length, device=device)
        batch["padding_mask"][start:end] = False

    get_cp_slice_for_thd(
        batch,
        cp_group,
        keys=("tokens", "labels", "loss_mask", "position_ids", "padding_mask"),
        cp_partition_mode="contiguous",
    )
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=batch["cu_seqlens_padded"],
        cu_seqlens_kv=batch["cu_seqlens_padded"],
        cu_seqlens_q_padded=batch["cu_seqlens_padded"],
        cu_seqlens_kv_padded=batch["cu_seqlens_padded"],
        max_seqlen_q=max(padded_lengths),
        max_seqlen_kv=max(padded_lengths),
        cp_partition_mode="contiguous",
    )
    return PackedBatch(
        tokens=batch["tokens"].view(1, -1).contiguous(),
        labels=batch["labels"].view(1, -1).contiguous(),
        loss_mask=batch["loss_mask"].view(1, -1).contiguous(),
        position_ids=batch["position_ids"].view(1, -1).contiguous(),
        padding_mask=batch["padding_mask"].view(1, -1).contiguous(),
        packed_seq_params=packed,
    )


def print_model_summary(model: list[torch.nn.Module], args: argparse.Namespace) -> None:
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    core_model = unwrap_model(model[0])
    local_params = sum(parameter.numel() for parameter in core_model.parameters())
    if dist.get_rank() == 0:
        print(
            f"MODEL_READY layers={args.num_layers} cp={args.context_parallel_size} "
            f"ep={args.expert_model_parallel_size} fused_dsa={args.fused_dsa} "
            f"indexer_loss_coeff={args.dsa_indexer_loss_coeff} local_params={local_params:,}",
            flush=True,
        )
        if args.print_structure:
            print(core_model, flush=True)
    dist.barrier()
    if parallel_state.get_expert_model_parallel_rank() == 0 and dist.get_rank() == 0:
        print("Random initialization only: bridge.load_hf_weights() was not called.", flush=True)


def train(args: argparse.Namespace) -> None:
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    world_size = dist.get_world_size()
    if world_size != 8:
        raise ValueError(f"These two supplied cases expect exactly 8 processes, got {world_size}.")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1")
    if args.micro_batch_size % args.context_parallel_size != 0:
        raise ValueError("Use a micro-batch size divisible by CP so each CP rank gets equal SFT work.")

    _bridge, provider, model = build_model(args)
    optimizer = build_optimizer(model, args)
    print_model_summary(model, args)

    for step in range(args.train_steps):
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        batch = pack_for_contiguous_cp(args, step, device)
        token_losses = model[0](
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels,
            packed_seq_params=batch.packed_seq_params,
            padding_mask=batch.padding_mask,
        )
        valid = batch.loss_mask.sum()
        loss = (token_losses.float().view(-1) * batch.loss_mask.view(-1)).sum() / valid.clamp_min(1.0)
        optimizer.scale_loss(loss).backward()
        finalize_model_grads(model)
        update_successful, grad_norm, _ = optimizer.step()

        log_sum = torch.stack(
            ((loss.detach() * valid).float(), valid.detach().float())
        )
        dist.all_reduce(log_sum, op=dist.ReduceOp.SUM)
        global_loss = (log_sum[0] / log_sum[1].clamp_min(1.0)).item()
        if dist.get_rank() == 0:
            grad_value = float(grad_norm) if grad_norm is not None else float("nan")
            print(
                f"STEP_OK step={step} loss={global_loss:.6f} "
                f"grad_norm={grad_value:.6f} update={bool(update_successful)}",
                flush=True,
            )
        if not update_successful or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid optimizer result at step {step}")

    if dist.get_rank() == 0:
        print(
            f"SFT_SMOKE_SUCCESS cp={args.context_parallel_size} layers={args.num_layers} "
            f"steps={args.train_steps} fused_dsa={args.fused_dsa}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
