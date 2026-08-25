#!/usr/bin/env python3
"""Standalone DSv4 SFT example with verl-style Megatron-Core Dynamic CP.

The batch is replicated across the DPxCP plane.  MCore's
DefaultDynamicCPScheduler assigns each variable-length sequence a runtime CP
group, and PackedSeqParams carries that group into DSv4 attention.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from train_dsv4_sft import build_hf_config, build_optimizer


@dataclass
class DynamicPackedBatch:
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor
    packed_seq_params: object
    sample_ids: list[int]
    group_ranks: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-seqlen-per-rank", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print-structure", action="store_true")
    parser.add_argument(
        "--fused-dsa",
        action="store_true",
        help="Use production DSv4 attention/indexer geometry and fused DSA kernels.",
    )
    parser.add_argument("--dsa-indexer-loss-coeff", type=float, default=0.0)
    return parser.parse_args()


def initialize_dynamic_parallel_state(args: argparse.Namespace) -> None:
    """Create MCore's reusable CP groups of size 1/2/4/8.

    The static model topology has CP=1 and DP=8.  At runtime, ranks from that
    DPxCP plane are regrouped per microbatch by PackedSeqParams.cp_group.
    """
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if parallel_state.is_initialized():
        raise RuntimeError("This example expects an uninitialized MCore parallel state.")
    # [Megatron-Core] Build the static topology and reusable dynamic DPxCP groups.
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(args.seed)


def build_dynamic_model(args: argparse.Namespace):
    """Build DSv4 with the same DCP-facing provider settings used by verl."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.utils.config_utils import create_ddp_config
    from megatron.core.transformer.enums import AttnBackend

    # build_hf_config only consumes num_layers.  seq_length below describes the
    # largest global example; max_seqlen_per_dp_cp_rank is the local DCP budget.
    args.sequence_length = 512
    hf_config = build_hf_config(args)
    # [Megatron-Bridge] Select and populate the DSv4 model provider.
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)

    # [Standalone, mirrors verl] These are the DCP-facing TransformerConfig overrides.
    overrides = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "context_parallel_size": 1,
        # verl initializes dynamic process groups separately, then keeps the
        # TransformerConfig flag false and passes the runtime group in THD metadata.
        "dynamic_context_parallel": False,
        "sequence_packing_scheduler": "default_dynamic_cp",
        "max_seqlen_per_dp_cp_rank": args.max_seqlen_per_rank,
        "calculate_per_token_loss": True,
        "sequence_parallel": False,
        "seq_length": 512,
        "variable_seq_lengths": True,
        "cp_partition_mode": "contiguous",
        "attention_backend": AttnBackend.flash,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_load_balancing_type": "none",
        # Keep unrelated optional fusions conservative.  Fused DSA is enabled
        # only by the explicit CLI flag and can be tested with Dynamic CP.
        "apply_dsa_kernel_fusion": args.fused_dsa,
        "dsa_indexer_loss_coeff": args.dsa_indexer_loss_coeff,
        "dsa_indexer_use_sparse_loss": args.fused_dsa,
        "use_fused_mhc": False,
        "moe_grouped_gemm": False,
        "moe_shared_expert_overlap": False,
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
    return provider, model


def get_schedule(args: argparse.Namespace) -> list[list[list[int]]]:
    """Return MCore's rank -> sample-id assignment for the replicated batch."""
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    sequence_lengths = [64, 128, 256, 512]
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=args.max_seqlen_per_rank,
        cp_size=1,
        dp_size=dist.get_world_size(),
        microbatch_group_size_per_vp_stage=None,
        min_cp_size=1,
    )
    # [Megatron-Core] The scheduling algorithm itself is not reimplemented here.
    assignments = scheduler.get_groups_and_subsamples(list(enumerate(sequence_lengths)))
    if not assignments:
        raise RuntimeError("MCore Dynamic CP scheduler returned no microbatches.")
    return assignments


def _unique_rank_groups(rank_assignments: list[list[int]]) -> list[dict]:
    sequence_lengths = [64, 128, 256, 512]
    groups = []
    seen = set()
    for rank, sample_ids in enumerate(rank_assignments):
        key = tuple(sample_ids)
        if key in seen:
            continue
        seen.add(key)
        ranks = [peer for peer, peer_ids in enumerate(rank_assignments) if peer_ids == sample_ids]
        groups.append(
            {
                "ranks": ranks,
                "cp_size": len(ranks),
                "sample_ids": list(sample_ids),
                "sequence_lengths": [sequence_lengths[index] for index in sample_ids],
            }
        )
    return groups


def print_schedule(assignments: list[list[list[int]]]) -> None:
    if dist.get_rank() != 0:
        return
    print(
        "DCP_EXPLANATION max_tokens_per_rank=128 "
        "rule=ceil(sequence_length/max_tokens_per_rank)->minimum_CP",
        flush=True,
    )
    for microbatch_id, rank_assignments in enumerate(assignments):
        print(
            f"DCP_PLAN microbatch={microbatch_id} groups={_unique_rank_groups(rank_assignments)}",
            flush=True,
        )


def _make_sample(sample_id: int, length: int, step: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * step + 7919 * sample_id)
    raw = torch.randint(4, 12800, (length + 1,), generator=generator)
    tokens = raw[:-1]
    labels = raw[1:]
    prompt_length = length // 3
    loss_mask = (torch.arange(length) + 1 >= prompt_length).float()
    return tokens, labels, loss_mask


def pack_dynamic_microbatch(
    args: argparse.Namespace,
    rank_assignments: list[list[int]],
    step: int,
    device: torch.device,
) -> DynamicPackedBatch:
    """Build verl-style contiguous THD rows for this runtime CP group."""
    from megatron.core import parallel_state
    from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd
    from megatron.core.packed_seq_params import PackedSeqParams

    sequence_lengths = [64, 128, 256, 512]
    dcp_rank = parallel_state.get_data_parallel_group(with_context_parallel=True).rank()
    sample_ids = list(rank_assignments[dcp_rank])
    group_ranks = [rank for rank, ids in enumerate(rank_assignments) if ids == sample_ids]
    local_cp_size = len(group_ranks)
    if group_ranks != list(range(group_ranks[0], group_ranks[0] + local_cp_size)):
        raise RuntimeError(f"Dynamic CP ranks must be contiguous, got {group_ranks}")
    if group_ranks[0] % local_cp_size != 0:
        raise RuntimeError(f"Dynamic CP group is not naturally aligned: {group_ranks}")

    cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
    if cp_group.size() != local_cp_size:
        raise RuntimeError("Resolved MCore dynamic CP group has the wrong size.")

    samples = [
        _make_sample(sample_id, sequence_lengths[sample_id], step, args.seed)
        for sample_id in sample_ids
    ]
    original_lengths = [item[0].numel() for item in samples]
    padded_lengths = [math.ceil(length / local_cp_size) * local_cp_size for length in original_lengths]
    cu_padded = [0]
    for length in padded_lengths:
        cu_padded.append(cu_padded[-1] + length)

    total_rows = cu_padded[-1]
    batch = {
        "tokens": torch.zeros(total_rows, dtype=torch.long, device=device),
        "labels": torch.zeros(total_rows, dtype=torch.long, device=device),
        "loss_mask": torch.zeros(total_rows, dtype=torch.float32, device=device),
        "position_ids": torch.zeros(total_rows, dtype=torch.long, device=device),
        "padding_mask": torch.ones(total_rows, dtype=torch.bool, device=device),
        "cu_seqlens_padded": torch.tensor(cu_padded, dtype=torch.int32, device=device),
    }
    for index, (tokens, labels, loss_mask) in enumerate(samples):
        start = cu_padded[index]
        end = start + tokens.numel()
        batch["tokens"][start:end] = tokens.to(device)
        batch["labels"][start:end] = labels.to(device)
        batch["loss_mask"][start:end] = loss_mask.to(device)
        batch["position_ids"][start:end] = torch.arange(tokens.numel(), device=device)
        batch["padding_mask"][start:end] = False

    # [Megatron-Core] Physically slice the packed global rows across this runtime group.
    get_cp_slice_for_thd(
        batch,
        cp_group,
        keys=("tokens", "labels", "loss_mask", "position_ids", "padding_mask"),
        cp_partition_mode="contiguous",
    )
    # [Megatron-Core metadata] DSv4 attention reads the runtime group from here.
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=batch["cu_seqlens_padded"],
        cu_seqlens_kv=batch["cu_seqlens_padded"],
        cu_seqlens_q_padded=batch["cu_seqlens_padded"],
        cu_seqlens_kv_padded=batch["cu_seqlens_padded"],
        max_seqlen_q=max(padded_lengths),
        max_seqlen_kv=max(padded_lengths),
        local_cp_size=local_cp_size,
        cp_group=cp_group,
        cp_partition_mode="contiguous",
    )
    return DynamicPackedBatch(
        tokens=batch["tokens"].view(1, -1).contiguous(),
        labels=batch["labels"].view(1, -1).contiguous(),
        loss_mask=batch["loss_mask"].view(1, -1).contiguous(),
        position_ids=batch["position_ids"].view(1, -1).contiguous(),
        padding_mask=batch["padding_mask"].view(1, -1).contiguous(),
        packed_seq_params=packed_seq_params,
        sample_ids=sample_ids,
        group_ranks=group_ranks,
    )


def train(args: argparse.Namespace) -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.utils import unwrap_model

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    if dist.get_world_size() != 8:
        raise ValueError("The supplied Dynamic CP demonstration expects exactly 8 ranks.")
    if args.max_seqlen_per_rank != 128:
        raise ValueError("Keep --max-seqlen-per-rank=128 for the documented [4,2,1,1] plan.")
    if args.num_layers < 2:
        raise ValueError("Use at least two layers so the compressed DSv4 layer is exercised.")

    initialize_dynamic_parallel_state(args)
    provider, model = build_dynamic_model(args)
    optimizer = build_optimizer(model, args)
    assignments = get_schedule(args)
    print_schedule(assignments)

    if dist.get_rank() == 0:
        core_model = unwrap_model(model[0])
        local_params = sum(parameter.numel() for parameter in core_model.parameters())
        print(
            f"DYNAMIC_MODEL_READY layers={args.num_layers} static_cp=1 "
            f"runtime_cp_sizes=[1,2,4] ep=8 fused_dsa={args.fused_dsa} "
            f"indexer_loss_coeff={args.dsa_indexer_loss_coeff} local_params={local_params:,}",
            flush=True,
        )
        print("Random initialization only: bridge.load_hf_weights() was not called.", flush=True)
        if args.print_structure:
            print(core_model, flush=True)
    dist.barrier()

    for step in range(args.train_steps):
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        local_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_num_tokens = torch.zeros((), dtype=torch.int64, device=device)

        for rank_assignments in assignments:
            batch = pack_dynamic_microbatch(args, rank_assignments, step, device)
            if step == 0:
                print(
                    f"DCP_RANK rank={dist.get_rank()} samples={batch.sample_ids} "
                    f"group={batch.group_ranks} local_rows={batch.tokens.numel()}",
                    flush=True,
                )
            token_losses = model[0](
                input_ids=batch.tokens,
                position_ids=batch.position_ids,
                attention_mask=None,
                labels=batch.labels,
                packed_seq_params=batch.packed_seq_params,
                padding_mask=batch.padding_mask,
            )
            loss_sum = (
                token_losses.float().view(-1) * batch.loss_mask.view(-1)
            ).sum()
            valid_tokens = batch.loss_mask.sum().to(torch.int64)
            optimizer.scale_loss(loss_sum).backward()
            local_loss_sum += loss_sum.detach()
            local_num_tokens += valid_tokens

        # This is verl/MCore's per-token regime: DDP sums gradients across the
        # full DPxCP plane and finalize_model_grads divides by the global token count.
        # finalize_model_grads all-reduces num_tokens in place.  Keep the local
        # value intact for the independent loss/diagnostic reduction below.
        num_tokens_for_grad_finalize = local_num_tokens.clone()
        # [Megatron-Core] Finish DDP synchronization and divide by global valid tokens.
        finalize_model_grads(model, num_tokens=num_tokens_for_grad_finalize)
        update_successful, grad_norm, _ = optimizer.step()

        totals = torch.stack((local_loss_sum, local_num_tokens.float()))
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        global_loss = (totals[0] / totals[1].clamp_min(1.0)).item()
        if dist.get_rank() == 0:
            print(
                f"DYNAMIC_STEP_OK step={step} loss={global_loss:.6f} "
                f"global_response_tokens={int(totals[1].item())} "
                f"grad_norm={float(grad_norm):.6f} update={bool(update_successful)}",
                flush=True,
            )
        if not update_successful or not torch.isfinite(totals[0]):
            raise RuntimeError(f"Invalid Dynamic CP optimizer result at step {step}")

    if dist.get_rank() == 0:
        print(
            f"DYNAMIC_CP_SFT_SUCCESS layers={args.num_layers} steps={args.train_steps} "
            f"fused_dsa={args.fused_dsa}",
            flush=True,
        )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
