#!/usr/bin/env python3
"""Profile one real DSv4 SFT forward without importing verl.

This diagnostic is intentionally separate from the smoke-test trainers.  It
registers read-only hooks on selected modules and profiles rank 0 while every
rank participates in the real EP/CP collectives.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from train_dsv4_dynamic_cp_sft import (
    build_dynamic_model,
    get_schedule,
    initialize_dynamic_parallel_state,
    pack_dynamic_microbatch,
)
from train_dsv4_sft import build_model, pack_for_contiguous_cp


@dataclass
class HookRecord:
    order: int
    name: str
    module_type: str
    inputs: str
    outputs: str
    start: torch.cuda.Event
    end: torch.cuda.Event
    memory_before: int
    memory_after: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cp1", "dynamic"), default="cp1")
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args()


def _tensor_summary(value: Any, depth: int = 0) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)}:{str(value.dtype).removeprefix('torch.')}"
    if value is None:
        return "None"
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        items = ", ".join(_tensor_summary(item, depth + 1) for item in value[:8])
        suffix = ", ..." if len(value) > 8 else ""
        return f"{type(value).__name__}({items}{suffix})"
    if isinstance(value, dict):
        items = ", ".join(
            f"{key}={_tensor_summary(item, depth + 1)}" for key, item in list(value.items())[:12]
        )
        suffix = ", ..." if len(value) > 12 else ""
        return f"dict({items}{suffix})"
    return type(value).__name__


def _should_trace(name: str) -> bool:
    if name in {"embedding", "embedding.word_embeddings", "decoder.final_layernorm", "output_layer"}:
        return True
    if not name.startswith("decoder.layers."):
        return False
    parts = name.split(".")
    if len(parts) == 3:
        return True
    suffixes = (
        "input_layernorm",
        "self_attention_hyper_connection",
        "self_attention",
        "self_attention.linear_q_down_proj",
        "self_attention.q_layernorm",
        "self_attention.linear_q_up_proj",
        "self_attention.linear_kv_proj",
        "self_attention.kv_layernorm",
        "self_attention.core_attention",
        "self_attention.core_attention.compressor",
        "self_attention.core_attention.indexer",
        "self_attention.core_attention.indexer.compressor",
        "self_attention.linear_proj",
        "mlp_hyper_connection",
        "pre_mlp_layernorm",
        "mlp",
        "mlp.router",
        "mlp.experts",
        "mlp.shared_experts",
    )
    layer_suffix = ".".join(parts[3:])
    return layer_suffix in suffixes


def register_trace_hooks(core_model: torch.nn.Module) -> tuple[list, list[HookRecord]]:
    handles = []
    records: list[HookRecord] = []
    pending: dict[str, tuple[torch.cuda.Event, int, str]] = {}

    for name, module in core_model.named_modules():
        if not _should_trace(name):
            continue

        def pre_hook(_module, args, kwargs, *, module_name=name):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            pending[module_name] = (
                start,
                torch.cuda.memory_allocated(),
                _tensor_summary((args, kwargs)),
            )

        def post_hook(_module, args, kwargs, output, *, module_name=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start, memory_before, inputs = pending.pop(module_name)
            records.append(
                HookRecord(
                    order=len(records),
                    name=module_name,
                    module_type=type(_module).__name__,
                    inputs=inputs,
                    outputs=_tensor_summary(output),
                    start=start,
                    end=end,
                    memory_before=memory_before,
                    memory_after=torch.cuda.memory_allocated(),
                )
            )

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    return handles, records


def print_provider_snapshot(provider) -> None:
    fields = (
        "experimental_attention_variant",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "v_head_dim",
        "qk_pos_emb_head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "csa_compress_ratios",
        "csa_window_size",
        "dsa_indexer_n_heads",
        "dsa_indexer_head_dim",
        "dsa_indexer_topk",
        "apply_dsa_kernel_fusion",
        "apply_rope_fusion",
        "enable_hyper_connections",
        "use_fused_mhc",
        "num_residual_streams",
        "ffn_hidden_size",
        "moe_ffn_hidden_size",
        "moe_shared_expert_intermediate_size",
        "num_moe_experts",
        "moe_router_topk",
        "moe_n_hash_layers",
        "moe_token_dispatcher_type",
        "moe_grouped_gemm",
        "moe_shared_expert_overlap",
        "moe_permute_fusion",
        "context_parallel_size",
        "cp_partition_mode",
    )
    snapshot = {field: getattr(provider, field, None) for field in fields}
    print(f"TRACE_PROVIDER {snapshot}", flush=True)


def print_module_inventory(core_model: torch.nn.Module) -> None:
    print("TRACE_MODULE_INVENTORY_BEGIN", flush=True)
    for name, module in core_model.named_modules():
        if _should_trace(name):
            local_parameters = sum(
                parameter.numel() for parameter in module.parameters(recurse=False)
            )
            direct_parameter_shapes = {
                parameter_name: tuple(parameter.shape)
                for parameter_name, parameter in module.named_parameters(recurse=False)
            }
            print(
                f"TRACE_MODULE_DEF name={name} type={type(module).__name__} "
                f"python_type={type(module).__module__}.{type(module).__qualname__} "
                f"direct_params={local_parameters} direct_parameter_shapes={direct_parameter_shapes}",
                flush=True,
            )
    print("TRACE_MODULE_INVENTORY_END", flush=True)


def print_profiler_summary(profiler) -> None:
    print("TRACE_PROFILER_CUDA_TOP_BEGIN", flush=True)
    try:
        print(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=35, max_name_column_width=100
            ),
            flush=True,
        )
    except KeyError:
        print(
            profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=35, max_name_column_width=100
            ),
            flush=True,
        )
    print("TRACE_PROFILER_CUDA_TOP_END", flush=True)

    communication = []
    for event in profiler.key_averages():
        key = event.key.lower()
        if any(marker in key for marker in ("nccl", "all_to_all", "alltoall", "all_gather", "allgather")):
            communication.append(
                (
                    event.key,
                    event.count,
                    float(getattr(event, "self_cuda_time_total", 0.0)),
                    float(getattr(event, "self_device_time_total", 0.0)),
                    float(getattr(event, "self_cpu_time_total", 0.0)),
                )
            )
    print(f"TRACE_COMM_EVENTS {communication}", flush=True)

    noteworthy = []
    markers = (
        "hadamard",
        "dsv4_cp",
        "cutlass",
        "gemm",
        "softmax",
        "flash_mla",
        "compiledfxgraph",
    )
    for event in profiler.key_averages():
        key = event.key.lower()
        if any(marker in key for marker in markers):
            noteworthy.append(
                (
                    event.key,
                    event.count,
                    float(getattr(event, "self_cuda_time_total", 0.0)),
                    float(getattr(event, "self_device_time_total", 0.0)),
                )
            )
    noteworthy.sort(key=lambda item: max(item[2], item[3]), reverse=True)
    print(f"TRACE_NOTEWORTHY_EVENTS {noteworthy[:60]}", flush=True)


def main() -> None:
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model
    from torch.profiler import ProfilerActivity, profile, record_function

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    if dist.get_world_size() != 8:
        raise ValueError("The forward trace expects exactly 8 ranks.")

    if args.mode == "cp1":
        build_args = SimpleNamespace(
            num_layers=args.num_layers,
            context_parallel_size=1,
            expert_model_parallel_size=8,
            sequence_length=128,
            micro_batch_size=2,
            train_steps=0,
            learning_rate=1.0e-4,
            seed=1234,
            print_structure=False,
        )
        _bridge, provider, model = build_model(build_args)
        batch = pack_for_contiguous_cp(build_args, step=0, device=device)
    else:
        build_args = SimpleNamespace(
            num_layers=args.num_layers,
            max_seqlen_per_rank=128,
            learning_rate=1.0e-4,
            seed=1234,
            print_structure=False,
        )
        initialize_dynamic_parallel_state(build_args)
        provider, model = build_dynamic_model(build_args)
        assignments = get_schedule(build_args)
        batch = pack_dynamic_microbatch(build_args, assignments[0], step=0, device=device)

    core_model = unwrap_model(model[0])
    model[0].train()
    if dist.get_rank() == 0:
        print_provider_snapshot(provider)
        print_module_inventory(core_model)
    dist.barrier()

    def run_forward():
        return model[0](
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels,
            packed_seq_params=batch.packed_seq_params,
            padding_mask=batch.padding_mask,
        )

    # Compile the training/autograd variants before collecting timings.  A
    # no_grad warmup is insufficient here because native mHC uses torch.compile
    # and produces a separate graph when gradients are enabled.
    warmup_output = run_forward()
    del warmup_output
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if dist.get_rank() == 0:
        handles, records = register_trace_hooks(core_model)
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
    else:
        handles, records = [], []
        profiler_context = contextlib.nullcontext()

    with profiler_context as profiler:
        if dist.get_rank() == 0:
            with record_function("dsv4_sft_forward"):
                output = run_forward()
        else:
            output = run_forward()

    torch.cuda.synchronize()
    dist.barrier()
    output_finite = bool(torch.isfinite(output).all().item())

    if dist.get_rank() == 0:
        for handle in handles:
            handle.remove()
        print(
            f"TRACE_FORWARD_RESULT mode={args.mode} output={_tensor_summary(output)} "
            f"finite={output_finite} peak_memory_bytes={torch.cuda.max_memory_allocated()}",
            flush=True,
        )
        print("TRACE_FORWARD_HOOKS_BEGIN", flush=True)
        for record in records:
            elapsed_ms = record.start.elapsed_time(record.end)
            memory_delta = record.memory_after - record.memory_before
            print(
                f"TRACE_MODULE_RUN order={record.order} name={record.name} "
                f"type={record.module_type} time_ms={elapsed_ms:.4f} "
                f"memory_delta_bytes={memory_delta} inputs={record.inputs} "
                f"outputs={record.outputs}",
                flush=True,
            )
        print("TRACE_FORWARD_HOOKS_END", flush=True)
        print_profiler_summary(profiler)
        print(f"TRACE_FORWARD_SUCCESS mode={args.mode}", flush=True)

    del output
    dist.barrier()
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
