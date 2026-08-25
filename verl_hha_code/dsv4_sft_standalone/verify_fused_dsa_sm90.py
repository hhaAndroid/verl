#!/usr/bin/env python3
"""Run the exact FlashMLA-forward/cuDNN-DSA-backward pair used on H200."""

import inspect

import torch
from cudnn import DSA
from flash_mla import flash_mla_sparse_fwd


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    print(f"device={torch.cuda.get_device_name(device)} capability={capability}")
    if capability != (9, 0):
        raise RuntimeError(f"This verifier targets Hopper SM90, got {capability}")

    indexer_signature = inspect.signature(DSA.indexer_forward_wrapper)
    print(f"indexer_forward_signature={indexer_signature}")
    if "q_causal_offsets" not in indexer_signature.parameters:
        raise RuntimeError(
            "cuDNN Frontend is too old: install MCore's pinned source commit "
            "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
        )

    torch.manual_seed(2026)
    total_q, total_kv, num_heads, head_dim, topk = 16, 64, 64, 512, 128
    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(total_kv, head_dim, device=device, dtype=torch.bfloat16)
    sink = torch.randn(num_heads, device=device, dtype=torch.float32)

    # FlashMLA's SM90 sparse prefill requires TopK aligned to 128.  Only the
    # first 64 entries are real here; -1 is its padding sentinel.
    indices = torch.full((total_q, topk), -1, device=device, dtype=torch.int32)
    indices[:, :total_kv] = torch.arange(total_kv, device=device, dtype=torch.int32)
    softmax_scale = head_dim**-0.5

    out, _max_logits, lse = flash_mla_sparse_fwd(
        q,
        kv.unsqueeze(1),
        indices.unsqueeze(1),
        softmax_scale,
        d_v=head_dim,
        attn_sink=sink,
    )
    torch.cuda.synchronize(device)
    assert out.shape == (total_q, num_heads, head_dim)
    assert lse.shape == (total_q, num_heads)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    print(f"FLASHMLA_SM90_FWD_OK out={list(out.shape)} lse={list(lse.shape)}")

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        torch.randn_like(out),
        lse,
        sink,
        indices,
        softmax_scale=softmax_scale,
        topk_length=None,
    )
    torch.cuda.synchronize(device)
    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert dq.shape == q.shape and dkv.shape == kv.shape and d_sink.shape == sink.shape
    assert torch.isfinite(dq).all() and torch.isfinite(dkv).all() and torch.isfinite(d_sink).all()
    print(
        "CUDNN_DSA_SM90_BWD_OK "
        f"dq={list(dq.shape)} dkv={list(dkv.shape)} d_sink={list(d_sink.shape)}"
    )
    print("FUSED_DSA_SM90_VERIFY_SUCCESS")


if __name__ == "__main__":
    main()
