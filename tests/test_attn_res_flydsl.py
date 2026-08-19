# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL attention-residual kernel vs its reference and vs Triton (Kimi-K3).

The FlyDSL kernel in ``attention_residual_flydsl`` is a drop-in for the Triton
``_attn_res_fused_kernel``, selected by ``ATOM_ATTN_RES_USE_FLYDSL``. Both are
checked against the same pure-torch oracle, so a mismatch says which one moved.

What the sweep is chosen to cover:

* ``B`` picks out every tile shape the host can choose. ``_pick_cand`` sizes the
  candidate tile to B, so B=1/4/5 tile exactly, B=15 needs three tiles, and B=0
  leaves the candidate loop with zero trip count and only the prefix to mix.
* ``T`` crosses the decode/prefill divide (one workgroup at T=1, more
  workgroups than CUs at T=2048) without changing what the kernel computes.
* The flag matrix is the four fusion folds. ``add_hidden2`` without
  ``add_hidden`` is rejected rather than silently ignored, and ``out_norm``
  changes the store, so both orders of both matter.

Tolerance is one bf16 ulp at the output magnitude (the kernel accumulates in
fp32 and rounds once on store), and ``prefix_out`` must match to 1e-2.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares GPU kernels against their reference; needs a real GPU",
        allow_module_level=True,
    )

pytest.importorskip("flydsl", reason="the kernel under test is written in FlyDSL")

from atom.model_ops.kimi_k3.attention_residual import _apply_attn_res_impl
from atom.model_ops.kimi_k3.attention_residual_flydsl import (
    _pick_cand,
    _pick_config,
    flydsl_apply_attn_res,
    flydsl_attn_res_supported,
)

DEV = "cuda"
DT = torch.bfloat16
H = 7168  # Kimi-K3's residual stream, never TP-sharded
EPS = 1e-6

FLAGS = [
    (False, False, False),
    (False, False, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
]


def attn_res_reference(
    prefix_sum,
    block_residual,
    score_weight,
    eps,
    add_hidden=None,
    out_norm_weight=None,
    out_eps=1e-6,
    add_hidden2=None,
):
    """Oracle for the fused op, all of it in fp32. Returns (y, prefix_out)."""
    _, _, hidden = block_residual.shape
    out_dtype = prefix_sum.dtype

    ps = prefix_sum.float()
    if add_hidden is not None:
        ps = ps + add_hidden.float()
        if add_hidden2 is not None:
            ps = ps + add_hidden2.float()
    prefix_out = ps.to(out_dtype) if add_hidden is not None else prefix_sum

    # The B block rows, then the (summed) prefix as candidate B.
    v = torch.cat([block_residual.float(), ps.unsqueeze(1)], dim=1)
    rstd = torch.rsqrt(v.pow(2).sum(-1) / hidden + eps)
    scores = (v * score_weight.float()).sum(-1) * rstd
    probs = torch.softmax(scores, dim=-1)
    y = (probs.unsqueeze(-1) * v).sum(1)

    if out_norm_weight is not None:
        y = y * torch.rsqrt(y.pow(2).sum(-1, keepdim=True) / hidden + out_eps)
        y = y * out_norm_weight.float()
    return y.to(out_dtype), prefix_out


def _inputs(T, B, do_add, do_add2, out_norm, seed=0, hidden=H):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def randn(*shape, dtype=DT):
        return torch.randn(*shape, device=DEV, dtype=dtype, generator=gen)

    return {
        "prefix_sum": randn(T, hidden),
        "block_residual": randn(T, B, hidden),
        # score_weight folds the rmsnorm gain into the scoring projection, so
        # it is fp32 and small; a unit-scale one would saturate the softmax.
        "score_weight": randn(hidden, dtype=torch.float32) * 0.05,
        "eps": EPS,
        "add_hidden": randn(T, hidden) if do_add else None,
        "out_norm_weight": (
            randn(hidden, dtype=torch.float32) * 0.1 + 1 if out_norm else None
        ),
        "out_eps": EPS,
        "add_hidden2": randn(T, hidden) if do_add2 else None,
    }


def _assert_matches(got, want, tag):
    y, prefix = got
    y_ref, prefix_ref = want
    scale = max(y_ref.float().abs().max().item(), 1e-3)
    y_err = (y.float() - y_ref.float()).abs().max().item()
    prefix_err = (prefix.float() - prefix_ref.float()).abs().max().item()
    assert y_err <= 2e-2 * scale, f"{tag}: y off by {y_err:.3e} (|y|max {scale:.3f})"
    assert prefix_err <= 1e-2, f"{tag}: prefix_out off by {prefix_err:.3e}"


@pytest.mark.parametrize("T,B", [(1, 1), (1, 15), (4, 0), (8, 5), (64, 15), (2048, 4)])
@pytest.mark.parametrize("do_add,do_add2,out_norm", FLAGS)
def test_flydsl_matches_reference(T, B, do_add, do_add2, out_norm):
    kwargs = _inputs(T, B, do_add, do_add2, out_norm)
    got = flydsl_apply_attn_res(**kwargs)
    _assert_matches(got, attn_res_reference(**kwargs), f"T={T} B={B}")


@pytest.mark.parametrize("T,B", [(1, 15), (64, 15), (2048, 8)])
@pytest.mark.parametrize("do_add,do_add2,out_norm", [FLAGS[0], FLAGS[5]])
def test_flydsl_matches_triton(T, B, do_add, do_add2, out_norm, monkeypatch):
    """Both kernels against the oracle, so a failure localises to one of them."""
    monkeypatch.setenv("ATOM_ATTN_RES_USE_FLYDSL", "never")
    kwargs = _inputs(T, B, do_add, do_add2, out_norm)
    want = attn_res_reference(**kwargs)
    triton_got = _apply_attn_res_impl(**kwargs)
    _assert_matches(triton_got, want, f"triton T={T} B={B}")
    _assert_matches(flydsl_apply_attn_res(**kwargs), want, f"flydsl T={T} B={B}")


def test_prefix_out_aliases_input_without_add():
    """No addend means no prefix to write, so the input comes straight back."""
    kwargs = _inputs(4, 3, False, False, True)
    _, prefix = flydsl_apply_attn_res(**kwargs)
    assert prefix is kwargs["prefix_sum"]


def test_add_hidden2_requires_add_hidden():
    kwargs = _inputs(4, 3, False, True, False)
    with pytest.raises(ValueError, match="add_hidden2 requires add_hidden"):
        flydsl_apply_attn_res(**kwargs)


def test_strided_inputs():
    """The model hands over slices of wider tensors; the kernel must not care."""
    T, B = 8, 4
    kwargs = _inputs(T, B, True, False, True)
    kwargs["prefix_sum"] = torch.randn(T, 2 * H, device=DEV, dtype=DT)[:, :H]
    kwargs["block_residual"] = torch.randn(T, B, 2 * H, device=DEV, dtype=DT)[:, :, :H]
    assert not kwargs["prefix_sum"].is_contiguous()
    _assert_matches(
        flydsl_apply_attn_res(**kwargs), attn_res_reference(**kwargs), "strided"
    )


@pytest.mark.parametrize("ow_dtype", [torch.float32, torch.bfloat16])
def test_out_norm_weight_dtypes(ow_dtype):
    kwargs = _inputs(8, 4, True, True, True)
    kwargs["out_norm_weight"] = kwargs["out_norm_weight"].to(ow_dtype)
    _assert_matches(
        flydsl_apply_attn_res(**kwargs), attn_res_reference(**kwargs), str(ow_dtype)
    )


@pytest.mark.parametrize("hidden", [4096, 2048, 512])
def test_other_hidden_sizes(hidden):
    """H only has to tile exactly across a workgroup, not be 7168."""
    kwargs = _inputs(4, 3, True, False, True, hidden=hidden)
    _assert_matches(
        flydsl_apply_attn_res(**kwargs), attn_res_reference(**kwargs), f"H={hidden}"
    )


def test_cuda_graph_capture():
    """The decode path is graph-captured, so replay must recompute from the
    captured buffers rather than replaying a stale result."""
    kwargs = _inputs(64, 15, True, True, True)
    flydsl_apply_attn_res(**kwargs)  # compile before capture, as warmup does
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(graph):
        out = flydsl_apply_attn_res(**kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    for seed in (1, 2):
        fresh = _inputs(64, 15, True, True, True, seed=seed)
        for key, value in fresh.items():
            if torch.is_tensor(value):
                kwargs[key].copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches(out, attn_res_reference(**kwargs), f"replay seed={seed}")


def test_dispatch_modes(monkeypatch):
    """auto/always agree with never; always refuses what it cannot serve."""
    kwargs = _inputs(8, 5, True, False, True)
    monkeypatch.setenv("ATOM_ATTN_RES_USE_FLYDSL", "never")
    triton_y, _ = _apply_attn_res_impl(**kwargs)
    for mode in ("auto", "always"):
        monkeypatch.setenv("ATOM_ATTN_RES_USE_FLYDSL", mode)
        y, _ = _apply_attn_res_impl(**kwargs)
        scale = max(triton_y.float().abs().max().item(), 1e-3)
        assert (y.float() - triton_y.float()).abs().max().item() <= 4e-2 * scale

    odd = dict(kwargs)
    odd["prefix_sum"] = torch.randn(4, 4095, device=DEV, dtype=DT)
    odd["block_residual"] = torch.randn(4, 2, 4095, device=DEV, dtype=DT)
    odd["score_weight"] = torch.randn(4095, device=DEV, dtype=torch.float32) * 0.05
    odd["add_hidden"] = torch.randn(4, 4095, device=DEV, dtype=DT)
    odd["out_norm_weight"] = torch.randn(4095, device=DEV, dtype=torch.float32)
    assert not flydsl_attn_res_supported(
        odd["prefix_sum"], odd["block_residual"], odd["score_weight"]
    )
    monkeypatch.setenv("ATOM_ATTN_RES_USE_FLYDSL", "auto")
    _apply_attn_res_impl(**odd)  # falls back to Triton
    monkeypatch.setenv("ATOM_ATTN_RES_USE_FLYDSL", "always")
    with pytest.raises(RuntimeError, match="ATOM_ATTN_RES_USE_FLYDSL=always"):
        _apply_attn_res_impl(**odd)


def test_support_gate():
    kwargs = _inputs(4, 2, False, False, True)
    ps, br, sw = kwargs["prefix_sum"], kwargs["block_residual"], kwargs["score_weight"]
    assert flydsl_attn_res_supported(ps, br, sw, kwargs["out_norm_weight"])
    # score_weight arrives pre-multiplied in fp32; a 16-bit one is a caller bug.
    assert not flydsl_attn_res_supported(ps, br, sw.bfloat16())
    assert not flydsl_attn_res_supported(ps, br.float(), sw)


def test_token_count_is_not_a_compile_time_constant():
    """Two token counts that share a workgroup shape must share one build.

    This is the guard on the module's ban on ``from __future__ import
    annotations``: with annotations stringified, flydsl stops recognising ``T``
    and ``B`` as runtime arguments and bakes their VALUES into the JIT cache
    key, so every batch size would pay a fresh ~40ms compile. That failure is
    invisible in output values and only shows up as a build count.
    """
    from atom.model_ops.kimi_k3.attention_residual_flydsl import _build

    # Same B, and both token counts on the same side of the block-width
    # crossover, so the only thing differing is the runtime grid.
    first = _inputs(8, 4, True, True, True)
    flydsl_apply_attn_res(**first)
    builds = _build.cache_info().currsize
    for T in (16, 64, 200):
        flydsl_apply_attn_res(**_inputs(T, 4, True, True, True))
    assert _build.cache_info().currsize == builds, (
        "token count leaked into the JIT cache key; check that this module has "
        "no `from __future__ import annotations`"
    )


def test_tile_choice_covers_b():
    """Every B must be covered by ceil(B/cand) tiles with minimal duplication."""
    block_threads, _vec = _pick_config(H, 16)
    for B in range(1, 17):
        cand = _pick_cand(B, block_threads, H)
        iters = -(-B // cand)
        assert iters * cand >= B
        # A tile that divides B exists for every B here, or costs at most one
        # duplicate read per iteration.
        assert iters * cand - B <= iters
