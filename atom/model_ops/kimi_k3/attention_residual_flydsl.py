# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the fused attention-residual kernel (Kimi-K3).

Drop-in for the Triton ``_attn_res_fused_kernel`` in ``attention_residual.py``,
which owns the algorithm's documentation, the custom-op registration, and the
dispatch that selects between the two. Same math, same
``(mixed_output, prefix_out)`` contract, same four fusion folds (``DO_ADD`` /
``DO_ADD2`` / ``WRITE_PREF`` / ``OUT_NORM``).

Same single pass as Triton: one workgroup per token row, the whole of H tiled
across its threads so the running output stays in registers and the softmax
over candidates runs online (flash-style), each candidate read exactly once.

The tiling is where the two differ, and all of it is chosen on the host from
values it already has:

* The workgroup is sized so that H lands on it exactly (``_pick_config``), which
  is what removes the per-lane ``o_d < H`` mask that Triton needs for
  ``BD = next_pow2(H)``. Its width follows the token count, the same way Triton
  picks ``num_warps``: wide when the grid cannot fill the GPU, narrow when it
  can (see ``_WIDE_BLOCK``).
* Several candidates are loaded per iteration, for the same reason Triton
  carries ``BL``: one candidate per iteration exposes a full HBM round trip per
  candidate, which at B=15 is most of the runtime. How many is
  ``_pick_cand(B)``, bounded by a register budget.
* A tile's per-candidate ``(sum v^2, <v, score_weight>)`` reductions are folded
  into ONE cross-workgroup reduction, so a tile costs two barriers rather than
  two per candidate. Those reductions are the floor on this kernel: a
  cross-workgroup one costs ~0.6us per candidate at decode shapes, against
  ~0.03us for the wave-local part of it.

Smaller deliberate differences:

* ``exp2(x * log2e)`` rather than ``exp``, which is what ``tl.exp`` already
  lowers to on ROCm.
* ``eps``/``out_eps`` are baked into the kernel rather than passed as runtime
  scalars (the FlyDSL norm-kernel convention). They are per-model constants, so
  this costs no extra variants.
* The prefix candidate is peeled out after the loop instead of selected
  per-lane by ``tl.where(is_last, ps, v)``; it is already in registers.

B itself stays a RUNTIME argument, as in Triton, so the candidate loop has a
runtime trip count and B only reaches the JIT cache key through the tile width.
A K3 layer stack walks B from 1 up to ``num_hidden_layers / attn_res_block_size``
and hits three flag combinations, which comes to roughly 30 kernel variants at
~35ms each: about a second of JIT inside the existing pre-capture warmup on a
cold FlyDSL cache, and nothing on a warm one.

Measured against the Triton kernel it replaces on MI355X (gfx950), bf16,
H=7168, over the (T, B) pairs of a served K3 trace weighted by their share of
this op's GPU time (``tools/bench_attn_res_flydsl.py``): **1.28x** overall, no
case below parity. Decode (T <= 64) runs 1.00-1.23x: both kernels are latency-
bound there on the per-candidate cross-workgroup reductions, with only 64 of 256
CUs given work by the grid, and B=1 has nothing to tile so it lands at parity.
Prefill token counts run 1.00-1.73x.
"""

# NOTE: do NOT add `from __future__ import annotations` to this file. PEP 563
# turns annotations into strings, which defeats flydsl's runtime-argument
# detection in JitFunction._make_cache_key (`hasattr(ann, "__get_c_pointers__")`
# is False for the string "fx.Int32"). The `B` and `T` parameters would then be
# treated as compile-time constants and have their VALUE embedded in the cache
# key, so every distinct block count and token count would trigger a fresh
# ~40ms JIT compile instead of hitting the cached CompiledFunction.

import math
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import ReductionOp, arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.runtime.device import is_rdna_arch

WARP = 32 if is_rdna_arch() else 64
_LOG2E = 1.4426950408889634

# Workgroup width by token count, the counterpart of Triton's num_warps table.
# One workgroup per row, so below roughly one workgroup per CU the grid cannot
# fill the GPU and a wider block -- more waves on the same row -- is the only
# parallelism left. Past that the tradeoff inverts: narrower blocks pack more
# per CU, so one row's serial chain of candidate reductions overlaps another
# row's memory traffic. Measured on MI355X (256 CUs) the crossover sits between
# 256 and 384 tokens for every B, and is worth 1.1x either side of it.
_WIDE_BLOCK_MAX_TOKENS = 256
_WIDE_BLOCK = 448
_NARROW_BLOCK = 256
# Capped by the AMDGPU flat workgroup limit.
_MAX_BLOCK_THREADS = 1024
# Bounds the register footprint: the accumulator, the prefix, the score weight
# and each in-flight candidate are all this many f32 per lane.
_MAX_ELEMS_PER_THREAD = 32
# Per-lane tile width in ELEMENTS, uniform across dtypes so the f32 score
# weight and the 16-bit candidates share one lane-to-column mapping. 8 makes a
# 16-bit tile one 128b transaction, which is what the bandwidth-bound prefill
# shapes want; wider-than-128b tiles are split into several copies.
_VEC_CANDIDATES = (8, 4, 2, 1)
# Ceiling on the candidates loaded per iteration (Triton's BL). Their loads all
# issue before the tile's reduction, so a wider tile buys fewer exposed HBM
# round trips with registers.
_MAX_CAND = 8
# What it can spend: per-lane f32 for the accumulator, the prefix, the score
# weight and the in-flight candidate tile. Past this the kernel spills, and on
# the narrow (prefill) config it also gives up the occupancy that made a narrow
# block worth choosing -- 4 candidates of 28 elements measured 0.72x where 3
# measured 1.00x, while 8 of 16 on the wide config is the fastest decode point.
_MAX_LANE_ELEMS = 176
_RESIDENT_ROWS = 3  # accumulator, prefix, score weight

_DTYPE_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}
_ELEM_TYPE = {
    "bf16": fx.BFloat16,
    "f16": fx.Float16,
    "f32": fx.Float32,
}
_ELEM_BITS = {"bf16": 16, "f16": 16, "f32": 32}


def _copy_op(bits: int):
    """Buffer copy atom moving ``bits`` per lane."""
    if bits == 128:
        return fx.rocdl.BufferCopy128b()
    if bits == 64:
        return fx.rocdl.BufferCopy64b()
    if bits == 32:
        return fx.rocdl.BufferCopy32b()
    if bits == 16:
        return fx.rocdl.BufferCopy16b()
    raise ValueError(f"no buffer copy atom for {bits} bits")


def _pick_config(H: int, elem_bits: int, tokens: int = 0):
    """Pick ``(block_threads, vec)`` tiling H exactly, or None if none does.

    Block width leads and the tile width follows, because which width the token
    count wants is worth more (1.1x either side of the crossover) than moving a
    16-bit row in 128b rather than 64b transactions.
    """
    target = _WIDE_BLOCK if tokens <= _WIDE_BLOCK_MAX_TOKENS else _NARROW_BLOCK
    # At or below the target, widest first; then anything wider, so an unusual H
    # that only tiles across a wide block still gets a kernel.
    widths = list(range(min(target, _MAX_BLOCK_THREADS), WARP - 1, -WARP))
    widths += list(range(_MAX_BLOCK_THREADS, target, -WARP))
    for block_threads in widths:
        for vec in _VEC_CANDIDATES:
            if H % vec:
                continue
            groups = H // vec
            max_tiles = max(1, _MAX_ELEMS_PER_THREAD // vec)
            if groups % block_threads == 0 and groups // block_threads <= max_tiles:
                return block_threads, vec
    return None


def _pick_cand(B: int, block_threads: int, H: int) -> int:
    """Candidates per iteration: fewest iterations, then fewest duplicate reads.

    A tile that does not divide B re-reads a neighbouring candidate to fill
    itself out, which costs as much bandwidth as a real one, so an exact fit is
    worth more than a wide tile. Bounded by the per-lane register budget, which
    is what makes the wide (decode) config take eight candidates and the narrow
    (prefill) one take three.
    """
    elems = max(1, H // block_threads)
    budget = max(1, _MAX_LANE_ELEMS // elems - _RESIDENT_ROWS)
    limit = max(1, min(_MAX_CAND, budget, max(1, B)))
    best = None
    for cand in range(1, limit + 1):
        iters = -(-B // cand) if B else 1
        key = (iters, cand * iters - B, cand)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


@cache
def _build(
    H: int,
    elem_dtype_str: str,
    ow_dtype_str: str,
    eps: float,
    out_eps: float,
    do_add: bool,
    do_add2: bool,
    write_pref: bool,
    out_norm: bool,
    block_threads: int,
    vec: int,
    cand: int,
):
    """Build the launcher for one specialization. Cached; a call is ~40ms."""
    tiles = H // (block_threads * vec)
    if tiles * block_threads * vec != H:
        raise ValueError(f"H={H} does not tile into {block_threads}x{vec}")
    red_slots = max(1, block_threads // WARP)
    # LDS partials: one slot group per reduced value, and a tile of candidates
    # reduces (sum v^2, dot) for each of them in one pass.
    n_red_max = 2 * cand

    elem_dtype = _ELEM_TYPE[elem_dtype_str]
    elem_bits = _ELEM_BITS[elem_dtype_str]
    ow_dtype = _ELEM_TYPE[ow_dtype_str]
    ow_bits = _ELEM_BITS[ow_dtype_str]
    inv_H = 1.0 / H

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots * n_red_max, 16]

    kernel_kwargs = (
        {} if block_threads <= 256 else {"known_block_size": [block_threads, 1, 1]}
    )

    @flyc.kernel(**kernel_kwargs)
    def attn_res_kernel(
        br: fx.Tensor,  # [T, B, H]  elem
        ps_t: fx.Tensor,  # [T, H]     elem
        sw_t: fx.Tensor,  # [H]        f32
        y_t: fx.Tensor,  # [T, H]     elem
        hs_t: fx.Tensor,  # [T, H]     elem      (aliases ps_t unless DO_ADD)
        hs2_t: fx.Tensor,  # [T, H]     elem      (aliases ps_t unless DO_ADD2)
        pref_t: fx.Tensor,  # [T, H]     elem      (aliases ps_t unless WRITE_PREF)
        ow_t: fx.Tensor,  # [H]        ow dtype  (aliases sw_t unless OUT_NORM)
        B: fx.Int32,
    ):
        t = fx.block_idx.x
        tid = fx.thread_idx.x
        fm = arith.FastMathFlags.fast
        lane = tid % WARP
        wave = tid // WARP
        c_zero = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots * n_red_max, 1))

        def wave_reduce_add(x):
            w = x
            for _step in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _step)
                w = w.addf(w.shuffle_xor(off, WARP), fastmath=fm)
            return w

        def block_reduce_add(values):
            """Sum each value across the workgroup; every thread gets them all.

            One pass for the whole list, so a tile of candidates costs two
            barriers no matter how many reductions it needs.
            """
            n = len(values)
            waved = [wave_reduce_add(v) for v in values]
            if const_expr(red_slots == 1):
                return waved
            if lane == 0:
                for k in range_constexpr(n):
                    fx.memref_store(waved[k], s_red, k * red_slots + wave)
            gpu.barrier()
            # One value per wave, so the cross-wave stage costs one wave_reduce
            # rather than n of them. With more values than waves it takes
            # another round rather than serialising them all onto wave 0.
            for r in range_constexpr(-(-n // red_slots)):
                k = wave + r * red_slots
                if k < n:
                    in_range = lane < red_slots
                    slot = k * red_slots + in_range.select(lane, 0)
                    part = in_range.select(fx.memref_load(s_red, slot), c_zero)
                    part = wave_reduce_add(part)
                    if lane == 0:
                        fx.memref_store(part, s_red, k * red_slots)
            gpu.barrier()
            return [fx.memref_load(s_red, k * red_slots) for k in range_constexpr(n)]

        def chunking(bits):
            """Elements per copy and copies per tile for a ``bits``-wide dtype.

            A tile is ``vec`` elements; anything past a 128b transaction has to
            be split, which is how an f32 weight row rides the same lane-to-
            column mapping as a 16-bit candidate row.
            """
            chunk = min(vec, max(1, 128 // bits))
            return chunk, vec // chunk

        def port(buf, leading, dtype, bits):
            """Bind a [H] row of ``buf`` to this lane's tiles.

            Returns ``(load, store)`` over tile index; the chunk split, the copy
            atom and the divided views are all resolved once here.
            """
            chunk, n_copy = chunking(bits)
            atom = fx.make_copy_atom(_copy_op(chunk * bits), bits)
            chunk_lay = fx.make_layout(chunk, 1)
            view = fx.slice(buf, leading + (None,)) if leading else buf
            div = fx.logical_divide(view, chunk_lay)

            def chunks(reg, i):
                """(global, register) view pairs covering tile ``i``.

                A list, not a generator: ``yield`` inside a traced function is
                claimed by the scf.for rewriter.
                """
                reg_div = fx.logical_divide(reg, chunk_lay)
                base = (tid + i * block_threads) * n_copy
                return [
                    (
                        fx.slice(div, (None, base + k)),
                        fx.slice(reg_div, (None, k)),
                    )
                    for k in range_constexpr(n_copy)
                ]

            def load(i):
                """Tile ``i`` of this lane's columns, widened to f32."""
                reg = fx.make_rmem_tensor(vec, dtype)
                for src, dst in chunks(reg, i):
                    fx.copy(atom, src, dst)
                loaded = fx.memref_load_vec(reg)
                return (
                    loaded if const_expr(dtype is fx.Float32) else loaded.to(fx.Float32)
                )

            def store(i, value):
                reg = fx.make_rmem_tensor(vec, dtype)
                fx.memref_store_vec(
                    value if const_expr(dtype is fx.Float32) else value.to(dtype), reg
                )
                for dst, src in chunks(reg, i):
                    fx.copy(atom, src, dst)

            return load, store

        def elem_port(tensor, leading=()):
            return port(
                fx.rocdl.make_buffer_tensor(tensor), leading, elem_dtype, elem_bits
            )

        def load_row(load):
            return [load(i) for i in range_constexpr(tiles)]

        br_buf = fx.rocdl.make_buffer_tensor(br)
        sw_load, _sw_store = port(fx.rocdl.make_buffer_tensor(sw_t), (), fx.Float32, 32)
        sw = load_row(sw_load)

        # The prefix is the last candidate: loaded once and kept in registers
        # for the whole loop, since re-reading it per tile would undo the
        # single-pass property. The caller's `prefix_sum = prefix_sum + ...`
        # adds fold into this load.
        ps = load_row(elem_port(ps_t, (t,))[0])
        if const_expr(do_add):
            hs = load_row(elem_port(hs_t, (t,))[0])
            for i in range_constexpr(tiles):
                ps[i] = ps[i] + hs[i]
        if const_expr(do_add2):
            hs2 = load_row(elem_port(hs2_t, (t,))[0])
            for i in range_constexpr(tiles):
                ps[i] = ps[i] + hs2[i]
        if const_expr(write_pref):
            _pref_load, pref_store = elem_port(pref_t, (t,))
            for i in range_constexpr(tiles):
                pref_store(i, ps[i])

        def fold(cands, valid, m_prev, den_prev, acc_prev):
            """Score a tile of candidates and fold it into the online softmax.

            score_weight is norm_weight * proj_weight, precomputed at load
            time, so one dot product covers both the rmsnorm gain and the
            scoring projection. ``valid`` is None when every candidate counts.
            """
            n = len(cands)
            partials = []
            for j in range_constexpr(n):
                sum_sq = c_zero
                dot = c_zero
                for i in range_constexpr(tiles):
                    v = cands[j][i]
                    sum_sq = sum_sq + (v * v).reduce(ReductionOp.ADD, fastmath=fm)
                    dot = dot + (v * sw[i]).reduce(ReductionOp.ADD, fastmath=fm)
                partials.append(sum_sq)
                partials.append(dot)
            sums = block_reduce_add(partials)

            scores = []
            for j in range_constexpr(n):
                rstd = fmath.rsqrt(sums[2 * j] * inv_H + eps, fastmath=fm)
                score = sums[2 * j + 1] * rstd
                # Candidates past B score -inf, so they carry no softmax mass.
                scores.append(
                    score if valid is None else valid[j].select(score, c_neg_inf)
                )

            m_new = m_prev
            for j in range_constexpr(n):
                m_new = fx.maxnumf(m_new, scores[j])
            # One rescale for the whole tile, as in Triton's BL formulation.
            rescale = fmath.exp2((m_prev - m_new) * _LOG2E, fastmath=fm)
            weights = [
                fmath.exp2((scores[j] - m_new) * _LOG2E, fastmath=fm)
                for j in range_constexpr(n)
            ]

            den = den_prev * rescale
            for j in range_constexpr(n):
                den = den + weights[j]
            acc = []
            for i in range_constexpr(tiles):
                a = acc_prev[i] * rescale
                for j in range_constexpr(n):
                    a = a + cands[j][i] * weights[j]
                acc.append(a)
            return m_new, den, acc

        # Online softmax over the B block_residual candidates, `cand` per
        # iteration. Runtime trip count, so the state (next candidate index,
        # running max, running denominator, running weighted sum) rides
        # scf.for's iter_args.
        state_init = [fx.Int32(0), c_neg_inf, c_zero] + [
            fx.Vector.filled(vec, 0.0, fx.Float32) for _ in range_constexpr(tiles)
        ]
        n_iters = (B + (cand - 1)) // cand
        results = state_init
        for _it, state in range(
            fx.Index(0), fx.Index(n_iters), fx.Index(1), init=state_init
        ):
            b0 = fx.Int32(state[0])
            cands = []
            valid = []
            # Every load in the tile issues before the first reduction, so
            # their latencies overlap instead of serializing per candidate.
            # Candidates past B read a neighbour and are masked out of the
            # softmax below. Predicating the copy off instead would save that
            # read, but putting the loads inside an scf.if stops them issuing
            # together, which costs far more than the duplicate read; the host
            # keeps the duplicates rare by sizing `cand` to B.
            for j in range_constexpr(cand):
                b = b0 + j
                in_range = b < B
                valid.append(in_range)
                cand_load, _ = port(
                    br_buf, (t, in_range.select(b, B - 1)), elem_dtype, elem_bits
                )
                cands.append([cand_load(i) for i in range_constexpr(tiles)])
            m_new, den, acc = fold(
                cands,
                valid,
                fx.Float32(state[1]),
                fx.Float32(state[2]),
                [fx.Vector(state[3 + i]) for i in range_constexpr(tiles)],
            )
            results = yield [b0 + cand, m_new, den] + acc

        # Candidate B: the prefix, already in registers.
        _m, den, acc = fold(
            [ps],
            None,
            fx.Float32(results[1]),
            fx.Float32(results[2]),
            [fx.Vector(results[3 + i]) for i in range_constexpr(tiles)],
        )

        inv_den = fx.Float32(1.0) / den
        out = [acc[i] * inv_den for i in range_constexpr(tiles)]
        if const_expr(out_norm):
            # Free: out is already fully formed in registers. The weight load
            # issues before the reduction so the two overlap.
            ow_load, _ = port(fx.rocdl.make_buffer_tensor(ow_t), (), ow_dtype, ow_bits)
            ow = load_row(ow_load)
            sum_sq = c_zero
            for i in range_constexpr(tiles):
                sum_sq = sum_sq + (out[i] * out[i]).reduce(ReductionOp.ADD, fastmath=fm)
            rstd = fmath.rsqrt(
                block_reduce_add([sum_sq])[0] * inv_H + out_eps, fastmath=fm
            )
            for i in range_constexpr(tiles):
                out[i] = out[i] * rstd * ow[i]

        _y_load, y_store = elem_port(y_t, (t,))
        for i in range_constexpr(tiles):
            y_store(i, out[i])

    @flyc.jit
    def launch(
        br: fx.Tensor,
        ps_t: fx.Tensor,
        sw_t: fx.Tensor,
        y_t: fx.Tensor,
        hs_t: fx.Tensor,
        hs2_t: fx.Tensor,
        pref_t: fx.Tensor,
        ow_t: fx.Tensor,
        B: fx.Int32,
        T: fx.Int32,
        stream: fx.Stream,
    ):
        attn_res_kernel(br, ps_t, sw_t, y_t, hs_t, hs2_t, pref_t, ow_t, B).launch(
            grid=(T, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def _run_compiled(launcher, *args):
    """First call compiles and runs; later calls dispatch the cached function.

    One CompiledFunction per launcher is correct because ``_build`` is keyed on
    everything the kernel specializes over; the tensor shapes, ``B`` and ``T``
    all stay dynamic.
    """
    compiled = getattr(launcher, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return
    try:
        launcher._cf = flyc.compile(launcher, *args)
    except Exception:
        # flyc.compile leaks the ir.Context on failure; pop it so a retry
        # starts from a clean state.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def flydsl_attn_res_supported(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    out_norm_weight: torch.Tensor | None = None,
) -> bool:
    """Whether this shape/dtype combination has a FlyDSL kernel.

    The gate exists because the kernel deliberately has no per-lane H mask:
    widths that do not tile exactly across a workgroup stay on Triton.
    """
    elem_dtype_str = _DTYPE_STR.get(prefix_sum.dtype)
    if elem_dtype_str is None or block_residual.dtype is not prefix_sum.dtype:
        return False
    if score_weight.dtype is not torch.float32:
        return False
    if out_norm_weight is not None and out_norm_weight.dtype not in _DTYPE_STR:
        return False
    H = prefix_sum.shape[-1]
    if block_residual.shape[-1] != H or score_weight.numel() != H:
        return False
    return _pick_config(H, _ELEM_BITS[elem_dtype_str]) is not None


def flydsl_apply_attn_res(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]
    score_weight: torch.Tensor,  # [H] f32
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H]
    out_norm_weight: torch.Tensor | None = None,  # [H]
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,  # [T, H]
    config: tuple | None = None,
) -> tuple:
    """FlyDSL implementation of ``_apply_attn_res_impl``; same contract.

    ``config`` overrides the ``(block_threads, vec, cand)`` tiling; it exists
    for the benchmark and is not used in production.
    """
    T, B, H = block_residual.shape
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    if do_add2 and not do_add:
        raise ValueError("add_hidden2 requires add_hidden")
    out_norm = out_norm_weight is not None

    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    ow = out_norm_weight.contiguous() if out_norm else sw
    # hs/hs2/pref are always bound (the kernel signature is fixed); when not
    # adding they alias ps and are never dereferenced, exactly as in Triton.
    hs = add_hidden.contiguous() if do_add else ps
    hs2 = add_hidden2.contiguous() if do_add2 else ps
    pref = torch.empty_like(ps) if do_add else ps

    elem_dtype_str = _DTYPE_STR[ps.dtype]
    if config is None:
        block_threads, vec = _pick_config(H, _ELEM_BITS[elem_dtype_str], T)
        config = (block_threads, vec, _pick_cand(B, block_threads, H))
    launcher = _build(
        H,
        elem_dtype_str,
        _DTYPE_STR[ow.dtype],
        float(eps),
        float(out_eps),
        do_add,
        do_add2,
        do_add,  # write_pref: store the summed prefix iff we summed one
        out_norm,
        *config,
    )
    _run_compiled(
        launcher,
        br,
        ps,
        sw,
        y,
        hs,
        hs2,
        pref,
        ow,
        B,
        T,
        fx.Stream(torch.cuda.current_stream()),
    )
    return y, (pref if do_add else prefix_sum)
