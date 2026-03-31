#!/usr/bin/env python3
"""Hierarchical roofline analysis for GPT-OSS 120B on AMD Instinct MI355X.

Two memory ceilings:
  - HBM3e: 8 TB/s   (main memory)
  - Infinity Cache (LLC): 256 MB, ~24 TB/s estimated
    Architecture: 128 slices × 64 B/cycle (CDNA 4 whitepaper).
    MI300X measured at 17.2 TB/s (2.1 GHz IOD).  MI355X IOD is "significantly
    enhanced" per whitepaper; 24 TB/s is derived from kernel timing data
    (consistent with ~2.9 GHz effective IOD clock).

Trace timestamps are in microseconds (Chrome Trace Format standard); the input
CSV column labelled avg_time_ms actually holds microsecond values.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "gpt_oss_120b_kernel_analysis.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "gpt_oss_120b_roofline.csv")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "gpt_oss_120b_roofline.png")

# ── MI355X hardware specs (per GPU) ──────────────────────────────────────────
PEAK_BF16_TFLOPS = 2_500
PEAK_MXFP4_TFLOPS = 10_100

PEAK_HBM_BW_GBs = 8_000       # HBM3e
PEAK_IC_BW_GBs = 24_000        # Infinity Cache (256 MB LLC), estimated
IC_SIZE_MB = 256

# Ridge points (FLOPs/byte where memory slope meets compute ceiling)
RIDGE_HBM_BF16 = PEAK_BF16_TFLOPS * 1e3 / PEAK_HBM_BW_GBs     # 312.5
RIDGE_HBM_MXFP4 = PEAK_MXFP4_TFLOPS * 1e3 / PEAK_HBM_BW_GBs   # 1262.5
RIDGE_IC_BF16 = PEAK_BF16_TFLOPS * 1e3 / PEAK_IC_BW_GBs         # 104.2
RIDGE_IC_MXFP4 = PEAK_MXFP4_TFLOPS * 1e3 / PEAK_IC_BW_GBs      # 420.8

B = 4096
AVG_SEQ_LEN = 1024


# ── FLOP / byte estimation per kernel ───────────────────────────────────────

def compute_flops_bytes(short_name, full_name, model_layer):
    """Return (flops, bytes, precision, cache_level) or None."""

    # ── BF16 GEMM kernels ────────────────────────────────────────────────────
    if short_name == "Cijk GEMM (QKV decode)":
        M, K, N = 4096, 2880, 640
        flops = 2 * M * K * N
        byt = (M * K + N * K + M * N) * 2  # ~32.5 MB, fits in IC
        return flops, byt, "bf16", "ic"

    if short_name == "Cijk GEMM (O-proj decode)":
        M, K, N = 4096, 512, 2880
        flops = 2 * M * K * N
        byt = (M * K + N * K + M * N) * 2  # ~30.7 MB, fits in IC
        return flops, byt, "bf16", "ic"

    if short_name == "bf16gemm (gating GEMM)":
        M, K, N = 4096, 2880, 128
        flops = 2 * M * K * N
        byt = (M * K + N * K + M * N) * 2  # ~25.4 MB, fits in IC
        return flops, byt, "bf16", "ic"

    if short_name == "Cijk GEMM (LM head prefill)":
        M, K, N = 4, 2880, 25136
        flops = 2 * M * K * N
        byt = (M * K + N * K + M * N) * 2  # ~145 MB, fits in IC
        return flops, byt, "bf16", "ic"

    # ── MXFP4 MoE GEMM kernels ──────────────────────────────────────────────
    if short_name == "MoeFlatmm (gate+up SwiGLU)":
        dispatched = 20476
        K_in, N_out = 3072, 1024
        flops = 2 * dispatched * K_in * N_out
        act_bytes = dispatched * K_in * 2
        wt_bytes = 128 * N_out * (K_in // 2)
        scale_bytes = 131072 * 96 + 20476 * 96 + 128 * 1024 * 4
        out_bytes = dispatched * (N_out // 2) * 2
        byt = act_bytes + wt_bytes + scale_bytes + out_bytes
        # Total ~363 MB exceeds IC, but kernel tiles through experts;
        # weights (~201 MB FP4) stay resident in IC across iterations.
        return flops, byt, "mxfp4", "ic"

    if short_name == "MoeFlatmm (down proj)":
        dispatched = 20476
        K_in, N_out = 512, 3072
        flops = 2 * dispatched * K_in * N_out
        act_bytes = dispatched * K_in * 2
        wt_bytes = 128 * N_out * (K_in // 2)
        scale_bytes = 20476 * 96 + 393216 * 16 + 128 * 3072 * 4
        out_bytes = B * N_out * 2
        byt = act_bytes + wt_bytes + scale_bytes + out_bytes
        return flops, byt, "mxfp4", "ic"  # ~157 MB, fits in IC

    # ── Elementwise / memory-bound kernels ───────────────────────────────────
    if "add_rmsnorm_quant" in short_name:
        H = 2880
        flops = 5 * B * H
        byt = (2 * B * H) * 2 + H * 2 + (2 * B * H) * 2 + B * H + B * 32 * 4
        return flops, byt, "bf16", "ic"  # ~107 MB, fits in IC

    if short_name == "fused_qk_rope_reshape_and_cache":
        q_dim, k_dim = 640, 64
        flops = 6 * B * (q_dim + k_dim)
        byt = (2 * B * q_dim + 4 * B * k_dim) * 2
        return flops, byt, "bf16", "ic"  # ~12.6 MB


    if short_name == "paged_attention_ps_reduce":
        flops = B * 512 * 2 * 2
        byt = B * 512 * 2 * 4
        return flops, byt, "bf16", "ic"  # ~16.8 MB

    if short_name == "topkGatingSoftmax":
        flops = B * 128 * 5
        byt = B * 128 * 2 + B * 4 * (4 + 4)
        return flops, byt, "bf16", "ic"

    if short_name == "triton_fused_pad_moe":
        flops = B * 3072
        byt = 2 * B * 3072 * 2
        return flops, byt, "bf16", "ic"

    if short_name == "MoeSortingKernel":
        flops = B * 4 * 10
        byt = B * 3072 * 2 + B * 4 * 8
        return flops, byt, "bf16", "ic"

    if short_name == "masked_embedding":
        flops = 0
        byt = B * 2880 * 2
        return flops, byt, "bf16", "ic"

    if short_name == "kv_indices_generate":
        flops = B * 10
        byt = B * 8 * 2
        return flops, byt, "bf16", "ic"

    if short_name == "mix_sample_outer_exponential":
        vocab, bs = 201088, 4
        flops = bs * vocab * 5
        byt = bs * vocab * (2 + 4 + 4) + bs * 4
        return flops, byt, "bf16", "ic"

    if model_layer in ("nccl-allreduce", "output-allgather", "memcopy"):
        return None

    return None


# ── Roofline helpers ─────────────────────────────────────────────────────────

def roofline_tflops(oi, precision, bw_gbs):
    """Roofline-limited TFLOPS at given OI and bandwidth."""
    peak = PEAK_MXFP4_TFLOPS if precision == "mxfp4" else PEAK_BF16_TFLOPS
    mem_limited = oi * bw_gbs / 1e3
    return min(peak, mem_limited)


def classify_bottleneck(oi, precision, cache_level):
    """Classify based on OI vs ridge point for the applicable cache level."""
    if cache_level == "ic":
        ridge = RIDGE_IC_MXFP4 if precision == "mxfp4" else RIDGE_IC_BF16
    else:
        ridge = RIDGE_HBM_MXFP4 if precision == "mxfp4" else RIDGE_HBM_BF16
    return "compute" if oi >= ridge else f"memory ({cache_level.upper()})"


def main():
    rows_in = []
    with open(INPUT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows_in.append(r)

    results = []
    for row in rows_in:
        layer = row["model_layer"]
        short = row["kernel_short_name"]
        full = row["kernel_full_name"]
        avg_us = float(row["avg_time_ms"])

        ret = compute_flops_bytes(short, full, layer)
        if ret is None:
            results.append({
                "model_layer": layer,
                "kernel_short_name": short,
                "avg_time_us": round(avg_us, 3),
                "flops_per_instance": "N/A",
                "bytes_per_instance": "N/A",
                "operational_intensity": "N/A",
                "attained_tflops": "N/A",
                "roofline_tflops": "N/A",
                "efficiency_pct": "N/A",
                "precision": "N/A",
                "cache_level": "N/A",
                "bottleneck": "communication",
            })
            continue

        flops, byt, precision, cache_level = ret
        bw = PEAK_IC_BW_GBs if cache_level == "ic" else PEAK_HBM_BW_GBs

        oi = flops / byt if byt > 0 else 0.0
        time_s = avg_us / 1e6
        attained = (flops / time_s) / 1e12 if time_s > 0 else 0.0
        roof = roofline_tflops(oi, precision, bw)
        eff = (attained / roof * 100) if roof > 0 else 0.0
        bottleneck = classify_bottleneck(oi, precision, cache_level)

        results.append({
            "model_layer": layer,
            "kernel_short_name": short,
            "avg_time_us": round(avg_us, 3),
            "flops_per_instance": int(flops),
            "bytes_per_instance": int(byt),
            "operational_intensity": round(oi, 4),
            "attained_tflops": round(attained, 4),
            "roofline_tflops": round(roof, 4),
            "efficiency_pct": round(eff, 2),
            "precision": precision,
            "cache_level": cache_level,
            "bottleneck": bottleneck,
        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    fieldnames = [
        "model_layer", "kernel_short_name", "avg_time_us",
        "flops_per_instance", "bytes_per_instance", "operational_intensity",
        "attained_tflops", "roofline_tflops", "efficiency_pct",
        "precision", "cache_level", "bottleneck",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Roofline CSV written to {OUTPUT_CSV}")

    # ── Generate roofline plot ───────────────────────────────────────────────
    plot_results = [
        r for r in results
        if r["bottleneck"] != "communication"
        and r["operational_intensity"] != "N/A"
        and float(r["operational_intensity"]) > 0
    ]

    fig, ax = plt.subplots(figsize=(18, 11))

    oi_min, oi_max = 0.01, 5000
    oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), 500)

    # Memory bandwidth slopes
    hbm_line = oi_range * PEAK_HBM_BW_GBs / 1e3
    ic_line = oi_range * PEAK_IC_BW_GBs / 1e3

    # Composite rooflines (min of slope and compute ceiling)
    bf16_hbm_roof = np.minimum(hbm_line, PEAK_BF16_TFLOPS)
    mxfp4_hbm_roof = np.minimum(hbm_line, PEAK_MXFP4_TFLOPS)
    bf16_ic_roof = np.minimum(ic_line, PEAK_BF16_TFLOPS)
    mxfp4_ic_roof = np.minimum(ic_line, PEAK_MXFP4_TFLOPS)

    # Plot roofline ceilings
    ax.plot(oi_range, mxfp4_ic_roof, "r-", linewidth=2.5,
            label=f"MXFP4 + IC ({PEAK_MXFP4_TFLOPS:,} TFLOPS / {PEAK_IC_BW_GBs//1000} TB/s)")
    ax.plot(oi_range, bf16_ic_roof, "b-", linewidth=2.5,
            label=f"BF16 + IC ({PEAK_BF16_TFLOPS:,} TFLOPS / {PEAK_IC_BW_GBs//1000} TB/s)")
    ax.plot(oi_range, mxfp4_hbm_roof, "r--", linewidth=1.8, alpha=0.5,
            label=f"MXFP4 + HBM ({PEAK_MXFP4_TFLOPS:,} TFLOPS / {PEAK_HBM_BW_GBs//1000} TB/s)")
    ax.plot(oi_range, bf16_hbm_roof, "b--", linewidth=1.8, alpha=0.5,
            label=f"BF16 + HBM ({PEAK_BF16_TFLOPS:,} TFLOPS / {PEAK_HBM_BW_GBs//1000} TB/s)")

    # Reference lines
    ax.axhline(y=PEAK_MXFP4_TFLOPS, color="r", linestyle=":", lw=0.7, alpha=0.3)
    ax.axhline(y=PEAK_BF16_TFLOPS, color="b", linestyle=":", lw=0.7, alpha=0.3)

    # Ridge-point annotations
    for ridge, label, color, va_off in [
        (RIDGE_IC_BF16, f"IC/BF16\nridge {RIDGE_IC_BF16:.0f}", "b", 0.008),
        (RIDGE_IC_MXFP4, f"IC/MXFP4\nridge {RIDGE_IC_MXFP4:.0f}", "r", 0.008),
        (RIDGE_HBM_BF16, f"HBM/BF16\nridge {RIDGE_HBM_BF16:.0f}", "b", 0.015),
        (RIDGE_HBM_MXFP4, f"HBM/MXFP4\nridge {RIDGE_HBM_MXFP4:.0f}", "r", 0.015),
    ]:
        ax.axvline(x=ridge, color=color, linestyle=":", lw=0.6, alpha=0.2)
        ax.text(ridge * 0.8, va_off, label, fontsize=6, color=color,
                alpha=0.5, ha="right", va="bottom")

    # ── Kernel data points ───────────────────────────────────────────────────
    layer_colors = {
        "embedding":                       "#888888",
        "rmsnorm+shortcut (pre-attention)": "#1f77b4",
        "qkv-projection":                  "#ff7f0e",
        "qk rope + kvcache":               "#2ca02c",
        "attention":                        "#d62728",
        "o-project":                        "#9467bd",
        "rmsnorm+shortcut (pre-MoE)":       "#17becf",
        "gating":                           "#bcbd22",
        "quant/sort":                       "#e377c2",
        "moe-stage-1":                      "#ff4500",
        "moe-stage-2":                      "#8b0000",
        "lm-head":                          "#556b2f",
        "sampling":                         "#708090",
    }

    label_offsets = {
        "Cijk GEMM (QKV decode)":          (0.3, 1.6),
        "Cijk GEMM (O-proj decode)":       (0.3, 0.5),
        "bf16gemm (gating GEMM)":          (1.3, 0.4),
        "Cijk GEMM (LM head prefill)":     (1.3, 1.5),
        "MoeFlatmm (gate+up SwiGLU)":      (0.3, 1.4),
        "MoeFlatmm (down proj)":           (0.3, 0.5),
        "add_rmsnorm_quant (decode)":      (1.4, 1.3),
        "fused_qk_rope_reshape_and_cache": (1.5, 1.4),
        "paged_attention_decode":          (1.4, 0.6),
        "paged_attention_ps_reduce":       (0.3, 0.5),
        "topkGatingSoftmax":               (1.3, 1.5),
        "triton_fused_pad_moe":            (0.3, 1.5),
        "MoeSortingKernel":                (1.3, 1.4),
        "mix_sample_outer_exponential":    (0.3, 0.45),
        "kv_indices_generate":             (0.3, 1.5),
    }

    seen_labels = set()
    for r in plot_results:
        oi = float(r["operational_intensity"])
        att = float(r["attained_tflops"])
        label = r["kernel_short_name"]
        prec = r["precision"]
        eff = float(r["efficiency_pct"])
        cl = r["cache_level"]
        color = layer_colors.get(r["model_layer"], "#333333")
        marker = "D" if prec == "mxfp4" else "o"
        size = 140 if prec == "mxfp4" else 90

        dedup_key = f"{label}_{oi:.1f}_{att:.1f}"
        if dedup_key in seen_labels:
            ax.scatter(oi, att, c=color, marker=marker, s=size, zorder=5,
                       edgecolors="black", linewidth=0.5, alpha=0.6)
            continue
        seen_labels.add(dedup_key)

        ax.scatter(oi, att, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="black", linewidth=0.5)

        dx, dy = label_offsets.get(label, (1.3, 1.3))
        cl_tag = "IC" if cl == "ic" else "HBM"
        ax.annotate(
            f"{label}\n({eff:.0f}% {cl_tag})",
            xy=(oi, att), fontsize=6.5, color=color, weight="bold",
            xytext=(oi * dx, att * dy),
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.6),
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(oi_min, oi_max)
    ax.set_ylim(0.005, 20_000)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=14)
    ax.set_ylabel("Attained Performance (TFLOPS)", fontsize=14)
    ax.set_title(
        "Hierarchical Roofline — GPT-OSS 120B (Prefill + Decode) on AMD MI355X\n"
        f"HBM3e = {PEAK_HBM_BW_GBs//1000} TB/s  |  "
        f"Infinity Cache (256 MB) = {PEAK_IC_BW_GBs//1000} TB/s (est.)  |  "
        f"BF16 = {PEAK_BF16_TFLOPS:,} TFLOPS  |  "
        f"MXFP4 = {PEAK_MXFP4_TFLOPS:,} TFLOPS\n"
        f"Attn KV cache assumed avg seq_len = {AVG_SEQ_LEN} (streamed from HBM)",
        fontsize=10.5,
    )

    marker_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label="BF16 kernel"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label="MXFP4 kernel"),
    ]

    roof_handles, roof_labels = ax.get_legend_handles_labels()
    all_handles = roof_handles + marker_legend
    all_labels = roof_labels + [h.get_label() for h in marker_legend]
    ax.legend(all_handles, all_labels, loc="upper left", fontsize=8.5,
              framealpha=0.9, ncol=1)

    ax.grid(True, which="both", alpha=0.12, linestyle="-")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Roofline plot written to {OUTPUT_PNG}")

    # ── Print summary table ──────────────────────────────────────────────────
    hdr = (f"{'Model Layer':<35} {'Kernel':<35} {'Prec':>5} {'Cache':>5} "
           f"{'OI':>8} {'Attained':>10} {'Roof':>10} {'Eff%':>7} {'Bottleneck':<16}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        def fmt(key, f=".1f"):
            v = r[key]
            return "N/A" if v == "N/A" else f"{float(v):{f}}"
        print(
            f"{r['model_layer']:<35} {r['kernel_short_name']:<35} "
            f"{str(r.get('precision','')):>5} {str(r.get('cache_level','')):>5} "
            f"{fmt('operational_intensity'):>8} "
            f"{fmt('attained_tflops', '.2f'):>10} {fmt('roofline_tflops', '.2f'):>10} "
            f"{fmt('efficiency_pct'):>7} {r['bottleneck']:<16}"
        )


if __name__ == "__main__":
    main()
