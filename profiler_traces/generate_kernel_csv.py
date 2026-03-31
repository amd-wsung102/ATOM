import re
import csv
import os
from collections import defaultdict

RANKS = list(range(8))
EXTRACT_DIR = '/tmp/kernel_extract'

# Raw Input Dims extracted directly from the trace file CPU ops
RAW_SHAPES = {
    'add_rmsnorm': '[[4096, 2880], [4096, 2880], [4096, 2880], [4096, 2880], [2880], []]',
    'gemm_qkv': '[[4096, 2880], [640, 2880], [640], [], [], [], []]',
    'gemm_oproj': '[[4096, 512], [2880, 512], [2880], [], [], [], []]',
    'gemm_gating': '[[4096, 2880], [128, 2880], [128], [], [], [], []]',
    'topk_softmax': '[[4096, 4], [4096, 4], [4096, 4], [4096, 128], [], [], []]',
    'moe_sorting': '[[4096, 4], [4096, 4], [20476], [20476], [640], [2], [4096, 3072], [], [], [], [], []]',
    'moe_gemm1': '[[4096, 3072], [128, 1024, 1536], [4096, 4, 512], [20476], [640], [2], [], [], [], [], [20476, 96], [131072, 96], [128, 1024], [], [], [], []]',
    'moe_gemm2': '[[4096, 4, 512], [128, 3072, 256], [4096, 3072], [20476], [640], [2], [], [], [], [20476], [20476, 96], [393216, 16], [128, 3072], [], [], [], []]',
    'attention': '[[4096, 512], [], [4096, 64], [4096, 64], [4096], [], [], []]',
    'all_reduce': '[[1], [], [4096, 2880], [4096, 2880], [], [], [134217728], [134217728]]',
    'all_gather': '[[1], [], [4, 25136], [134217728], [4, 201088], [], []]',
    'masked_embedding': '[[4096], [25136, 2880], [], []]',
    'mm_prefill': '[[4, 2880], [2880, 25136]]',
    'mix_sample': '[[4], [4, 201088], [4, 201088], [4], []]',
}


def classify_kernel(name):
    """Map GPU kernel name to list of (model_layer, short_name, raw_shape, fraction)."""

    if 'add_rmsnorm_quant_kernel' in name and 'true' in name:
        return [
            ('rmsnorm+shortcut (pre-attention)', 'add_rmsnorm_quant (decode)', RAW_SHAPES['add_rmsnorm'], 0.5),
            ('rmsnorm+shortcut (pre-MoE)', 'add_rmsnorm_quant (decode)', RAW_SHAPES['add_rmsnorm'], 0.5),
        ]
    if 'add_rmsnorm_quant_kernel' in name and 'false' in name:
        return [('rmsnorm+shortcut (pre-attention)', 'add_rmsnorm_quant (prefill)', RAW_SHAPES['add_rmsnorm'], 1.0)]

    if 'Cijk_Alik_Bljk' in name and 'DTLA1_DTLB1' in name:
        return [('qkv-projection', 'Cijk GEMM (QKV decode)', RAW_SHAPES['gemm_qkv'], 1.0)]
    if 'Cijk_Alik_Bljk' in name and 'MT16x16x256' in name and 'EPS1' in name:
        return [('o-project', 'Cijk GEMM (O-proj decode)', RAW_SHAPES['gemm_oproj'], 1.0)]
    if 'Cijk_Alik_Bljk' in name and 'MT256x16x64' in name:
        return [('lm-head', 'Cijk GEMM (LM head prefill)', RAW_SHAPES['mm_prefill'], 1.0)]

    if '_fused_qk_rope_reshape_and_cache_kernel' in name:
        return [('qk rope + kvcache', 'fused_qk_rope_reshape_and_cache', '', 1.0)]

    if 'paged_attention_decode_sliding_window' in name:
        return [('attention', 'paged_attention_decode', RAW_SHAPES['attention'], 1.0)]
    if 'paged_attention_decode_ps_reduce' in name:
        return [('attention', 'paged_attention_ps_reduce', '', 1.0)]

    if 'bf16gemm_fp32bf16_tn' in name:
        return [('gating', 'bf16gemm (gating GEMM)', RAW_SHAPES['gemm_gating'], 1.0)]

    if 'topkGatingSoftmax' in name:
        return [('gating', 'topkGatingSoftmax', RAW_SHAPES['topk_softmax'], 1.0)]

    if 'MoeSortingKernel' in name:
        return [('quant/sort', 'MoeSortingKernel', RAW_SHAPES['moe_sorting'], 1.0)]
    if 'triton_poi_fused_constant_pad_nd_moe_forward' in name:
        return [('gating', 'triton_fused_pad_moe', '', 1.0)]

    if 'MoeFlatmmKernel' in name and 'Swiglu' in name:
        return [('moe-stage-1', 'MoeFlatmm (gate+up SwiGLU)', RAW_SHAPES['moe_gemm1'], 1.0)]
    if 'MoeFlatmmKernel' in name and 'MoeSilu' in name:
        return [('moe-stage-2', 'MoeFlatmm (down proj)', RAW_SHAPES['moe_gemm2'], 1.0)]

    if 'mscclKernel_Sum' in name:
        return [('nccl-allreduce', 'mscclKernel_Sum', '', 1.0)]
    if 'cross_device_reduce_1stage' in name:
        return [('nccl-allreduce', 'cross_device_reduce_1stage', RAW_SHAPES['all_reduce'], 1.0)]

    if 'allgather_lastdim' in name:
        return [('output-allgather', 'allgather_lastdim', RAW_SHAPES['all_gather'], 1.0)]
    if 'mix_sample_outer_exponential' in name:
        return [('sampling', 'mix_sample_outer_exponential', RAW_SHAPES['mix_sample'], 1.0)]
    if 'ncclDevKernel_Generic' in name:
        return [('nccl-allreduce', 'ncclDevKernel_Generic_1', '', 1.0)]
    if 'kv_indices_generate_kernel' in name:
        return [('attention', 'kv_indices_generate', '', 1.0)]
    if '_masked_embedding_kernel' in name:
        return [('embedding', 'masked_embedding', RAW_SHAPES['masked_embedding'], 1.0)]
    if '__amd_rocclr_fillBufferAligned' in name:
        return [('memcopy', 'fillBufferAligned', '', 1.0)]
    if '__amd_rocclr_copyBuffer' in name:
        return [('memcopy', 'copyBuffer', '', 1.0)]

    return [('unknown', name[:80], '', 1.0)]


def parse_kernel_file(filepath):
    kernels = []
    current_name = None
    current_dur = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == '--':
                if current_name is not None and current_dur is not None:
                    kernels.append((current_name, current_dur))
                current_name = None
                current_dur = None
                continue

            name_match = re.search(r'"name":\s*"([^"]*)"', line)
            if name_match and '"cat": "kernel"' in line:
                current_name = name_match.group(1)

            dur_match = re.search(r'"dur":\s*([\d.]+)', line)
            if dur_match:
                current_dur = float(dur_match.group(1))

    if current_name is not None and current_dur is not None:
        kernels.append((current_name, current_dur))

    return kernels


def aggregate_kernels(kernels):
    agg = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
    for name, dur in kernels:
        entry = agg[name]
        entry['total_dur'] += dur
        entry['count'] += 1
    return agg


# Parse all ranks
all_rank_data = {}
for rank in RANKS:
    filepath = os.path.join(EXTRACT_DIR, f'rank_{rank}_kernels.txt')
    print(f"Parsing rank_{rank}...")
    kernels = parse_kernel_file(filepath)
    agg = aggregate_kernels(kernels)
    all_rank_data[rank] = agg
    print(f"  {len(kernels)} events, {len(agg)} unique kernels")

kernel_names_set = set()
for rank_data in all_rank_data.values():
    kernel_names_set.update(rank_data.keys())

# Aggregate across all ranks: sum instances and total_time, then avg = total / instances
global_agg = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
for rank_data in all_rank_data.values():
    for name, info in rank_data.items():
        global_agg[name]['total_dur'] += info['total_dur']
        global_agg[name]['count'] += info['count']

LAYER_ORDER = [
    'embedding',
    'rmsnorm+shortcut (pre-attention)',
    'qkv-projection',
    'qk rope + kvcache',
    'attention',
    'o-project',
    'nccl-allreduce',
    'rmsnorm+shortcut (pre-MoE)',
    'gating',
    'quant/sort',
    'moe-stage-1',
    'moe-stage-2',
    'lm-head',
    'output-allgather',
    'sampling',
    'memcopy',
    'unknown',
]

rows = []
for name in sorted(kernel_names_set):
    classifications = classify_kernel(name)
    info = global_agg[name]

    for layer, short_name, shape, fraction in classifications:
        effective_count = int(info['count'] * fraction)
        effective_total = info['total_dur'] * fraction
        avg_ms = effective_total / effective_count if effective_count > 0 else 0.0
        rows.append({
            'model_layer': layer,
            'kernel_short_name': short_name,
            'kernel_full_name': name,
            'instance_count': effective_count,
            'total_time_ms': round(effective_total, 6),
            'avg_time_ms': round(avg_ms, 6),
            'shape (Input Dims)': shape,
        })


def layer_sort_key(layer):
    try:
        return LAYER_ORDER.index(layer)
    except ValueError:
        return len(LAYER_ORDER)


rows.sort(key=lambda r: (layer_sort_key(r['model_layer']), r['kernel_short_name']))

output_path = '/app/ATOM/profiler_traces/gpt_oss_120b_kernel_analysis.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'model_layer', 'kernel_short_name', 'kernel_full_name',
        'instance_count', 'total_time_ms', 'avg_time_ms', 'shape (Input Dims)'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV written to: {output_path}")
print(f"Total rows: {len(rows)}")

print("\n" + "="*90)
print("SUMMARY: Kernel time per decode step by Model Layer (total across 8 ranks)")
print("="*90)
for layer in LAYER_ORDER:
    layer_rows = [r for r in rows if r['model_layer'] == layer]
    if not layer_rows:
        continue
    layer_total_avg = sum(r['avg_time_ms'] for r in layer_rows)
    print(f"\n{layer} — sum of avg kernel times: {layer_total_avg:.3f} ms")
    for r in layer_rows:
        print(f"  {r['kernel_short_name']}: avg={r['avg_time_ms']:.3f} ms, "
              f"instances={r['instance_count']}, total={r['total_time_ms']:.1f} ms")
