import re
import csv
import os
import ast
import gzip
import glob
import json
from collections import defaultdict

RANKS = list(range(8))
EXTRACT_DIR = '/tmp/kernel_extract'
TRACE_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPES_CACHE = os.path.join(TRACE_DIR, 'kernel_shapes.json')


def _iter_trace_events(trace_file):
    """Stream events from a Chrome trace JSON file without loading it all into memory.

    Handles both formats:
      - Array-only:  [{event}, ...]         → events at brace depth 1
      - Wrapped:     {"traceEvents": [...]} → events at brace depth 2

    Only yields events containing "Input Dims" or "cat": "kernel" (the two
    kinds needed for shape extraction), skipping everything else for speed.
    """
    opener = gzip.open if trace_file.endswith('.gz') else open
    with opener(trace_file, 'rt') as f:
        first_char = None
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break

        event_depth = 1 if first_char == '[' else 2
        f.seek(0)

        depth = 0
        buf = []
        in_str = False
        esc = False

        for line in f:
            for ch in line:
                if esc:
                    if buf:
                        buf.append(ch)
                    esc = False
                    continue
                if in_str:
                    if buf:
                        buf.append(ch)
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    if buf:
                        buf.append(ch)
                    continue
                if ch == '{':
                    depth += 1
                    if depth == event_depth:
                        buf = ['{']
                    elif buf:
                        buf.append(ch)
                elif ch == '}':
                    if buf:
                        buf.append(ch)
                    if depth == event_depth:
                        event_str = ''.join(buf)
                        buf = []
                        if '"Input Dims"' in event_str or '"cat": "kernel"' in event_str:
                            try:
                                yield json.loads(event_str)
                            except json.JSONDecodeError:
                                pass
                    depth -= 1
                elif buf:
                    buf.append(ch)


def _classify_cijk_by_shape(shape_str):
    """Distinguish Cijk GEMM variants (QKV / O-proj / LM-head / gating) by dims."""
    try:
        dims = ast.literal_eval(shape_str)
    except (ValueError, SyntaxError):
        return None
    flat = set()
    for d in dims:
        if isinstance(d, list):
            for v in d:
                if isinstance(v, int):
                    flat.add(v)
    if 25136 in flat:
        return 'Cijk GEMM (LM head prefill)'
    for d in dims:
        if isinstance(d, list) and len(d) == 2:
            if 640 in d:
                return 'Cijk GEMM (QKV decode)'
            if 512 in d:
                return 'Cijk GEMM (O-proj decode)'
            if 128 in d and d[0] != d[1]:
                return 'bf16gemm (gating GEMM)'
    return None


def _classify_by_cpu_op(op_name, shape_str):
    """Map a CPU operator name (stable across runs) to a kernel short name."""
    _PATTERNS = [
        ('masked_embedding', 'masked_embedding'),
        ('add_rmsnorm', 'add_rmsnorm_quant (decode)'),
        ('rmsnorm2d_fwd_with_add', 'add_rmsnorm_quant (decode)'),
        ('rmsnorm2d_fwd_', 'add_rmsnorm_quant (prefill)'),
        ('fused_qk_rope', 'fused_qk_rope_reshape_and_cache'),
        ('unified_attention', 'sliding_window_attention'),
        ('mha_varlen', 'sliding_window_attention'),
        ('topk_softmax', 'topkGatingSoftmax'),
        ('moe_sorting', 'MoeSortingKernel'),
        ('moe_cktile2stages_gemm1', 'MoeFlatmm (gate+up SwiGLU)'),
        ('fused_moe_', 'MoeFlatmm (gate+up SwiGLU)'),
        ('moe_cktile2stages_gemm2', 'MoeFlatmm (down proj)'),
        ('cross_device_reduce', 'cross_device_reduce_1stage'),
        ('all_reduce', 'cross_device_reduce_1stage'),
        ('allgather', 'allgather_lastdim'),
        ('mix_sample', 'mix_sample_outer_exponential'),
        ('triton_poi_fused', 'triton_fused_pad_moe'),
    ]
    for pattern, short in _PATTERNS:
        if pattern in op_name:
            return short
    if 'gemm_a16w16' in op_name or op_name in ('aten::addmm', 'aten::matmul'):
        return _classify_cijk_by_shape(shape_str)
    return None


def extract_shapes_from_traces(trace_dir):
    """Parse a PyTorch profiler trace to build a short-name → Input Dims mapping.

    CPU op events carry ``"Input Dims"``, ``"External id"``, and a stable
    operator name (e.g. ``aiter::gemm_a16w16``).  GPU kernel events carry the
    same ``"External id"``, linking them to the parent CPU op.

    We join on External id and classify each kernel — first by GPU kernel name,
    then by CPU operator name — so shapes are matched correctly even when GPU
    kernel names change between profiling runs (due to autotuning, etc.).

    Only the first trace file found is parsed (shapes are identical across ranks).
    """
    trace_files = sorted(
        glob.glob(os.path.join(trace_dir, '*.pt.trace.json.gz'))
        + glob.glob(os.path.join(trace_dir, '*.pt.trace.json'))
        + glob.glob(os.path.join(trace_dir, 'rank_*', '*.pt.trace.json.gz'))
        + glob.glob(os.path.join(trace_dir, 'rank_*', '*.pt.trace.json'))
    )
    if not trace_files:
        print(f"Warning: No trace files (*.pt.trace.json[.gz]) found in {trace_dir} or rank_*/ subdirs")
        print("  Shapes will be empty. Re-run profiling to generate trace files.")
        return {}, {}

    trace_file = trace_files[0]
    print(f"Extracting shapes from: {trace_file}")

    ext_id_to_dims = {}
    ext_id_to_op = {}
    kernel_to_ext_id = {}

    for event in _iter_trace_events(trace_file):
        args = event.get('args')
        if not isinstance(args, dict):
            continue

        if 'Input Dims' in args:
            ext_id = args.get('External id')
            if ext_id is not None:
                ext_id_to_dims[ext_id] = str(args['Input Dims'])
                ext_id_to_op[ext_id] = event.get('name', '')

        if event.get('cat') == 'kernel':
            name = event.get('name', '')
            ext_id = args.get('External id')
            if name and ext_id is not None and name not in kernel_to_ext_id:
                kernel_to_ext_id[name] = ext_id

    kernel_shapes = {}
    for name, ext_id in kernel_to_ext_id.items():
        if ext_id in ext_id_to_dims:
            kernel_shapes[name] = ext_id_to_dims[ext_id]

    shapes_by_short = {}
    for gpu_name, shape_str in kernel_shapes.items():
        classifications = classify_kernel(gpu_name)
        layer = classifications[0][0]
        if layer != 'unknown':
            for _, short, _ in classifications:
                shapes_by_short[short] = shape_str
            continue
        ext_id = kernel_to_ext_id[gpu_name]
        cpu_op = ext_id_to_op.get(ext_id, '')
        short = _classify_by_cpu_op(cpu_op, shape_str)
        if short:
            shapes_by_short[short] = shape_str

    matched = len(shapes_by_short)
    total = len(kernel_to_ext_id)
    print(f"  Extracted shapes for {matched} short names from {total} unique kernel types")
    return kernel_shapes, shapes_by_short


def load_shapes_cache(path):
    """Load cached shapes keyed by kernel short name from a JSON file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_shapes_cache(cache, path):
    """Persist the short-name → shape mapping to JSON for future runs."""
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    print(f"Shapes cache written to: {path}")


def classify_kernel(name):
    """Map GPU kernel name to list of (model_layer, short_name, fraction)."""

    if 'rmsnorm' in name and 'true' in name:
        return [
            ('rmsnorm+shortcut (pre-attention)', 'add_rmsnorm_quant (decode)', 0.5),
            ('rmsnorm+shortcut (pre-MoE)', 'add_rmsnorm_quant (decode)', 0.5),
        ]
    if 'rmsnorm' in name and 'false' in name:
        return [('rmsnorm+shortcut (pre-attention)', 'add_rmsnorm_quant (prefill)', 1.0)]

    if 'Cijk_Alik_Bljk' in name and 'DTLA1_DTLB1' in name:
        return [('qkv-projection', 'Cijk GEMM (QKV decode)', 1.0)]
    if 'Cijk_Alik_Bljk' in name and 'MT16x16x256' in name and 'EPS1' in name:
        return [('o-project', 'Cijk GEMM (O-proj decode)', 1.0)]
    if 'Cijk_Alik_Bljk' in name and 'MT256x16x64' in name:
        return [('lm-head', 'Cijk GEMM (LM head prefill)', 1.0)]

    if '_fused_qk_rope_reshape_and_cache_kernel' in name:
        return [('qk rope + kvcache', 'fused_qk_rope_reshape_and_cache', 1.0)]

    if 'attention' in name and 'window' in name:
        return [('attention', 'sliding_window_attention', 1.0)]
    if 'attention' in name and 'reduce' in name:
        return [('attention', 'attention_reduce', 1.0)]

    if 'bf16gemm_fp32bf16_tn' in name:
        return [('gating', 'bf16gemm (gating GEMM)', 1.0)]

    if 'topkGatingSoftmax' in name:
        return [('gating', 'topkGatingSoftmax', 1.0)]

    if 'MoeSortingKernel' in name:
        return [('quant/sort', 'MoeSortingKernel', 1.0)]
    if 'triton_poi_fused_constant_pad_nd_moe_forward' in name:
        return [('gating', 'triton_fused_pad_moe', 1.0)]

    if 'MoeFlatmmKernel' in name and 'Swiglu' in name:
        return [('moe-stage-1', 'MoeFlatmm (gate+up SwiGLU)', 1.0)]
    if 'MoeFlatmmKernel' in name and 'MoeSilu' in name:
        return [('moe-stage-2', 'MoeFlatmm (down proj)', 1.0)]

    if 'mscclKernel_Sum' in name:
        return [('nccl-allreduce', 'mscclKernel_Sum', 1.0)]
    if 'cross_device_reduce_1stage' in name:
        return [('nccl-allreduce', 'cross_device_reduce_1stage', 1.0)]

    if 'allgather_lastdim' in name:
        return [('output-allgather', 'allgather_lastdim', 1.0)]
    if 'mix_sample_outer_exponential' in name:
        return [('sampling', 'mix_sample_outer_exponential', 1.0)]
    if 'ncclDevKernel_Generic' in name:
        return [('nccl-allreduce', 'ncclDevKernel_Generic_1', 1.0)]
    if 'kv_indices_generate_kernel' in name:
        return [('attention', 'kv_indices_generate', 1.0)]
    if '_masked_embedding_kernel' in name:
        return [('embedding', 'masked_embedding', 1.0)]
    if '__amd_rocclr_fillBufferAligned' in name:
        return [('memcopy', 'fillBufferAligned', 1.0)]
    if '__amd_rocclr_copyBuffer' in name:
        return [('memcopy', 'copyBuffer', 1.0)]

    return [('unknown', name[:80], 1.0)]


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

kernel_shapes_by_full, shape_by_short = extract_shapes_from_traces(TRACE_DIR)

# Merge with any previously cached shapes (trace extraction takes priority)
cached = load_shapes_cache(SHAPES_CACHE)
cached.update(shape_by_short)
shape_by_short = cached

if shape_by_short:
    save_shapes_cache(shape_by_short, SHAPES_CACHE)

rows = []
for name in sorted(kernel_names_set):
    classifications = classify_kernel(name)
    info = global_agg[name]

    shape = kernel_shapes_by_full.get(name, '')
    if not shape:
        short_name_key = classifications[0][1]
        shape = shape_by_short.get(short_name_key, '')

    for layer, short_name, fraction in classifications:
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
