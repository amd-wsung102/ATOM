import argparse
import ast
import csv
import glob
import gzip
import json
import os
from collections import defaultdict

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, 'mapping_rules.yaml')
DEFAULT_TRACE_DIR = os.path.join(SCRIPT_DIR, 'profiler_traces')


def load_config(config_path):
    """Load mapping rules YAML and compile AMD rules into match-ready form."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gpu_rules = []
    for rule in config.get('amd_gpu_kernel_rules', []):
        substrings = rule['match_all']
        targets = []
        for t in rule['targets']:
            targets.append((
                t['layer'],
                t['short_name'],
                t.get('fraction', 1.0),
            ))
        gpu_rules.append((substrings, targets))

    cpu_rules = []
    for rule in config.get('amd_cpu_op_rules', []):
        cpu_rules.append((rule['pattern'], rule['short_name']))

    gemm_cpu_ops = config.get('amd_gemm_cpu_ops', [])
    layer_order = config.get('layer_order', [])

    return config, gpu_rules, cpu_rules, gemm_cpu_ops, layer_order


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


def _classify_by_cpu_op(op_name, shape_str, cpu_rules, gemm_cpu_ops):
    """Map a CPU operator name (stable across runs) to a kernel short name."""
    for pattern, short in cpu_rules:
        if pattern in op_name:
            return short
    for gop in gemm_cpu_ops:
        if gop in op_name or op_name == gop:
            return _classify_cijk_by_shape(shape_str)
    return None


def extract_from_traces(trace_dir, gpu_rules, cpu_rules, gemm_cpu_ops):
    """Extract kernel timing and shape data from PyTorch profiler traces.

    Reads ALL trace files for timing aggregation (one file per rank).
    Uses the first trace file for shape extraction (shapes are identical
    across ranks — CPU op ``"Input Dims"`` are joined to GPU kernels via
    ``"External id"``).

    Returns (global_agg, kernel_shapes_by_full, shapes_by_short).
    """
    trace_files = sorted(
        glob.glob(os.path.join(trace_dir, '*.pt.trace.json.gz'))
        + glob.glob(os.path.join(trace_dir, '*.pt.trace.json'))
        + glob.glob(os.path.join(trace_dir, 'rank_*', '*.pt.trace.json.gz'))
        + glob.glob(os.path.join(trace_dir, 'rank_*', '*.pt.trace.json'))
    )
    if not trace_files:
        print(f"Warning: No trace files (*.pt.trace.json[.gz]) found in {trace_dir} or rank_*/ subdirs")
        return {}, {}, {}

    print(f"Found {len(trace_files)} trace file(s)")

    global_agg = defaultdict(lambda: {'total_dur': 0.0, 'count': 0})
    ext_id_to_dims = {}
    ext_id_to_op = {}
    kernel_to_ext_id = {}

    for i, trace_file in enumerate(trace_files):
        print(f"  Processing: {os.path.basename(trace_file)}")
        is_first = (i == 0)

        for event in _iter_trace_events(trace_file):
            args = event.get('args')
            if not isinstance(args, dict):
                continue

            if is_first and 'Input Dims' in args:
                ext_id = args.get('External id')
                if ext_id is not None:
                    ext_id_to_dims[ext_id] = str(args['Input Dims'])
                    ext_id_to_op[ext_id] = event.get('name', '')

            if event.get('cat') == 'kernel':
                name = event.get('name', '')
                dur = event.get('dur')
                if name and dur is not None:
                    entry = global_agg[name]
                    entry['total_dur'] += dur
                    entry['count'] += 1
                if is_first:
                    ext_id = args.get('External id')
                    if name and ext_id is not None and name not in kernel_to_ext_id:
                        kernel_to_ext_id[name] = ext_id

    kernel_shapes = {}
    for name, ext_id in kernel_to_ext_id.items():
        if ext_id in ext_id_to_dims:
            kernel_shapes[name] = ext_id_to_dims[ext_id]

    shapes_by_short = {}
    for gpu_name, shape_str in kernel_shapes.items():
        classifications = classify_kernel(gpu_name, gpu_rules)
        layer = classifications[0][0]
        if layer != 'unknown':
            for _, short, _ in classifications:
                shapes_by_short[short] = shape_str
            continue
        ext_id = kernel_to_ext_id[gpu_name]
        cpu_op = ext_id_to_op.get(ext_id, '')
        short = _classify_by_cpu_op(cpu_op, shape_str, cpu_rules, gemm_cpu_ops)
        if short:
            shapes_by_short[short] = shape_str

    print(f"  {len(global_agg)} unique kernels, shapes for {len(shapes_by_short)} short names")
    return global_agg, kernel_shapes, shapes_by_short


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


def classify_kernel(name, gpu_rules):
    """Map GPU kernel name to list of (model_layer, short_name, fraction).

    Rules are loaded from the YAML config's ``amd_gpu_kernel_rules`` section.
    Each rule requires all substrings in ``match_all`` to appear in ``name``.
    """
    for substrings, targets in gpu_rules:
        if all(s in name for s in substrings):
            return targets
    return [('unknown', name[:80], 1.0)]


def main():
    parser = argparse.ArgumentParser(
        description="Map AMD GPU kernels from profiling data to model layers."
    )
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG,
        help='Path to YAML mapping rules (default: mapping_rules.yaml next to this script).',
    )
    parser.add_argument(
        '--trace-dir', default=DEFAULT_TRACE_DIR,
        help='Directory containing *.pt.trace.json[.gz] files (default: profiler_traces/).',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output CSV path (default: <trace-dir>/kernel_analysis.csv).',
    )
    args = parser.parse_args()

    trace_dir = args.trace_dir
    output_path = args.output or os.path.join(trace_dir, 'kernel_analysis.csv')
    shapes_cache = os.path.join(trace_dir, 'kernel_shapes.json')

    _, gpu_rules, cpu_rules, gemm_cpu_ops, layer_order = load_config(args.config)
    print(f"Config:     {args.config}")
    print(f"Trace dir:  {trace_dir}")
    print()

    global_agg, kernel_shapes_by_full, shape_by_short = extract_from_traces(
        trace_dir, gpu_rules, cpu_rules, gemm_cpu_ops,
    )

    if not global_agg:
        print("No kernel data found. Exiting.")
        return

    cached = load_shapes_cache(shapes_cache)
    cached.update(shape_by_short)
    shape_by_short = cached

    if shape_by_short:
        save_shapes_cache(shape_by_short, shapes_cache)

    rows = []
    for name in sorted(global_agg):
        classifications = classify_kernel(name, gpu_rules)
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
            return layer_order.index(layer)
        except ValueError:
            return len(layer_order)

    rows.sort(key=lambda r: (layer_sort_key(r['model_layer']), r['kernel_short_name']))

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_layer', 'kernel_short_name', 'kernel_full_name',
            'instance_count', 'total_time_ms', 'avg_time_ms', 'shape (Input Dims)'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {output_path}")
    print(f"Total rows: {len(rows)}")

    print("\n" + "=" * 90)
    print("SUMMARY: Kernel time per decode step by Model Layer (total across ranks)")
    print("=" * 90)
    for layer in layer_order:
        layer_rows = [r for r in rows if r['model_layer'] == layer]
        if not layer_rows:
            continue
        layer_total_avg = sum(r['avg_time_ms'] for r in layer_rows)
        print(f"\n{layer} — sum of avg kernel times: {layer_total_avg:.3f} ms")
        for r in layer_rows:
            print(f"  {r['kernel_short_name']}: avg={r['avg_time_ms']:.3f} ms, "
                  f"instances={r['instance_count']}, total={r['total_time_ms']:.1f} ms")


if __name__ == '__main__':
    main()
