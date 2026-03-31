import re
import json
import csv
import os
from collections import defaultdict

def parse_kernel_file(filepath):
    """Parse extracted kernel events from grep output."""
    kernels = []
    current_name = None
    current_dur = None
    current_ext_id = None
    current_args_line = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == '--':
                if current_name is not None and current_dur is not None:
                    kernels.append({
                        'name': current_name,
                        'dur': current_dur,
                        'ext_id': current_ext_id,
                        'args_line': current_args_line,
                    })
                current_name = None
                current_dur = None
                current_ext_id = None
                current_args_line = None
                continue
            
            name_match = re.search(r'"name":\s*"([^"]*)"', line)
            if name_match and '"cat": "kernel"' in line:
                current_name = name_match.group(1)
            
            dur_match = re.search(r'"dur":\s*([\d.]+)', line)
            if dur_match:
                current_dur = float(dur_match.group(1))
            
            ext_id_match = re.search(r'"External id":\s*(\d+)', line)
            if ext_id_match:
                current_ext_id = int(ext_id_match.group(1))
                current_args_line = line
    
    if current_name is not None and current_dur is not None:
        kernels.append({
            'name': current_name,
            'dur': current_dur,
            'ext_id': current_ext_id,
            'args_line': current_args_line,
        })
    
    return kernels


def aggregate_kernels(kernels):
    """Aggregate kernels by name: total_dur, count, sample ext_id."""
    agg = defaultdict(lambda: {'total_dur': 0.0, 'count': 0, 'sample_ext_id': None})
    for k in kernels:
        entry = agg[k['name']]
        entry['total_dur'] += k['dur']
        entry['count'] += 1
        if entry['sample_ext_id'] is None:
            entry['sample_ext_id'] = k['ext_id']
    return agg


print("Parsing rank_0...")
kernels_0 = parse_kernel_file('/tmp/kernel_extract/rank_0_kernels.txt')
print(f"  Found {len(kernels_0)} kernel events")

agg_0 = aggregate_kernels(kernels_0)
print(f"  Unique kernel names: {len(agg_0)}")

print("\nTop 30 kernels by total duration:")
sorted_kernels = sorted(agg_0.items(), key=lambda x: x[1]['total_dur'], reverse=True)
for name, info in sorted_kernels[:30]:
    avg = info['total_dur'] / info['count']
    short_name = name[:100] + '...' if len(name) > 100 else name
    print(f"  {short_name}")
    print(f"    count={info['count']}, total_dur={info['total_dur']:.3f}ms, avg={avg:.3f}ms, ext_id={info['sample_ext_id']}")

print(f"\nAll {len(agg_0)} unique kernel names:")
for name, info in sorted_kernels:
    avg = info['total_dur'] / info['count']
    short_name = name[:120] + '...' if len(name) > 120 else name
    print(f"  [{info['count']:6d}x] avg={avg:.6f}ms  {short_name}")
