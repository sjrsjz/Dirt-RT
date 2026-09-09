"""Compare every rule and metric in local GPU capture YAML files.

Usage:
    python tools/compare_gpu_profiles.py temp/analysis_9.yaml \
        temp/analysis_10.yaml temp/analysis_11.yaml --labels 9 10 11 \
        --output-dir temp/bench9/profile

This is a CPU-only reader; it neither compiles shaders nor executes GPU work.
Range names plus occurrence numbers provide a lookup, not proof that shader
roles match. Raw order, missing data, conflicting values and work counts remain
visible. Cycle/thread ratios are elapsed-cycle proxies, never milliseconds or
dynamic instruction counts. Requires PyYAML; reports must stay under temp/.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import yaml

from analyze_gpu_profile import aggregate_evidence, exact_duplicates, metric_values


ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / 'temp'
KEY_METRICS = [
    'GR Cycles Elapsed', 'GR Cycles Active', 'GR Cycles Idle [%]',
    'Threads Launched CS', 'Threads Launched PS', 'Threads Launched VS',
    'Dispatch Started', 'Draw Started',
    'SM Inst Executed [%]', 'SM Pipe ALU [%]', 'SM Pipe FMA Active [%]',
    'SM Pipe FMA Heavy [%]', 'SM Pipe SFU [%]',
    'Active Warps per Cycle All [%]', 'Active Warps per Cycle All',
    'CS Warp Launch Stalled Register Allocation [%]',
    'CS Warp Launch Stalled Shared Memory Allocation [%]',
    'CS Warp Launch Stalled Warp Slot Allocation [%]',
    'CS Warp Launch Stalled Thread Group Slot Allocation [%]',
    'Warps Issue Stalled Long Scoreboard L1 [%]',
    'Warps Issue Stalled Short Scoreboard [%]', 'Warps Issue Stalled Wait [%]',
    'Warps Issue Stalled Math Pipe Throttle [%]',
    'Warps Issue Stalled TEX Throttle [%]', 'Warps Issue Stalled LG Throttle [%]',
    'Warps Issue Stalled MIO Throttle MIO [%]',
    'Warps Issue Stalled Barrier [%]', 'Warps Issue Stalled Membar [%]',
    'Warps Issue Stalled Not Selected [%]',
    'L1TEX TEX Data Wavefronts [%]', 'L1TEX LSU Data Wavefronts [%]',
    'L1TEX LSU Data Wavefronts Shared Mem [%]',
    'L1TEX Filter Stage Wavefronts [%]', 'L1TEX Hit Rate [%]',
    'L2 Hit Rate [%]', 'L2 Throughput [%]', 'VRAM Throughput [%]',
    'L1TEX Tag-Stage Sectors Texture TLD [%]',
    'L1TEX Tag-Stage Sectors Global Load [%]',
    'L1TEX Tag-Stage Sectors Global Store [%]',
    'L1TEX Tag-Stage Sectors Surface Store [%]',
    'L1TEX Miss Sectors Texture TLD [%]',
    'L1TEX Miss Sectors Global Load [%]',
    'L1TEX Miss Sectors Global Store [%]',
    'L1TEX Miss Sectors Surface Store [%]',
    'Cumulative Warp Latency CS', 'Cumulative Warp Latency PS',
    'PCIe Bytes Received by GPU', 'PCIe Bytes Transmitted by GPU',
]

# These are numbered-label groups only. Historical shader roles need external
# source/event evidence, especially when some members are absent or renamed.
LABEL_GROUPS = {
    'labels_51_56_66_71': list(range(51, 57)) + list(range(66, 72)),
    'labels_51_55_66_70': list(range(51, 56)) + list(range(66, 71)),
    'labels_51_53_66_68': [51, 52, 53, 66, 67, 68],
    'labels_54_55_69_70': [54, 55, 69, 70],
    'labels_54_56_69_71': [54, 55, 56, 69, 70, 71],
    'labels_50_65': [50, 65],
    'labels_56_71': [56, 71],
    'labels_58_72': [58, 72],
    'labels_1_3_50_59_61_63_65': [1, 3, 50, 59, 61, 63, 65],
    'labels_90_91_92_93_98_99': [90, 91, 92, 93, 98, 99],
}

LIMITS = [
    'Matching Range plus occurrence is a lookup, not proof of equivalent roles, code, inputs or event scopes.',
    'Original range/rule order and every raw metric occurrence are retained. Missing metrics are not zero.',
    'RelativeFrameDuration is a fraction, not milliseconds; pass_time_ratio = share_ratio * frame_time_ratio.',
    'Elapsed GR cycles cannot establish milliseconds without capture clock/time evidence.',
    'Cycles per launched thread normalize reported launch volume only; they are not per-pixel instructions or useful-work counts.',
    'Mixed CS/graphics scopes and zero dispatch counts with positive CS threads invalidate simple dispatch normalization.',
    'The export does not bind range names to source hashes or pass semantics. Numbered groups are explicitly label groups.',
    'FrameGain and RangeSpeedupFactor are diagnostic predictions and are not used as measured gains.',
    'Warp/launch stalls overlap and cannot be added as exclusive time buckets; shared/register stalls are not resource counts.',
    'Hardware percentages have different domains and traffic mixes. Higher hit rate or occupancy alone is not a speedup.',
    'Cumulative warp latency includes concurrent waiting warps and can rise while elapsed cycles fall.',
    'An aggregate parent and identical duplicated ranges are excluded only using explicit local evidence.',
    'Unclassified frame share is not automatically PT. Same launch counts do not prove equal resolution or renderScale.',
]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(a, b):
    """Return b/a only for unambiguous finite values and a nonzero divisor."""
    return b / a if numeric(a) and numeric(b) and a != 0 else None


def unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def single(values):
    values = unique(values)
    return values[0] if len(values) == 1 else values or None


def metric_occurrences(record):
    by_id = defaultdict(list)
    for ri, rule in enumerate(record.get('Rules', [])):
        for mi, metric in enumerate(rule.get('Metrics', [])):
            by_id[metric['Id']].append({
                'rule_index': ri, 'rule_name': rule['Name'],
                'rule_category': rule.get('Category'), 'metric_index': mi,
                **metric,
            })
    return dict(by_id)


def workload(values):
    cycles = values.get('GR Cycles Elapsed')
    cs = values.get('Threads Launched CS')
    ps = values.get('Threads Launched PS')
    dispatch = values.get('Dispatch Started')
    graphics = {stage: values.get(f'Threads Launched {stage}')
                for stage in ('VS', 'HS', 'DS', 'GS', 'PS')}
    graphics_known_zero = all(numeric(v) and v == 0 for v in graphics.values())
    eligible = numeric(cs) and cs > 0 and graphics_known_zero
    notes = []
    if numeric(cs) and cs > 0 and any(numeric(v) and v > 0 for v in graphics.values()):
        notes.append('mixed_compute_and_graphics_threads')
    if numeric(cs) and cs > 0 and dispatch == 0:
        notes.append('positive_cs_threads_with_zero_reported_dispatches')
    active = values.get('GR Cycles Active')
    if numeric(cycles) and numeric(active) and active > cycles:
        notes.append('active_cycles_exceed_elapsed_counter')
    return {
        'cs_threads': cs, 'ps_threads': ps, 'graphics_threads': graphics,
        'dispatches': dispatch, 'draws': values.get('Draw Started'),
        'gr_cycles_per_cs_thread_proxy': ratio(cs, cycles),
        'gr_cycles_per_ps_thread_proxy': ratio(ps, cycles),
        'exclusive_cs_normalization_eligible': eligible,
        'exclusive_cs_cycles_per_thread': ratio(cs, cycles) if eligible else None,
        'gr_cycles_per_reported_dispatch': ratio(dispatch, cycles)
            if eligible and numeric(dispatch) and dispatch > 0 else None,
        'cumulative_cs_warp_latency_per_cs_thread_proxy':
            ratio(cs, values.get('Cumulative Warp Latency CS')),
        'notes': notes,
    }


def read_profile(path, label):
    data = path.read_bytes()
    records = yaml.load(data.decode('utf-8-sig'), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a flat list of range records')
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('Range'), str):
            raise ValueError(f'{path}: invalid Range at index {index}')
        if not numeric(record.get('RelativeFrameDuration')) or record['RelativeFrameDuration'] < 0:
            raise ValueError(f'{path}: invalid RelativeFrameDuration at index {index}')
    values = [metric_values(r) for r in records]
    duplicates = exact_duplicates(records)
    aggregates = aggregate_evidence(records, values, duplicates)
    for aggregate in aggregates:
        parent = values[aggregate['index']].get('GR Cycles Elapsed')
        children = [values[i].get('GR Cycles Elapsed') for i in aggregate['child_indices']]
        child_sum = sum(children) if all(numeric(v) for v in children) else None
        aggregate['gr_cycle_evidence'] = {
            'parent': parent, 'children_sum': child_sum,
            'parent_minus_children': parent-child_sum
                if numeric(parent) and numeric(child_sum) else None,
        }
    excluded = set(duplicates) | {a['index'] for a in aggregates}
    seen = Counter()
    ranges = []
    explicit_dimensions_or_clocks = []
    for index, (record, metrics) in enumerate(zip(records, values)):
        occurrence = seen[record['Range']]
        seen[record['Range']] += 1
        occurrences = metric_occurrences(record)
        ranges.append({
            'index': index, 'range': record['Range'], 'occurrence': occurrence,
            'counted_as_leaf': index not in excluded,
            'duplicate_of_index': duplicates.get(index),
            'frame_percent': 100 * record['RelativeFrameDuration'],
            'metrics_by_name': metrics, 'metrics_by_id': occurrences,
            'workload': workload(metrics), 'raw_record': record,
        })
        for name, value in metrics.items():
            if re.search(r'resolution|render.?scale|viewport|\bwidth\b|\bheight\b|frequency|clock', name, re.I):
                explicit_dimensions_or_clocks.append({'range_index': index, 'name': name, 'value': value})
    covered = math.fsum(r['RelativeFrameDuration'] for i, r in enumerate(records) if i not in excluded)
    return {
        'label': label, 'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
        'range_count': len(records), 'rule_count': sum(len(r.get('Rules', [])) for r in records),
        'metric_occurrence_count': sum(len(m.get('Metrics', [])) for r in records for m in r.get('Rules', [])),
        'duplicate_indices': duplicates, 'aggregate_evidence': aggregates,
        'covered_leaf_frame_percent': 100 * covered,
        'unclassified_frame_percent': 100 * (1 - covered),
        'explicit_dimension_or_clock_metric_matches': explicit_dimensions_or_clocks,
        'ranges_in_original_order': ranges,
    }


def changes(values, labels):
    pairs = unique([(labels[0], b) for b in labels[1:]] + list(zip(labels, labels[1:])))
    return {f'{a}->{b}': {
        'before': values.get(a), 'after': values.get(b),
        'difference': values[b]-values[a]
            if numeric(values.get(a)) and numeric(values.get(b)) else None,
        'ratio': ratio(values.get(a), values.get(b)),
    } for a, b in pairs}


def rule_keys(record):
    seen = Counter()
    result = {}
    for index, rule in enumerate(record['raw_record'].get('Rules', [])):
        prefix = (rule['Name'], rule.get('Category'))
        key = (*prefix, seen[prefix])
        seen[prefix] += 1
        result[key] = index
    return result


def compare_ranges(profiles):
    labels = [p['label'] for p in profiles]
    lookups = [{(r['range'], r['occurrence']): r for r in p['ranges_in_original_order']} for p in profiles]
    order = unique([key for lookup in lookups for key in lookup])
    aligned = []
    for name, occurrence in order:
        selected = {label: lookup.get((name, occurrence)) for label, lookup in zip(labels, lookups)}
        all_ids = sorted({mid for row in selected.values() if row for mid in row['metrics_by_id']})
        by_id = {}
        for mid in all_ids:
            occ = {label: row['metrics_by_id'].get(mid, []) if row else [] for label, row in selected.items()}
            vals = {label: single([v.get('Value') for v in items]) for label, items in occ.items()}
            by_id[mid] = {
                'names': unique([v['Name'] for items in occ.values() for v in items]),
                'values': vals, 'changes': changes(vals, labels),
                'occurrence_references': {label: [{'rule_index': v['rule_index'], 'metric_index': v['metric_index']}
                                               for v in items] for label, items in occ.items()},
            }
        rule_maps = {label: rule_keys(row) if row else {} for label, row in selected.items()}
        all_rule_keys = unique([key for mapping in rule_maps.values() for key in mapping])
        shares = {label: row['frame_percent'] if row else None for label, row in selected.items()}
        work_values = {label: row['workload'] if row else None for label, row in selected.items()}
        normalized = {key: changes({label: w.get(key) if w else None for label, w in work_values.items()}, labels)
                      for key in ['gr_cycles_per_cs_thread_proxy', 'gr_cycles_per_ps_thread_proxy',
                                  'exclusive_cs_cycles_per_thread', 'cumulative_cs_warp_latency_per_cs_thread_proxy']}
        aligned.append({
            'range': name, 'occurrence': occurrence,
            'alignment': 'label_and_occurrence_only; semantic_equivalence_unverified',
            'source_indices': {label: row['index'] if row else None for label, row in selected.items()},
            'missing_from': [label for label, row in selected.items() if row is None],
            'frame_percent': shares, 'frame_percent_changes': changes(shares, labels),
            'workload': work_values, 'normalized_changes': normalized,
            'metrics_by_id': by_id,
            'rule_alignment': [{'name': key[0], 'category': key[1], 'occurrence': key[2],
                                'source_rule_indices': {label: mapping.get(key) for label, mapping in rule_maps.items()}}
                               for key in all_rule_keys],
        })
    return aligned


def groups(profiles):
    result = {}
    for name, numbers in LABEL_GROUPS.items():
        expected = [f'composite{n}' for n in numbers]
        snapshots = {}
        for profile in profiles:
            members = [r for r in profile['ranges_in_original_order'] if r['range'] in expected and r['counted_as_leaf']]
            row = {'member_labels': [r['range'] for r in members],
                   'missing_labels': [n for n in expected if n not in {r['range'] for r in members}],
                   'semantic_equivalence': 'unverified',
                   'frame_percent': math.fsum(r['frame_percent'] for r in members)}
            for metric in ['GR Cycles Elapsed', 'Threads Launched CS', 'Threads Launched PS', 'Dispatch Started', 'Draw Started']:
                vals = [r['metrics_by_name'].get(metric) for r in members]
                row[metric] = sum(vals) if vals and all(numeric(v) for v in vals) else None
            row['gr_cycles_per_cs_thread_proxy'] = ratio(row['Threads Launched CS'], row['GR Cycles Elapsed'])
            row['exclusive_cs_normalization_eligible'] = bool(members) and all(r['workload']['exclusive_cs_normalization_eligible'] for r in members)
            snapshots[profile['label']] = row
        result[name] = {'requested_labels': expected, 'captures': snapshots}
    return result


def csv_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def write_reports(report, output):
    labels = [p['label'] for p in report['profiles']]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison_all_rules.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    pairs = list(changes(dict.fromkeys(labels), labels))
    fields = ['range', 'occurrence', 'metric_id', 'metric_names'] + labels + [f'{p} {v}' for p in pairs for v in ('difference', 'ratio')]
    with (output / 'all_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['aligned_ranges']:
            for mid, metric in row['metrics_by_id'].items():
                data = {'range': row['range'], 'occurrence': row['occurrence'], 'metric_id': mid,
                        'metric_names': '; '.join(metric['names']), **metric['values']}
                data.update({f'{pair} {key}': value[key] for pair, value in metric['changes'].items() for key in ('difference', 'ratio')})
                writer.writerow({k: csv_value(v) for k, v in data.items()})
    work_keys = ['exclusive_cs_normalization_eligible', 'gr_cycles_per_cs_thread_proxy',
                 'gr_cycles_per_ps_thread_proxy', 'cumulative_cs_warp_latency_per_cs_thread_proxy', 'notes']
    fields = ['label', 'index', 'range', 'occurrence', 'counted_as_leaf', 'frame_percent'] + KEY_METRICS + work_keys
    with (output / 'pass_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p in report['profiles']:
            for r in p['ranges_in_original_order']:
                data = {'label': p['label'], **{k: r[k] for k in fields[1:6]},
                        **{k: r['metrics_by_name'].get(k) for k in KEY_METRICS},
                        **{k: r['workload'][k] for k in work_keys}}
                writer.writerow({k: csv_value(v) for k, v in data.items()})
    lines = ['# GPU capture comparison', '',
             'All Rules and metric occurrences are retained in `comparison_all_rules.json`. '
             '`all_metrics.csv` contains every metric ID; `pass_summary.csv` preserves original range order.', '',
             'Range alignment is by label plus occurrence only; it does not establish historical shader equivalence.', '',
             '| Capture | Ranges | Rules | Metric occurrences | Counted leaf share |', '| --- | ---: | ---: | ---: | ---: |']
    for p in report['profiles']:
        lines.append(f"| {p['label']} | {p['range_count']} | {p['rule_count']} | {p['metric_occurrence_count']} | {p['covered_leaf_frame_percent']:.6f}% |")
    lines += ['', '## Elapsed cycles normalized by reported CS launches', '',
              'Each cell is `GR cycles / CS threads`. Mixed graphics/compute scopes are marked `mixed`; '
              'this proxy is not milliseconds, instructions per pixel, or proof of equivalent useful work.', '',
              '| Range | ' + ' | '.join(labels) + ' |', '| --- | ' + ' | '.join('---:' for _ in labels) + ' |']
    for row in report['aligned_ranges']:
        if not re.fullmatch(r'composite\d+', row['range']):
            continue
        cells = []
        for label in labels:
            w = row['workload'][label]
            value = w.get('gr_cycles_per_cs_thread_proxy') if w else None
            cells.append('—' if value is None else f"{value:.6f}" + ('' if w['exclusive_cs_normalization_eligible'] else ' mixed'))
        lines.append('| ' + row['range'] + (f" #{row['occurrence']+1}" if row['occurrence'] else '') + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Interpretation limits', ''] + ['- '+limit for limit in LIMITS]
    (output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', type=Path, nargs='+', help='Two or more local exported YAML files, in comparison order')
    parser.add_argument('--labels', nargs='+', help='Unique capture labels; defaults to input filename stems')
    parser.add_argument('--output-dir', type=Path, required=True, help='Report folder beneath the project temp/')
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error('at least two input captures are required')
    labels = args.labels or [p.stem for p in args.inputs]
    if len(labels) != len(args.inputs) or len(set(labels)) != len(labels):
        parser.error('labels must be unique and have the same count as inputs')
    output = args.output_dir.resolve()
    if not output.is_relative_to(TEMP.resolve()) or output == TEMP.resolve():
        parser.error('--output-dir must be a subdirectory beneath the project temp/')
    profiles = [read_profile(path.resolve(), label) for path, label in zip(args.inputs, labels)]
    ids_by_name = defaultdict(set)
    for profile in profiles:
        for row in profile['ranges_in_original_order']:
            for mid, occurrences in row['metrics_by_id'].items():
                for metric in occurrences:
                    ids_by_name[metric['Name']].add(mid)
    report = {'schema_version': 1, 'interpretation_limits': LIMITS,
              'display_names_with_multiple_metric_ids': {
                  name: sorted(ids) for name, ids in ids_by_name.items() if len(ids) > 1},
              'profiles': profiles, 'aligned_ranges': compare_ranges(profiles),
              'numbered_label_groups': groups(profiles)}
    write_reports(report, output)
    print(json.dumps({'output_dir': str(output), 'captures': [{k: p[k] for k in
        ['label', 'range_count', 'rule_count', 'metric_occurrence_count', 'covered_leaf_frame_percent']} for p in profiles]}, indent=2))


if __name__ == '__main__':
    main()
