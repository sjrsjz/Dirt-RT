"""Summarize a local Nsight GPU profile without counting parent ranges twice.

Range durations are exported fractions of the original frame, not milliseconds.
Rule FrameGain/RangeSpeedupFactor values are diagnostic model estimates; they
are retained in the raw records but never used as measured optimization gains.
Requires PyYAML. All generated reports must remain under the project temp/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]

# These are source-module groups, independent of the two signal adapters.
GROUPS = {
    'spatial_large': {
        'passes': [54, 55, 56, 69, 70, 71],
        'sources': ['lib/lighting/denoiser/atrous_large.glsl',
                    'lib/lighting/denoiser/atrous_tap.glsl',
                    'lib/lighting/denoiser/atrous_filter.glsl',
                    'lib/lighting/denoiser/scratch_io.glsl'],
        'investigate': 'L1/L2 access locality, common scratch layout, workgroup '
                       'shape and register live ranges; preserve tap positions '
                       'and shared signal-domain mathematics.',
    },
    'spatial_small': {
        'passes': [51, 52, 53, 66, 67, 68],
        'sources': ['lib/lighting/denoiser/atrous_small.glsl',
                    'lib/lighting/denoiser/atrous_tap.glsl',
                    'lib/lighting/denoiser/atrous_filter.glsl',
                    'lib/math/denoiser_uncertainty.glsl'],
        'investigate': 'Common per-tap instruction work, repeated scalar sums '
                       'and center invariants; retain separate uncertainty '
                       'statistics for the two estimators.',
    },
    'variance_prepare': {
        'passes': [50, 65],
        'sources': ['lib/lighting/denoiser/variance_prepare.glsl',
                    'lib/lighting/denoiser/variance_tile.glsl'],
        'investigate': 'Shared 7x7 moment pooling: producer-side precomputation '
                       'versus tile storage, instruction count and occupancy.',
    },
    'temporal_history_resolve': {
        'passes': [58, 72],
        'sources': ['post/denoiser/diffuse/resolve.glsl',
                    'post/denoiser/reflection/resolve.glsl',
                    'lib/lighting/denoiser/temporal_response.glsl'],
        'investigate': 'History and public-output traffic, repeated reads and '
                       'intermediate stores; maintain history commit order.',
    },
    'reflection_reprojection': {
        'passes': [61],
        'sources': ['post/denoiser/reflection/temporal.glsl',
                    'post/denoiser/reflection/reprojection.glsl'],
        'investigate': 'Reprojection footprint reads, decoding reuse and live '
                       'history state; preserve temporal validity decisions.',
    },
    'staging_and_reprojection_copy': {
        'passes': [1, 59, 63],
        'sources': [],
        'investigate': 'VRAM traffic and staging lifetimes before removing or '
                       'fusing passes; preserve all cross-pass consumers.',
    },
    'radiance_cache': {'passes': [3], 'sources': []},
    'lighting_resolve': {'passes': [81], 'sources': []},
    'overlay': {'passes': [82], 'sources': []},
    'bloom_prepare_and_blur': {'passes': [90, 91, 92, 93], 'sources': []},
    'bloom_reconstruction_and_tonemap': {
        'passes': [98],
        'sources': ['lib/post_processing/final.glsl'],
        'investigate': 'Texture filtering work: reduce reconstruction fetches '
                       'while preserving the interpolation kernel and atlas '
                       'edge behavior.',
    },
    'frame_state_and_exposure': {'passes': [99], 'sources': []},
}

SELECTED_METRICS = [
    'SM Inst Executed [%]', 'SM Pipe ALU [%]', 'SM Pipe FMA Heavy [%]',
    'SM Pipe FMA Active [%]', 'SM Pipe SFU [%]', 'VRAM Throughput [%]',
    'L2 Throughput [%]', 'L1TEX Filter Stage Wavefronts [%]',
    'L1TEX TEX Data Wavefronts [%]', 'L1TEX LSU Data Wavefronts Shared Mem [%]',
    'L1TEX Hit Rate [%]', 'L2 Hit Rate [%]',
    'Active Warps per Cycle All [%]', 'Active Warps per Cycle All',
    'Warps Issue Stalled Long Scoreboard L1 [%]',
    'Warps Issue Stalled Wait [%]', 'Warps Issue Stalled TEX Throttle [%]',
    'Warps Issue Stalled Math Pipe Throttle [%]',
    'Warps Issue Stalled Short Scoreboard [%]',
    'Warps Issue Stalled Barrier [%]',
    'CS Warp Launch Stalled Shared Memory Allocation [%]',
    'CS Warp Launch Stalled Register Allocation [%]',
    'PS Warp Launch Stalled Register Allocation [%]',
    'Threads Launched CS', 'Threads Launched PS', 'Draw Started',
    'Dispatch Started', 'GR Cycles Elapsed',
]


def metric_values(record):
    """Deduplicate metrics repeated by different diagnostic rules.

    Preserve conflicting values explicitly instead of silently taking whichever
    rule happened to appear last. Missing top-five metrics are not zero.
    """
    values = {}
    for rule in record.get('Rules', []):
        for metric in rule.get('Metrics', []):
            bucket = values.setdefault(metric['Name'], [])
            if metric.get('Value') not in bucket:
                bucket.append(metric.get('Value'))
    return {name: values[0] if len(values) == 1 else values
            for name, values in values.items()}


def exact_duplicates(records):
    first_seen = {}
    duplicate_of = {}
    for index, record in enumerate(records):
        key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if key in first_seen:
            duplicate_of[index] = first_seen[key]
        else:
            first_seen[key] = index
    return duplicate_of


def aggregate_evidence(records, metrics, duplicate_of):
    """Infer only the composite parent supported by duration AND work counts.

    The YAML has no timestamps, event IDs or hierarchy. A matching name alone
    is insufficient: subsequent numbered ranges must sum to its duration and
    independently reproduce at least one positive hardware work count.
    """
    aggregates = []
    for index, record in enumerate(records):
        if index in duplicate_of or record['Range'] != 'composite':
            continue
        children = []
        for child in range(index + 1, len(records)):
            if not re.fullmatch(r'composite\d+', records[child]['Range']):
                break
            if child not in duplicate_of:
                children.append(child)
        if not children:
            continue
        child_duration = math.fsum(records[i]['RelativeFrameDuration']
                                   for i in children)
        parent_duration = record['RelativeFrameDuration']
        durations_match = math.isclose(parent_duration, child_duration,
                                      rel_tol=1e-10, abs_tol=1e-12)
        work = {}
        positive_matches = []
        for name in ('Threads Launched CS', 'Threads Launched PS',
                     'Draw Started', 'Dispatch Started'):
            parent = metrics[index].get(name)
            child_values = [metrics[i].get(name) for i in children]
            if not isinstance(parent, (float, int)) or not all(
                    isinstance(v, (float, int)) for v in child_values):
                continue
            total = sum(child_values)
            work[name] = {'parent': parent, 'children_sum': total,
                          'equal': parent == total}
            if parent > 0 and parent == total:
                positive_matches.append(name)
        if durations_match and positive_matches:
            aggregates.append({
                'index': index, 'range': record['Range'],
                'child_indices': children,
                'parent_frame_percent': parent_duration * 100,
                'children_frame_percent': child_duration * 100,
                'duration_fraction_difference': parent_duration - child_duration,
                'work_count_evidence': work,
                'interpretation': 'Aggregate consistent with all following '
                                  'numbered composite ranges; exclude parent '
                                  'when summing those children. This is not '
                                  'an individual path-tracing pass.',
            })
    return aggregates


def source_entries(range_name):
    if not re.fullmatch(r'composite\d+', range_name):
        return []
    return [str(path.relative_to(ROOT)).replace('\\', '/')
            for path in sorted((ROOT / 'shaders').glob(range_name + '.*'))
            if path.suffix in {'.csh', '.fsh', '.vsh'}]


def summarize(records, source, source_bytes):
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('Range'), str):
            raise ValueError(f'range {index} has no string Range')
        duration = record.get('RelativeFrameDuration')
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) \
                or duration < 0:
            raise ValueError(f'range {index} has invalid RelativeFrameDuration')

    duplicates = exact_duplicates(records)
    metrics = [metric_values(record) for record in records]
    aggregates = aggregate_evidence(records, metrics, duplicates)
    parent_indices = {r['index'] for r in aggregates}
    excluded = set(duplicates) | parent_indices
    leaves = [i for i in range(len(records)) if i not in excluded]
    covered = math.fsum(records[i]['RelativeFrameDuration'] for i in leaves)

    groups = []
    grouped_indices = set()
    for name, definition in GROUPS.items():
        members = [i for i in leaves if records[i]['Range'] in
                   {'composite' + str(n) for n in definition['passes']}]
        if not members:
            continue
        grouped_indices.update(members)
        group = {
            'name': name,
            'indices': members,
            'ranges': [records[i]['Range'] for i in members],
            'frame_percent': 100 * math.fsum(
                records[i]['RelativeFrameDuration'] for i in members),
            'sources': ['shaders/' + p for p in definition['sources']],
            'entry_sources': [p for i in members
                              for p in source_entries(records[i]['Range'])],
        }
        if 'investigate' in definition:
            group['candidate_investigation'] = definition['investigate']
        # Raw per-pass metrics are authoritative. Min/max simply help scan
        # homogeneous groups; hardware percentages are not summed or averaged.
        group['observed_metric_ranges'] = {}
        for metric in SELECTED_METRICS:
            values = [metrics[i][metric] for i in members
                      if isinstance(metrics[i].get(metric), (int, float))]
            if values:
                group['observed_metric_ranges'][metric] = {
                    'minimum': min(values), 'maximum': max(values),
                    'present_passes': len(values), 'total_passes': len(members),
                }
        groups.append(group)
    groups.sort(key=lambda g: g['frame_percent'], reverse=True)

    limitations = [
        'RelativeFrameDuration is the recorded fraction of the original frame. '
        'No absolute frame time or millisecond durations are exported.',
        'FrameGain and RangeSpeedupFactor are diagnostic estimates, not '
        'measured changes. Multiple rules overlap and their estimates must '
        'not be added.',
        'The flat export lacks timestamps and event IDs. Aggregate detection '
        'uses the exact duration sum plus independent work-count evidence; '
        'fully identical records are counted once.',
        'Any uncovered frame fraction is unclassified. Missing ray-tracing '
        'ranges do not prove that all unclassified time belongs to PT.',
        'The export contains no shader-source hash, GPU model, resolution '
        'declaration, clock log or capture repetition distribution. It cannot '
        'establish that current disk sources match the captured binaries.',
        'Many rules export only their highest metrics. An absent hardware '
        'metric means not exported, never zero.',
        'Warp-launch shared-memory stall percentages alone do not identify '
        'explicit GLSL shared-array capacity as the cause; this counter also '
        'appears on small atrous kernels with no explicit shared arrays. '
        'Check compiled resources and measured scheduling before changing LDS.',
        'Draw/dispatch counters can straddle range boundaries: some leaf '
        'records have CS threads with zero Dispatch Started. Such counters '
        'must not be treated as exact source-pass identities.',
        'A group frame fraction describes optimization exposure, not an '
        'achievable gain or an end-to-end performance measurement.',
    ]
    if covered > 1 + 1e-8:
        limitations.append('Even after detected exclusions the sum exceeds '
                           '100%; further nested ranges are unresolved.')

    output_ranges = []
    for i, record in enumerate(records):
        output_ranges.append({
            'index': i, 'name': record['Range'],
            'frame_percent': record['RelativeFrameDuration'] * 100,
            'counted_as_leaf': i in leaves,
            'duplicate_of': duplicates.get(i),
            'inferred_aggregate': i in parent_indices,
            'sources': source_entries(record['Range']),
            'selected_metrics': {name: metrics[i].get(name)
                                 for name in SELECTED_METRICS},
            'raw': record,
        })
    return {
        'input': str(source),
        'input_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'range_count': len(records),
        'raw_sum_frame_percent': 100 * math.fsum(
            r['RelativeFrameDuration'] for r in records),
        'exact_duplicates': [{'index': i, 'duplicate_of': original,
                              'range': records[i]['Range'],
                              'frame_percent': records[i]['RelativeFrameDuration'] * 100}
                             for i, original in duplicates.items()],
        'aggregates': aggregates,
        'counted_leaf_sum_frame_percent': covered * 100,
        'unclassified_frame_percent': (1 - covered) * 100,
        'groups_by_recorded_frame_share': groups,
        'ungrouped_leaf_indices': [i for i in leaves if i not in grouped_indices],
        'ranges_in_original_order': output_ranges,
        'limitations': limitations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Local exported YAML file')
    parser.add_argument('--output', type=Path,
                        default=Path('temp/bench1080/profile_review/report.json'))
    args = parser.parse_args()
    source = args.input.resolve()
    target = args.output.resolve()
    if not target.is_relative_to((ROOT / 'temp').resolve()):
        parser.error('--output must stay under project temp/')
    try:
        source_bytes = source.read_bytes()
        records = yaml.safe_load(source_bytes.decode('utf-8-sig'))
        if not isinstance(records, list):
            raise ValueError('expected a top-level list of ranges')
        report = summarize(records, source, source_bytes)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(f'Profile: {source}')
    print(f'Ranges: {len(records)}; duplicate records: '
          f'{len(report["exact_duplicates"])}; aggregate parents: '
          f'{len(report["aggregates"])}')
    print(f'Counted leaf share: {report["counted_leaf_sum_frame_percent"]:.6f}%; '
          f'unclassified: {report["unclassified_frame_percent"]:.6f}%')
    for group in report['groups_by_recorded_frame_share']:
        print(f'  {group["frame_percent"]:9.5f}%  {group["name"]}')
    print('Percentages are original-frame shares, not measured savings.')
    print(f'Report: {target}')


if __name__ == '__main__':
    main()
