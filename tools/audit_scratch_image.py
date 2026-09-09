"""Audit lossless denoiser image scratch and its handoff to bloom.

Run without --cpu-only to execute the real GLSL storage helpers and production
composite90..93 passes. All fixtures and reports stay under temp/bench1080/.
This checks correctness and resource lifetime; it does not measure frame time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import moderngl
import numpy as np

from audit_post_pipeline import compute_group_size, gl_source


ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / 'shaders'
OUTPUT = ROOT / 'temp/bench1080/scratch_image'
BIAS = np.uint32(0x00800000)


def emit(value):
    print(json.dumps(value), flush=True)


def normal_float_patterns(words):
    encoded = np.ascontiguousarray(words + BIAS, dtype='<u4')
    exponent = (encoded >> np.uint32(23)) & np.uint32(255)
    assert np.all((exponent >= 1) & (exponent <= 248))
    assert np.all(np.isfinite(encoded.view('<f4')))
    assert np.array_equal(encoded.view('<f4').view('<u4') - BIAS, words)
    return encoded


def half_pair_cases():
    # Exhaust the upper half: its bits alone determine the transport FP32
    # exponent. Lower-half variation cannot affect that exponent or the bias.
    high = np.arange(65536, dtype=np.uint32)
    high = high[(high & np.uint32(0x7c00)) != np.uint32(0x7c00)]
    assert high.size == 63488
    assert np.all(np.isfinite(high.astype('<u2').view('<f2')))
    low = np.array([0x0000, 0x8000, 0x0001, 0x8001,
                    0x03ff, 0x83ff, 0x0400, 0x8400,
                    0x7bff, 0xfbff, 0x3c00, 0xbc00,
                    0x4000, 0xc000, 0x3800, 0xb800], dtype='<u4')
    assert np.all(np.isfinite(low.astype('<u2').view('<f2')))
    words = ((high[:, None] << np.uint32(16)) | low[None, :]).reshape(-1, 4)
    encoded = normal_float_patterns(words)
    report = {
        'finite_high_half_patterns': int(high.size),
        'low_half_patterns': [f'0x{value:04x}' for value in low],
        'tested_words': int(words.size),
        'transport_exponent_range': [
            int(((encoded >> np.uint32(23)) & np.uint32(255)).min()),
            int(((encoded >> np.uint32(23)) & np.uint32(255)).max())],
        'cpu_roundtrip': 'bitwise_equal',
        'exhaustiveness': 'all finite upper FP16 halves; lower bits cannot affect exponent bias',
    }
    return words, report


def signal_cases():
    rng = np.random.default_rng(730921)
    values = np.zeros((4096, 8), dtype='<f4')
    energy = np.exp(rng.uniform(-14, 10, values.shape[0])).astype('f4')
    direction = rng.normal(size=(values.shape[0], 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    values[:, :3] = direction * energy[:, None]
    values[:, 3] = energy
    values[:, 4:6] = rng.uniform(-1, 1, (values.shape[0], 2)) * energy[:, None]
    values[:, 6] = np.sqrt(energy)
    values[:, 7] = rng.uniform(0, 65504, values.shape[0])
    values[1] = np.array([0., -0., 0., -0., 0., -0., -0., -0.])
    values[2] = [-65504, 65504, -65504, 65504, -65504, 65504, 65504, 65504]
    values[3] = [1e-8, -1e-8, 6e-8, 6e-8, 6e-8, -6e-8, -2, 0]
    values[4, :6] *= np.float32(1e8)  # finite values beyond storage range
    values[::17, 6] = -2  # usable moments with unknown uncertainty
    values[::23, 6] = -1  # invalid metadata, including nonzero moment payloads
    values[5, 6] = -3
    values[6, 6] = np.nan
    values[7, 6] = np.inf
    values[8, 0] = np.nan
    values[9, 3] = np.inf
    values[10, 4] = -np.inf
    values[11, 7] = np.nan
    values[12, 7] = np.inf
    values[13, 7] = -1
    canonical_invalid = np.arange(values.shape[0]) % 19 == 0

    # Independent CPU interpretation of signal.glsl's sanitizing boundary.
    # Do not call or mechanically translate the transport mapping to build the
    # expected FP16 payload: NumPy performs the storage rounding independently.
    expected = values.copy()
    bad_moment = ~np.all(np.isfinite(expected[:, :4]), axis=1)
    expected[bad_moment, :4] = 0
    expected[bad_moment, 6] = -2
    bad_chroma = ~np.all(np.isfinite(expected[:, 4:6]), axis=1)
    expected[bad_chroma, 4:6] = 0
    sigma = expected[:, 6]
    known = (sigma >= 0) & (sigma <= 65504)
    expected[~(known | (sigma == -1) | (sigma == -2)), 6] = -2
    expected[~np.isfinite(expected[:, 7]), 7] = 0
    expected[:, 7] = np.clip(expected[:, 7], 0, 65504)
    expected[:, :6] = np.clip(expected[:, :6], -65504, 65504)
    expected[canonical_invalid] = [0, 0, 0, 0, 0, 0, -1, 0]
    words = np.ascontiguousarray(expected, dtype='<f2').view('<u4')
    normal_float_patterns(words)
    report = {'signals': int(values.shape[0]),
              'canonical_invalid': int(canonical_invalid.sum()),
              'invalid_metadata': int(np.count_nonzero(expected[:, 6] == -1)),
              'unknown_uncertainty': int(np.count_nonzero(expected[:, 6] == -2)),
              'includes': ['signed zero', 'half subnormals', 'finite extrema',
                           'unknown/invalid sigma', 'nonfinite ingress',
                           'out-of-range finite ingress']}
    return values, words, report


def fixture_source(name, body):
    path = OUTPUT / (name + '.csh')
    path.write_text(body, encoding='utf-8')
    return gl_source(SHADERS, path)


def pixel_dispatch(shader, width, height):
    gx, gy, gz = compute_group_size(shader)
    assert gz == 1
    return (math.ceil(width / gx), math.ceil(height / gy), 1)


def bind_scratch(shader, images):
    for unit, (name, texture) in enumerate(zip(('bloomAtlas', 'bloomBlur'), images)):
        if name in shader:
            shader[name].value = unit
        sampler = name + '_Sampler'
        if sampler in shader:
            shader[sampler].value = 6 + unit
        texture.use(6 + unit)
        texture.bind_to_image(unit, read=False, write=True)


def gpu_roundtrip(ctx, name, input_values, expected_words, signals=False):
    count = expected_words.shape[0]
    width = 512
    assert count % width == 0
    height = count // width
    input_declaration = ('layout(std430, binding=0) readonly buffer Inputs { vec4 values[]; };'
                         if signals else
                         'layout(std430, binding=0) readonly buffer Inputs { uvec4 values[]; };')
    build = '''
    DenoiserMaxEntSignal signal;
    signal.maxEntY = values[2u * index];
    vec4 state = values[2u * index + 1u];
    signal.CoCg = state.xy;
    signal.standardDeviation = state.z;
    signal.virtualDistance = state.w;
    uvec4 words = index % 19u == 0u ? denoiserInvalidMaxEntSignalWords()
        : denoiserPackMaxEntSignal(signal);
''' if signals else '    uvec4 words = values[index];\n'
    producer = ctx.compute_shader(fixture_source(name + '_store', '''#version 430 core
layout(local_size_x=16, local_size_y=16) in;
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"
''' + input_declaration + '''
uniform ivec2 testSize;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, testSize))) return;
    uint index = uint(pixel.y * testSize.x + pixel.x);
''' + build + '''
    denoiserScratchStoreA(pixel, words);
    denoiserScratchStoreB(pixel, words.wzyx);
}
'''))
    consumer = ctx.compute_shader(fixture_source(name + '_load', '''#version 430 core
layout(local_size_x=16, local_size_y=16) in;
#include "/lib/lighting/denoiser/scratch_io.glsl"
layout(std430, binding=2) writeonly buffer OutputA { uvec4 resultsA[]; };
layout(std430, binding=3) writeonly buffer OutputB { uvec4 resultsB[]; };
uniform ivec2 testSize;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, testSize))) return;
    uint index = uint(pixel.y * testSize.x + pixel.x);
    resultsA[index] = denoiserScratchLoadA(pixel);
    resultsB[index] = denoiserScratchLoadB(pixel);
}
'''))
    source = ctx.buffer(np.ascontiguousarray(input_values).tobytes())
    output_a = ctx.buffer(reserve=expected_words.nbytes)
    output_b = ctx.buffer(reserve=expected_words.nbytes)
    images = [ctx.texture((width, height), 4, dtype='f4') for _ in range(2)]
    resources = [producer, consumer, source, output_a, output_b, *images]
    try:
        source.bind_to_storage_buffer(0)
        producer['testSize'].value = (width, height)
        bind_scratch(producer, images)
        producer.run(*pixel_dispatch(producer, width, height))
        # Separate producer/consumer programs force the real image-store and
        # texture-cache path; an SSA bitcast identity cannot satisfy this test.
        ctx.memory_barrier()
        output_a.bind_to_storage_buffer(2)
        output_b.bind_to_storage_buffer(3)
        consumer['testSize'].value = (width, height)
        bind_scratch(consumer, images)
        consumer.run(*pixel_dispatch(consumer, width, height))
        ctx.memory_barrier()
        for image, output, expected in zip(images, (output_a, output_b),
                                           (expected_words, expected_words[:, ::-1])):
            actual = np.frombuffer(output.read(), dtype='<u4').reshape(-1, 4)
            if not np.array_equal(actual, expected):
                mismatch = np.argwhere(actual != expected)[0]
                raise AssertionError((name, 'roundtrip mismatch', mismatch.tolist(),
                                      hex(int(actual[tuple(mismatch)])),
                                      hex(int(expected[tuple(mismatch)]))))
            image_words = np.frombuffer(image.read(), dtype='<u4').reshape(-1, 4)
            assert np.array_equal(image_words, normal_float_patterns(expected)), name
        return {'test': name, 'size': [width, height], 'words_per_image': int(expected_words.size),
                'both_images': 'bitwise_equal', 'image_storage': 'finite_normal_fp32',
                'sha256': hashlib.sha256(output_a.read()).hexdigest()}
    finally:
        for resource in resources:
            resource.release()


def bloom_dispatch(shader, entry, source, width, height):
    gx, gy, gz = compute_group_size(shader)
    if entry == 'composite90.csh':
        assert re.search(r'^\s*#define\s+BLOOM_SPD_LOCAL\b', source, re.M)
        match = re.search(r'const\s+vec2\s+workGroupsRender\s*=\s*vec2\(([^)]+)\)', source)
        assert match, 'local SPD dispatch contract missing'
        scale = [float(value.strip()) for value in match[1].split(',')]
        assert len(scale) == 2
        return (math.ceil(width * scale[0] / gx), math.ceil(height * scale[1] / gy), 1)
    if entry == 'composite91.csh':
        assert re.search(r'^\s*#define\s+BLOOM_SPD_TAIL\b', source, re.M)
        match = re.search(r'const\s+ivec3\s+workGroups\s*=\s*ivec3\(([^)]+)\)', source)
        assert match, 'tail SPD dispatch contract missing'
        groups = tuple(int(value.strip()) for value in match[1].split(','))
        assert len(groups) == 3
        return groups
    assert entry in ('composite92.csh', 'composite93.csh') and gz == 1
    return pixel_dispatch(shader, width, height)


def bloom_levels(width, height, highest=8):
    levels = []
    for level in range(highest + 1):
        sx, sy = width >> (level + 1), height >> (level + 1)
        if sx > 0 and sy > 0:
            levels.append((level, width - (width >> level),
                           height - (height >> level), sx, sy))
    return levels


def bloom_snapshot(image, width, height, highest=None):
    values = np.frombuffer(image.read(), dtype='<f4').reshape(height, width, 4)
    if highest is None:
        return values.copy()
    selected = []
    for level, ox, oy, sx, sy in bloom_levels(width, height, highest):
        data = values[oy:oy+sy, ox:ox+sx]
        assert np.all(data[..., 3] == 1), ('SPD did not publish level', level, width, height)
        selected.append(data.reshape(-1, 4))
    return np.concatenate(selected) if selected else np.empty((0, 4), dtype='<f4')


def bloom_handoff(ctx, width, height, pair_words):
    rng = np.random.default_rng(125117 + width + 17 * height)
    scene_values = np.exp(rng.uniform(-4, 9, (height, width, 4))).astype('<f4')
    scene_values[..., 3] = 1
    scene_values[height//2, width//2, :3] = [1e6, 1e4, 1e2]
    scene = ctx.texture((width, height), 4, scene_values.tobytes(), dtype='f4')
    entries = [f'composite{stage}.csh' for stage in (90, 91, 92, 93)]
    sources = [gl_source(SHADERS, entry) for entry in entries]
    programs = [ctx.compute_shader(source) for source in sources]
    groups = [bloom_dispatch(shader, entry, source, width, height)
              for shader, entry, source in zip(programs, entries, sources)]
    zero = np.zeros((height, width, 4), dtype='<f4')
    # Include both very large positive/negative normal transport floats even
    # in 1x1 tests; tiny positive residues alone could round away in bloom.
    seeds = pair_words[[len(pair_words)//2 - 1, len(pair_words) - 1,
                        0, len(pair_words)//4, 3*len(pair_words)//4]]
    dirty = normal_float_patterns(np.resize(seeds, (height, width, 4)))
    clean_snapshots = []
    comparisons = []
    try:
        for dirty_start in (False, True):
            textures = [ctx.texture((width, height), 4,
                                   (np.roll(dirty, axis=-1, shift=unit).tobytes()
                                    if dirty_start else zero.tobytes()), dtype='f4')
                        for unit in range(2)]
            try:
                for index, (entry, shader, dispatch) in enumerate(zip(entries, programs, groups)):
                    scene.use(2)
                    if 'colortex0' in shader:
                        shader['colortex0'].value = 2
                    for unit, (name, texture) in enumerate(zip(('bloomAtlas', 'bloomBlur'), textures)):
                        if name in shader:
                            shader[name].value = unit
                        texture.bind_to_image(unit, read=True, write=True)
                    shader.run(*dispatch)
                    ctx.memory_barrier()
                    # 90 owns L0..L3, 91 finishes every nonempty level. The two
                    # blur passes overwrite the complete target, including gaps.
                    target = textures[1 if index == 2 else 0]
                    highest = 3 if index == 0 else (8 if index == 1 else None)
                    actual = bloom_snapshot(target, width, height, highest)
                    assert np.all(np.isfinite(actual)), (entry, width, height)
                    if not dirty_start:
                        clean_snapshots.append(actual)
                    else:
                        expected = clean_snapshots[index]
                        assert np.array_equal(actual.view('<u4'), expected.view('<u4')), (
                            'scratch residue reached bloom', entry, width, height,
                            int(np.count_nonzero(actual.view('<u4') != expected.view('<u4'))))
                        comparisons.append({'entry': entry, 'groups': list(dispatch),
                                            'local_size': list(compute_group_size(shader)),
                                            'compared_pixels': int(actual.size // 4),
                                            'scope': ('L0..L3' if index == 0 else
                                                      'L0..L8' if index == 1 else 'entire output image'),
                                            'result': 'bitwise_equal',
                                            'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
            finally:
                for texture in textures:
                    texture.release()
        return {'size': [width, height],
                'nonempty_levels': [level for level, *_ in bloom_levels(width, height)],
                'stages': comparisons}
    finally:
        scene.release()
        for shader in programs:
            shader.release()


def parse_size(text):
    match = re.fullmatch(r'(\d+)[xX](\d+)', text)
    if not match or min(map(int, match.groups())) < 1:
        raise argparse.ArgumentTypeError('expected positive WIDTHxHEIGHT')
    return tuple(map(int, match.groups()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true', help='prove encoding and build reference cases without opening a GPU context')
    parser.add_argument('--size', action='append', type=parse_size,
                        help='bloom handoff dimensions; repeatable')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    words, pair_report = half_pair_cases()
    signals, signal_words, signal_report = signal_cases()
    report = {'cpu_half_pairs': pair_report, 'signal_fixtures': signal_report}
    emit(report)
    destination = OUTPUT / ('cpu_report.json' if args.cpu_only else 'report.json')
    if not args.cpu_only:
        ctx = moderngl.create_standalone_context(require=430)
        try:
            report['renderer'] = ctx.info.get('GL_RENDERER')
            report['gpu_roundtrip'] = [gpu_roundtrip(ctx, 'all_finite_high_halves', words, words),
                                       gpu_roundtrip(ctx, 'production_signal_pack', signals, signal_words, signals=True)]
            emit({'gpu_roundtrip': report['gpu_roundtrip']})
            report['bloom_handoff'] = []
            for width, height in (args.size or [(1920, 1080), (257, 145),
                                                 (17, 9), (3, 2), (1, 1), (1, 7)]):
                result = bloom_handoff(ctx, width, height, words)
                report['bloom_handoff'].append(result)
                emit({'bloom_handoff': result})
        finally:
            ctx.release()
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    emit({'report': str(destination), 'result': 'passed', 'gpu_executed': not args.cpu_only})


if __name__ == '__main__':
    main()
