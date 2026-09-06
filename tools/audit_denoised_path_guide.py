"""Read-only contract checks for denoised screen-space path guiding."""
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    shaders = ROOT / 'shaders'
    source = {p.relative_to(shaders).as_posix(): p.read_text(encoding='utf-8')
              for p in shaders.rglob('*')
              if p.suffix in ('.glsl', '.csh', '.rgen', '.properties', '.lang')}
    for name, text in source.items():
        assert not re.search(r'RESTIR_GI_|PATHGUIDE_(?:SPATIAL_RADIUS|MAX_TEMPORAL_M)|DIF_N_RESTIR_', text), name
    props = source['shaders.properties']
    groups = re.search(r'^rt.groups = (.+)$', props, re.M)[1].split()
    assert groups == ['0', '1', '1', '1', '2', '3']
    assert sorted(p.name for p in shaders.glob('ray*.rgen')) == [f'ray{i}.rgen' for i in range(6)]
    assert not (shaders/'composite57.csh').exists()
    assert 'radiance_cache.glsl' in source['ray4.rgen']
    assert 'radiance_cache_rgen.glsl' in source['ray5.rgen']
    for i in range(6):
        for stage in ('rahit', 'rchit', 'rmiss'):
            assert (shaders/f'ray{i}_0.{stage}').exists()
    buf = source['lib/buffers/diffuse_buffer.glsl']
    layers = [int(v) for v in re.findall(r'^#define DIF_N_\w+\s+(\d+)u', buf, re.M)]
    assert sorted(layers) == list(range(11))
    assert 'bufferObject.2 = 176 true 1.1 1.1' in props
    resolve = source['post/denoiser/diffuse/resolve.glsl']
    assert 'uvec4(packedLight.xy, floatBitsToUint(1.0), floatBitsToUint(1.0))' in resolve
    assert 'colortex5' not in resolve
    assert 'if (!valid) maxEntY = vec4(0.0);' in buf
    # Packed guide moments equal packed resolved light, including FP16 rounding.
    rng = np.random.default_rng(906202634)
    moments = rng.normal(size=(10000, 4)).astype(np.float32)
    moments[:, 3] = np.linalg.norm(moments[:, :3], axis=1) + rng.random(10000)
    packed = moments.astype(np.float16).view(np.uint32).reshape(-1, 2)
    guide = np.column_stack((packed, np.full((len(packed), 2), 0x3f800000, dtype=np.uint32)))
    assert np.array_equal(guide[:, :2], packed)
    assert np.array_equal(guide[:, :2].copy().view(np.float16).reshape(-1, 4), moments.astype(np.float16))
    # Daytime stays on the canonical complementary MIS pair.
    trace = source['lib/rt/raytrace/path_trace.glsl']
    assert trace.count('float misWeight = powerHeuristic(lightPdf, proposalPdf);') == 2
    assert 'lastBsdfStrategyPdf, lightPdf' in trace
    assert 'writeDiffuseOutput(' in trace and 'L_direct_0_dir, ro' in trace
    print(json.dumps(dict(rt_stages=6, diffuse_layers=11, bytes_per_pixel=176,
                         bytes_released_per_pixel=64, packed_cases=10000,
                         obsolete_pg_symbols=0, scope='CPU packing and source contract; no GPU execution'), indent=2))


if __name__ == '__main__':
    main()
