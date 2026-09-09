"""Execute production compute passes on OpenGL with reproducible synthetic data.

Compare against a pre-edit shader directory (all includes come from that tree).
This checks real image/SSBO packing, edge workgroups and both estimator branches;
GPU timings are isolated dispatch measurements, not in-game frame rates.
"""
from pathlib import Path
import argparse
import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
import os
import json
import re
import struct
import time

import moderngl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def gl_get_program_iv():
    if os.name == 'nt':
        library = ctypes.WinDLL('opengl32')
        library.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        library.wglGetProcAddress.restype = ctypes.c_void_p
        address = library.wglGetProcAddress(b'glGetProgramiv')
        if not address: raise RuntimeError('glGetProgramiv unavailable')
        return ctypes.WINFUNCTYPE(None, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.POINTER(ctypes.c_int))(address)
    library = ctypes.CDLL(ctypes.util.find_library('GL'))
    function = library.glGetProgramiv
    function.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
    function.restype = None
    return function


def compute_group_size(shader):
    # Query the linked shader, including its active preprocessing branches.
    # Dispatch tests must follow workgroup tuning without domain-specific guesses.
    size = (ctypes.c_int * 3)()
    gl_get_program_iv()(shader.glo, 0x8267, size)  # GL_COMPUTE_WORK_GROUP_SIZE
    assert all(value > 0 for value in size), tuple(size)
    return tuple(size)


def expand(root, path):
    source = (root / path).read_text(encoding="utf-8-sig")
    return re.sub(r'^\s*#include\s+"([^"]+)"\s*$',
                  lambda m: expand(root, m[1].lstrip("/")), source, flags=re.M)


def gl_source(root, entry, defines=None):
    source = expand(root, entry)
    for name, value in (defines or {}).items():
        source, count = re.subn(r'^\s*#define[ \t]+' + re.escape(name) + r'(?:[ \t]+[^\n]*)?$',
                                '#define ' + name + ' ' + str(value), source, flags=re.M)
        assert count, ('missing source option', name)
    source = re.sub(r'^\s*#version[^\n]*', '', source, flags=re.M)
    # Descriptor set numbers are Vulkan-only; binding numbers are unchanged.
    source = re.sub(r'\bset\s*=\s*\d+\s*,\s*', '', source)
    # ModernGL's uniform setter does not update uimage2D units on this driver.
    # Explicit GLSL image bindings avoid silent writes to the default unit 0.
    def image_binding(match):
        qualifiers, name = match[1], match[2]
        unit = int(name[8:]) if name.startswith('colorimg') else {'bloomAtlas':0, 'bloomBlur':1}[name]
        return f'layout(binding={unit}, {qualifiers}) uniform writeonly uimage2D {name}'
    source = re.sub(r'layout\((rgba32ui)\)\s+uniform\s+writeonly\s+uimage2D\s+(colorimg\d+)',
                    image_binding, source)
    return '#version 430 core\n#define MC_GL_NV_gpu_shader5 1\n' + source


def frame_state(width, height):
    data = bytearray(1024)
    for offset in range(0, 256, 64):
        data[offset:offset+64] = np.eye(4, dtype='f4').tobytes()
    struct.pack_into('3f', data, 256, 0, 1, 0)
    for offset in (268, 284, 368, 372):
        struct.pack_into('f', data, offset, 1.0)
    struct.pack_into('f', data, 300, 2.0)
    struct.pack_into('2I2i', data, 336, width, height, 0, 3)
    struct.pack_into('2f', data, 356, 0.2, 0.4)
    struct.pack_into('4f', data, 384, 1.2, 1.7, 0.001, -0.002)
    return data


def tiled_indices(width, height):
    y, x = np.indices((height, width), dtype=np.uint32)
    return (((y >> 3) * ((width+7)//8) + (x >> 3)) * 64
            + (y & 7) * 8 + (x & 7)).ravel()


def half_words(values):
    return np.ascontiguousarray(values, dtype='<f2').view('<u4')


def timed_pair(ctx, runners, trials=31, repetitions=1, preparations=None):
    times = [[], []]
    # Warm both variants through the laptop GPU's idle-clock transition.
    deadline = time.perf_counter() + (0.5 if repetitions > 1 else 0.15)
    while time.perf_counter() < deadline:
        for index, run in enumerate(runners):
            if preparations: preparations[index]()
            run()
        ctx.finish()
    for trial in range(trials):
        for index in ([0, 1] if trial % 2 == 0 else [1, 0]):
            if preparations: preparations[index]()
            with ctx.query(time=True) as query:
                for _ in range(repetitions): runners[index]()
            times[index].append(query.elapsed / (1e6 * repetitions))
    result = {name: float(np.median(t)) for name, t in zip(('baseline_ms', 'current_ms'), times)}
    result['paired_speedup_median'] = float(np.median(np.array(times[0])/times[1]))
    result['p10_p90_ms'] = [np.percentile(t,[10,90]).tolist() for t in times]
    result['trials'] = trials
    result['dispatches_per_query'] = repetitions
    result['paired_ms'] = list(map(list, zip(*times)))
    return result


def compare_half(a, b):
    av = np.frombuffer(a, dtype='<f2').astype('f4')
    bv = np.frombuffer(b, dtype='<f2').astype('f4')
    assert np.all(np.isfinite(av)) and np.all(np.isfinite(bv))
    error = np.abs(av-bv) / np.maximum(1.0, np.abs(av))
    # A small number of FP16 midpoint ties can flip by one ULP after hoisting.
    assert np.max(error) <= 0.002, float(np.max(error))
    return {'max_scaled_error': float(np.max(error)),
            'changed_half_lanes': int(np.count_nonzero(av != bv))}


def spatial_consumed_words(data, stream, trimmed=False):
    """Canonicalize only spatial fields with no downstream consumer.

    Role-lifetime kernels must write zero to the discarded fields. The old
    implementation still computes them, so compare the remaining fields while
    full-chain tests independently check every final persistent output.
    """
    words = np.frombuffer(data, dtype='u4').reshape(-1, 4).copy()
    if stream == 'proposal':
        if trimmed: assert not np.any(words[:, 2]), 'proposal chroma must be zero'
        words[:, 2] = 0
    else:
        if trimmed: assert not np.any(words[:, 3] >> 16), 'current distance must be zero'
        words[:, 3] &= np.uint32(0xffff)
    return words.tobytes()


def check_signal_half_rounding(a, b, key, stride, ulp_limit):
    """Bound FP16 lighting rounding without tolerating metadata changes.

    Resource/lifetime changes can alter driver contraction at FP16 midpoint
    ties. Only spatial light moments/chroma and final reflection light payloads
    are eligible; sigma, distances, geometry, counts and raw histories stay exact.
    The caller must explicitly request a nonzero limit (the default is exact).
    """
    if not ulp_limit: return None
    aw, bw = [np.frombuffer(data, dtype='u4').reshape(-1, 4) for data in (a,b)]
    allowed = np.zeros(aw.shape, dtype=bool)
    match = re.match(r'composite(\d+)\.csh/(color[45]|independent)$', key)
    if match and (51 <= int(match[1]) <= 56 or 66 <= int(match[1]) <= 71):
        allowed[:, :3] = True
    elif key == 'final/buffer3':
        # specular_buffer.glsl: public light N0 and filtered-history N3.
        allowed.reshape(5, stride, 4)[[0,3], :, :3] = True
    else:
        return None
    assert not np.any((aw != bw) & ~allowed), ('changed non-lighting state', key)
    ah, bh = [words[allowed].copy().view('u2') for words in (aw,bw)]
    assert np.all(np.isfinite(ah.view('f2'))) and np.all(np.isfinite(bh.view('f2'))), key
    ordered = [np.where(words & 0x8000, 0x8000-(words.astype('i4') & 0x7fff),
                        0x8000+words.astype('i4')) for words in (ah,bh)]
    error = np.abs(ordered[0]-ordered[1])
    maximum = int(error.max(initial=0))
    assert maximum <= ulp_limit, (key, 'FP16 ULP error', maximum, ulp_limit)
    return {'result':'bounded FP16 lighting rounding; all metadata exact',
            'max_half_ulp':maximum, 'changed_half_lanes':int(np.count_nonzero(ah != bh)),
            'changed_records':int(np.count_nonzero(np.any(aw != bw, axis=1)))}


def atrous(ctx, trees, width, height):
    rng = np.random.default_rng(819)
    y, x = np.indices((height, width))
    count = width * height
    stride = ((width+7)//8) * ((height+7)//8) * 64
    indices = tiled_indices(width, height)
    frame = ctx.buffer(frame_state(width, height)); frame.bind_to_storage_buffer(1)
    signal = np.zeros((height, width, 8), dtype='f4')
    energy = np.exp(rng.uniform(-5, 7, (height, width)))
    direction = rng.normal(size=(height, width, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    signal[..., :3] = direction * energy[..., None] * rng.uniform(0, 1, (height, width, 1))
    signal[..., 3] = energy
    signal[..., 4:6] = rng.uniform(-0.5, 0.5, (height, width, 2)) * energy[..., None]
    signal[..., 6] = np.sqrt(energy)
    signal[..., 7] = 3 + x * 0.005 + y * 0.008
    signal[(x+y) % 17 == 0, 6] = -2  # usable, unknown uncertainty
    signal[(x+3*y) % 43 == 0, 6] = -1  # invalid
    current = signal.copy()
    current[..., :6] *= 0.7
    current[(x+2*y) % 19 == 0, 6] = -2  # independent availability of uncertainty
    current[..., 7] *= 1.1  # ensure independent virtual-distance propagation
    sw = half_words(signal)
    cw = half_words(current)
    input_texture = ctx.texture((width, height), 4, sw.tobytes(), dtype='u4')
    geometry = np.zeros((height, width, 4), dtype='u4')
    ray_length = np.sqrt(((2*x/width-1+0.001)/1.2)**2
                         + ((2*y/height-1-0.002)/1.7)**2 + 1)
    geometry[..., 0] = (ray_length * np.where(x < width//2, 3, 6)).astype('f4').view('u4')
    geometry[..., 1] = rng.integers(0, 2**32, (height, width), dtype='u4')
    geometry[(x+5*y) % 59 == 0, 0] = np.float32(-1).view('u4')
    report = []
    diffuse_digests = {}
    entries = [f'composite{i}.csh' for i in list(range(51, 57)) + list(range(66, 72))]
    for entry in entries:
        reflection = int(re.search(r'\d+', entry)[0]) >= 66
        # Exercise the same roughness range through both storage adapters.
        # Actual diffuse preparation (alpha=1) is checked by denoiser_chain.
        roughness = np.where(x % 3 == 0, 0, np.where(x % 3 == 1, 0.35, 1))
        geometry[..., 3] = half_words(np.stack((roughness, np.full_like(x, 4)), axis=-1))[..., 0]
        geom_tex = ctx.texture((width, height), 4, geometry.tobytes(), dtype='u4')
        sources = [re.sub(r'(#define MAXENT_SPATIAL_(?:DIFFUSE|SPECULAR)_LIGHT_FIELD_SENSITIVITY)\s+[^\n]+',
                          r'\1 0.2', gl_source(tree, entry)) for tree in trees]
        shaders = [ctx.compute_shader(source) for source in sources]
        outputs = [ctx.texture((width, height), 4, dtype='u4') for _ in trees]
        buffers, runners, scratch_images = [], [], []
        for shader, output, tree in zip(shaders, outputs, trees):
            source = (tree / entry).read_text()
            alternate = 'MAXENT_ATROUS_WRITE_ALTERNATE' in source
            signal_name = 'colortex5' if reflection == alternate else 'colortex4'
            output_name = 'colorimg4' if signal_name == 'colortex5' else 'colorimg5'
            shader['colortex3'].value = 0
            shader[signal_name].value = 1
            data = np.zeros((10*stride, 4), dtype='u4')
            data[(8 if alternate else 9)*stride+indices] = cw.reshape(count, 4)
            buffer = ctx.buffer(data.tobytes()); buffers.append(buffer)
            image_scratch = 'bloomAtlas_Sampler' in shader or 'bloomBlur_Sampler' in shader
            scratch_pair = None
            if image_scratch:
                encoded = (cw + np.uint32(0x00800000)).tobytes()
                scratch_pair = [ctx.texture((width,height),4,encoded,dtype='f4') for _ in range(2)]
                for n, name in enumerate(('bloomAtlas','bloomBlur')):
                    if name in shader: shader[name].value=n
                    if name+'_Sampler' in shader: shader[name+'_Sampler'].value=7+n
            scratch_images.append(scratch_pair)
            gx, gy, _ = compute_group_size(shader)
            groups = ((width + gx - 1) // gx, (height + gy - 1) // gy)
            unit = int(output_name[8:])
            def run(shader=shader, buffer=buffer, output=output, groups=groups, unit=unit, scratch_pair=scratch_pair):
                geom_tex.use(0); input_texture.use(1)
                if scratch_pair:
                    for n, texture in enumerate(scratch_pair):
                        texture.use(7+n); texture.bind_to_image(n,read=False,write=True)
                output.bind_to_image(unit, read=False, write=True)
                buffer.bind_to_storage_buffer(2)
                shader.run(*groups); ctx.memory_barrier()
            runners.append(run)
        timing = timed_pair(ctx, runners, repetitions=4)
        trimmed = ['struct DenoiserSpatialCurrentAccumulator' in source for source in sources]
        proposal_written = output_name in shaders[1]
        if proposal_written:
            proposal_data = [output.read() for output in outputs]
            if any(trimmed):
                proposal_data = [spatial_consumed_words(data, 'proposal', flag)
                                 for data, flag in zip(proposal_data, trimmed)]
            filtered = compare_half(*proposal_data)
        else:
            assert entry in ('composite56.csh', 'composite71.csh')
            filtered = {'result': 'final proposal has no consumer; image store absent'}
        output_plane = 9 if alternate else 8
        independent = []
        for buffer, scratch_pair in zip(buffers, scratch_images):
            if scratch_pair:
                words=np.frombuffer(scratch_pair[1 if alternate else 0].read(),dtype='u4')-np.uint32(0x00800000)
                independent.append(words.tobytes())
            else:
                independent.append(np.frombuffer(buffer.read(), dtype='u4').reshape(-1, 4)[output_plane*stride+indices].tobytes())
        if any(trimmed):
            independent = [spatial_consumed_words(data, 'independent', flag)
                           for data, flag in zip(independent, trimmed)]
        result = {'entry': entry, **timing, 'proposal': filtered,
                  'independent': compare_half(*independent),
                  'source_sha256': [hashlib.sha256(source.encode('utf-8')).hexdigest() for source in sources],
                  'workgroups': [compute_group_size(shader) for shader in shaders]}
        digest = hashlib.sha256((outputs[1].read() if proposal_written else b'') + independent[1]).hexdigest()
        step_index = int(re.search(r'\d+', entry)[0]) - (66 if reflection else 51)
        if reflection:
            assert digest == diffuse_digests[step_index], ('domain-dependent spatial math', entry)
            result['same_input_domain_equivalence'] = 'bitwise_equal'
        else:
            diffuse_digests[step_index] = digest
        report.append(result); print(json.dumps(result), flush=True)
        for resource in shaders + outputs + buffers + [geom_tex]:
            resource.release()
        for scratch_pair in scratch_images:
            if scratch_pair:
                for texture in scratch_pair: texture.release()
    frame.release(); input_texture.release()
    return report


def bloom(ctx, trees, width, height):
    rng = np.random.default_rng(419)
    data = np.exp(rng.uniform(-4, 15, (height, width, 4))).astype('f4')
    data[0, 0, :3] = [np.nan, np.inf, -np.inf]
    result = []
    for axis, entry in enumerate(('composite92.csh', 'composite93.csh')):
        srcname, dstname = ('bloomAtlas', 'bloomBlur') if axis == 0 else ('bloomBlur', 'bloomAtlas')
        src = ctx.texture((width, height), 4, data.tobytes(), dtype='f4')
        shaders = [ctx.compute_shader(gl_source(tree, entry)) for tree in trees]
        outputs = [ctx.texture((width, height), 4, dtype='f4') for _ in trees]
        runners = []
        for shader, output in zip(shaders, outputs):
            shader[srcname].value = 0; shader[dstname].value = 1
            def run(shader=shader, output=output):
                src.bind_to_image(0, read=True, write=False)
                output.bind_to_image(1, read=False, write=True)
                groups = ((width+255)//256, height) if axis == 0 else (width, (height+255)//256)
                shader.run(*groups); ctx.memory_barrier()
            runners.append(run)
        timing = timed_pair(ctx, runners)
        av, bv = [np.frombuffer(o.read(), dtype='f4') for o in outputs]
        error = np.max(np.abs(av-bv) / np.maximum(1.0, np.abs(av)))
        assert np.all(np.isfinite(bv)) and error < 3e-6, float(error)
        row = {'entry': entry, **timing, 'max_scaled_error': float(error)}
        result.append(row); print(json.dumps(row), flush=True)
        for resource in shaders + outputs + [src]: resource.release()
    return result


def exposure(ctx, trees):
    width, height = 257, 145
    rng = np.random.default_rng(728)
    data = np.exp(rng.uniform(-8, 8, (height, width, 3))).astype('f4')
    src = ctx.texture((width, height), 3, data.tobytes(), dtype='f4')
    src.filter = (moderngl.LINEAR, moderngl.LINEAR); src.use(0)
    shaders = [ctx.compute_shader(gl_source(tree, 'composite99.csh')) for tree in trees]
    buffers = [ctx.buffer(frame_state(width, height)) for _ in trees]
    runners = []
    for shader, buffer in zip(shaders, buffers):
        for name, value in dict(colortex1=0, frameCounter=1, frameTimeCounter=2.125,
                rainStrength=0.7, wetness=0.8, viewWidth=float(width), viewHeight=float(height),
                sunPosition=(1., 2., 3.)).items():
            shader[name].value = value
        shader['gbufferModelViewInverse'].write(np.eye(4, dtype='f4').tobytes())
        def run(shader=shader, buffer=buffer):
            buffer.bind_to_storage_buffer(1); shader.run(1); ctx.memory_barrier()
        runners.append(run)
    timing = timed_pair(ctx, runners)
    # Reset after timing: adaptation state must match before numerical checks.
    results = []
    for buffer, run in zip(buffers, runners):
        buffer.write(frame_state(width, height)); run()
        results.append(np.frombuffer(buffer.read(), dtype='f4').copy())
    offsets = [268, 284, 368, 372]
    error = max(abs(results[0][i//4]-results[1][i//4]) / max(1e-6, abs(results[0][i//4])) for i in offsets)
    assert error < 2e-4, float(error)
    retention = 2 ** (-0.09016844 * 0.125)
    assert abs(results[1][356//4] - (0.7 + (0.2-0.7)*retention)) < 2e-7
    assert abs(results[1][360//4] - (0.8 + (0.4-0.8)*retention)) < 2e-7
    row = {'entry': 'composite99.csh', **timing, 'exposure_relative_error': float(error),
           'wetness_exponential_response': 'passed'}
    print(json.dumps(row), flush=True)
    for resource in shaders + buffers + [src]: resource.release()
    return row


def denoiser_chain(ctx, trees, width, height, timing_stage=None, debug_view=0,
                   signal_half_ulp_tolerance=0, capture_sink=None):
    """Temporal -> preparation -> all six iterations -> persistent commit.

    Synthetic producer uses the production ABI writers, including valid
    reflection history and the ray0/ray1 prepared reprojection records.
    An optional capture_sink(tree, key, bytes) streams readbacks to a separate
    ABI/precision analysis without retaining every frame buffer in memory.
    Sink mode performs no cross-tree comparison and never reports a test pass.
    """
    if capture_sink is not None:
        assert timing_stage is None, 'capture sink is for untimed validation only'
    fixture = ROOT/'temp/allpass/denoiser_fixture.csh'
    fixture.write_text('''#version 430 core
layout(local_size_x=8, local_size_y=8) in;
#define DIFFUSE_BUFFER
#define REFLECT_BUFFER
#include "/post/denoiser/reflection/common.glsl"
void main() {
    uvec2 p=gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(p,resolution_global))) return;
    vec3 ray=reconstructPrimaryRay(p);
    debugWriteReflectionSampleDirection(p,ray);
    float d=(p.x < resolution_global.x/2u ? 3.0 : 6.0)/(-ray.z);
    if (all(equal(p,uvec2(2)))) d=uintBitsToFloat(0x80000000u);
    float roughness=float(p.x%3u)*0.5;
    bool sky=(p.x+3u*p.y)%43u==0u;
    vec3 normal=p.x<2u ? vec3(0,0,1)
        : normalize(vec3(0.3*sin(float(p.x)*0.11),0.4*cos(float(p.y)*0.13),1));
    geomBuffer.data[addr(0u,p)]=uvec4(encodeNormalU(normal),
        packHalf2x16(vec2(roughness*roughness,0)),encodeNormalU(normal),floatBitsToUint(sky?-1.0:d));
    geomBuffer.data[addr(GEO_N_MOTION,p)]=uvec4(0,0,0,floatBitsToUint(1.0));
    diffuseBuffer.data[addr(DIF_N_SURFACE,p)]=uvec4(0,0,0,floatBitsToUint(sky?-1.0:d));
    float energy=exp2(sin(float(p.x)*0.12)+cos(float(p.y)*0.07)+2.0);
    vec3 direction=normalize(vec3(sin(float(p.x)),cos(float(p.y)),1));
    MaxEntEncoding raw;
    raw.maxEntY=vec4(direction*energy*0.8,energy); raw.CoCg=vec2(0.1,-0.2)*energy;
    MaxEntEncoding hist=raw; hist.maxEntY*=0.6; hist.CoCg*=0.6;
    float samples=1.0+float(p.y%12u);
    writeDiffuseLightRT(p,raw,energy);
    writeDiffuseSwap(p,hist,samples,energy);
    writeDiffuseDenoisedReprojection(p,hist.maxEntY,hist.CoCg,0.3,1.0,1.0);
    SpecularMaxEnt spec; spec.maxEntY=hist.maxEntY; spec.CoCg=hist.CoCg;
    MaxEntSpecularHistory history;
    history.surfacePosition=ray*d; history.geometryNormal=normal;
    history.signal=spec; history.rootMeanY2=energy;
    history.hitDistance=0.5; history.roughness=roughness;
    history.historyEffectiveSamples=samples;
    history.materialID=specularHistoryMaterialSignature(0u);
    writeMaxEntSpecularTemporalHistory(p,history);
    writeMaxEntSpecularDenoisedHistory(p,spec,0.3,1.0);
    writeMaxEntSpecularPreparedSurfaceDenoised(p,spec,0.3,1.0);
    spec.maxEntY=raw.maxEntY; spec.CoCg=raw.CoCg;
    writeReflMaxEnt(p,spec,0.5,1.0);
}
''', encoding='utf-8')
    stride = ((width+7)//8)*((height+7)//8)*64
    stages = [1, 50, *range(51, 57), 58, 59, 61, 63, 65, *range(66, 72), 72]
    assert timing_stage in (None, 50, 58, 65, 72)
    captures = []
    trimmed = ['struct DenoiserSpatialCurrentAccumulator' in
               (tree/'lib/lighting/denoiser/atrous_filter.glsl').read_text(encoding='utf-8')
               for tree in trees]
    trimmed_preparation = ['independentCurrentSignal.virtualDistance = 0.0;' in
                           (tree/'lib/lighting/denoiser/variance_prepare.glsl').read_text(encoding='utf-8')
                           for tree in trees]
    runners, retained = [], []
    preparations = []
    for tree in trees:
        image_scratch = 'denoiserScratchToImage' in (tree/'lib/lighting/denoiser/scratch_io.glsl').read_text()
        state = frame_state(width, height)
        projection = np.array([[1.2,0,0.001,0],[0,1.7,-0.002,0],
                               [0,0,-1.0001,-0.2],[0,0,-1,0]], dtype='f4')
        for offset in (64,128): state[offset:offset+64] = projection.T.tobytes()
        frame = ctx.buffer(state); frame.bind_to_storage_buffer(1)
        buffers = {binding: ctx.buffer(reserve=stride*16*planes)
                   for binding, planes in ((0,6),(2,8 if image_scratch else 10),(3,5),(4,4),(6,2))}
        for binding, buffer in buffers.items():
            buffer.clear(); buffer.bind_to_storage_buffer(binding)
        setup = ctx.compute_shader(gl_source(tree, fixture, {'DEBUG_VIEW': debug_view}))
        setup.run((width+7)//8,(height+7)//8); ctx.memory_barrier()
        textures = {i:ctx.texture((width,height),4,dtype='u4') for i in range(3,7)}
        scratch_pair = [ctx.texture((width,height),4,dtype='f4') for _ in range(2)] if image_scratch else []
        class ForwardCaptures(dict):
            def __setitem__(self, key, value):
                capture_sink(tree, key, value)
        capture = {} if capture_sink is None else ForwardCaptures()
        for stage in stages:
            entry=f'composite{stage}.csh'
            source=gl_source(tree,entry, {'DEBUG_VIEW': debug_view})
            shader=ctx.compute_shader(source)
            if 'resolution' in shader: shader['resolution'].value=(float(width),float(height))
            writes=[]
            for i, texture in textures.items():
                sampler=f'colortex{i}'; target=f'colorimg{i}'
                if sampler in shader: shader[sampler].value=i; texture.use(i)
                if target in shader:
                    texture.bind_to_image(i,read=False,write=True); writes.append(i)
            for n, texture in enumerate(scratch_pair):
                name=('bloomAtlas','bloomBlur')[n]
                if name+'_Sampler' in shader:
                    shader[name+'_Sampler'].value=7+n; texture.use(7+n)
                if name in shader:
                    shader[name].value=n; texture.bind_to_image(n,read=False,write=True)
            gx,gy,_=compute_group_size(shader)
            frame.bind_to_storage_buffer(1)
            for binding, buffer in buffers.items(): buffer.bind_to_storage_buffer(binding)
            if stage == timing_stage:
                # Resolve overwrites the raw/filtered histories it consumes.
                # Save those inputs on the GPU and restore outside each query.
                snapshots = {}
                if stage in (58,72):
                    binding = 2 if stage==58 else 3
                    snapshots[binding] = ctx.buffer(reserve=buffers[binding].size)
                    ctx.copy_buffer(snapshots[binding], buffers[binding])
                    ctx.memory_barrier()
            # This dispatch prepares state or captures correctness. Timing it
            # alongside CPU readbacks would measure changing buffer residency.
            shader.run((width+gx-1)//gx,(height+gy-1)//gy); ctx.memory_barrier()
            if timing_stage is None:
                for i in writes: capture[f'{entry}/color{i}']=textures[i].read()
            if timing_stage is None and (stage in (50,65) or 51<=stage<=56 or 66<=stage<=71):
                plane = 0 if stage in (50,65) else (1 if (stage-(51 if stage<60 else 66))%2==0 else 0)
                if scratch_pair:
                    words=np.frombuffer(scratch_pair[plane].read(),dtype='u4')-np.uint32(0x00800000)
                else:
                    words=np.frombuffer(buffers[2].read(),dtype='u4').reshape(-1,4)[(8+plane)*stride+tiled_indices(width,height)]
                capture[f'{entry}/independent']=words.tobytes()
            if stage == 50 and timing_stage is None:
                geometry_words=np.frombuffer(textures[3].read(),dtype='u4').reshape(-1,4)
                assert geometry_words[0,0] == np.float32(-1).view('u4')
                assert geometry_words[1,1] == 0x80008000
            if stage == timing_stage:
                # Preparation only reads raw/temporal inputs and overwrites
                # separate outputs, so repeated dispatches do not advance history.
                def run(shader=shader, frame=frame, buffers=buffers,
                        textures=textures, writes=writes, gx=gx, gy=gy, scratch_pair=scratch_pair):
                    frame.bind_to_storage_buffer(1)
                    for binding, buffer in buffers.items(): buffer.bind_to_storage_buffer(binding)
                    for i, texture in textures.items():
                        if f'colortex{i}' in shader: texture.use(i)
                        if i in writes: texture.bind_to_image(i, read=False, write=True)
                    for n, texture in enumerate(scratch_pair):
                        texture.use(7+n); texture.bind_to_image(n,read=False,write=True)
                    shader.run((width+gx-1)//gx, (height+gy-1)//gy)
                    ctx.memory_barrier()
                runners.append(run)
                def restore(snapshots=snapshots, buffers=buffers):
                    for binding, snapshot in snapshots.items(): ctx.copy_buffer(buffers[binding],snapshot)
                    if snapshots: ctx.memory_barrier()
                preparations.append(restore)
                retained.extend(snapshots.values())
                retained.extend([shader, frame, setup, *buffers.values(), *textures.values(), *scratch_pair])
                break
            shader.release()
        if timing_stage is None:
            for binding, buffer in buffers.items():
                # Plane 8/9 moved to images; every producer is compared above.
                capture[f'final/buffer{binding}']=buffer.read(size=8*stride*16) if binding==2 else buffer.read()
        captures.append(capture)
        if timing_stage is None:
            for resource in [frame,setup,*buffers.values(),*textures.values(),*scratch_pair]: resource.release()
    if capture_sink is not None:
        return {'capture_only': True, 'comparison_performed': False,
                'dimensions': [width, height], 'trees': list(map(str, trees))}
    checks={}
    for key in captures[0]:
        if key not in captures[1]:
            assert key in ('composite56.csh/color4', 'composite71.csh/color5'), key
            checks[key]='dead final proposal image removed'
            continue
        a,b=captures[0][key],captures[1][key]
        stage_match = re.match(r'composite(\d+)\.csh/(color[45]|independent)$', key)
        if any(trimmed) and stage_match:
            stage = int(stage_match[1])
            spatial = 51 <= stage <= 56 or 66 <= stage <= 71
            preparation = stage in (50,65) and any(trimmed_preparation)
            if spatial or preparation:
                stream = 'independent' if stage_match[2] == 'independent' else 'proposal'
                a, b = [spatial_consumed_words(data, stream, flag)
                        for data, flag in zip((a,b), trimmed if spatial else trimmed_preparation)]
        if key == 'final/buffer6':
            old, new = [np.frombuffer(data,dtype='u4').reshape(2,stride,4) for data in (a,b)]
            if debug_view == 0:
                assert not np.any(new), 'default shader wrote diagnostic data'
                checks[key]='no diagnostic writes in default view'
                continue
            if debug_view in (23,30,36):
                lane={23:1,30:0,36:2}[debug_view]
                old, new = old[0,:,lane], new[0,:,lane]
            elif debug_view in (24,25,37,38):
                mask=0xffff if debug_view in (24,25) else 0xffff0000
                old, new = old[0,:,3]&mask, new[0,:,3]&mask
            elif debug_view in (31,32,35):
                old, new = old[1], new[1]
            else:
                raise AssertionError(('unmapped diagnostic view', debug_view))
            assert np.array_equal(old,new), ('changed diagnostic output',debug_view)
            checks[key]='consumed diagnostic lanes bitwise equal'
            continue
        # Exact ABI comparison includes the geometry and history metadata.
        changed=np.count_nonzero(np.frombuffer(a,dtype='u4') != np.frombuffer(b,dtype='u4'))
        if changed and signal_half_ulp_tolerance:
            rounding = check_signal_half_rounding(a, b, key, stride, signal_half_ulp_tolerance)
            if rounding is not None:
                checks[key] = rounding
                continue
        if changed:
            folder=ROOT/'temp/allpass'
            (folder/'mismatch_baseline.bin').write_bytes(a)
            (folder/'mismatch_current.bin').write_bytes(b)
        assert changed==0, (key,int(changed))
        checks[key]='bitwise_equal'
    if timing_stage is not None:
        result = {'entry': f'composite{timing_stage}.csh',
                  **timed_pair(ctx, runners, repetitions=1 if timing_stage in (58,72) else 8,
                               preparations=preparations),
                  'validation': 'separate denoiser_chain run; no readbacks before timing'}
        for resource in retained: resource.release()
        print(json.dumps(result), flush=True)
        return result
    print(json.dumps({'denoiser_chain': len(checks), 'result':'passed',
                      'signal_half_ulp_tolerance':signal_half_ulp_tolerance}),flush=True)
    return {'checks':checks, 'debug_view':debug_view, 'validation_only':True,
            'signal_half_ulp_tolerance':signal_half_ulp_tolerance}


def auxiliary_math(ctx, trees):
    count=65536
    fixture=ROOT/'temp/allpass/post_math.csh'
    fixture.write_text('''#version 430 core
#define main unusedCacheTemporalMain
#include "/post/temporal_radiance_cache.glsl"
#undef main
#include "/lib/post_processing/tonemap.glsl"
layout(std430,binding=7) buffer TestOutput { vec4 values[]; } testOutput;
layout(std430,binding=8) readonly buffer TestInput { vec4 values[]; } testInput;
void main() {
    if (gl_GlobalInvocationID.z != 0u) return;
    uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*256u
        +gl_GlobalInvocationID.z*256u*256u;
    RadianceCache a=emptyCache(); a.W=1.0; a.M=1.0;
    a.maxent=radiance_to_rgb_maxent(vec3(1),vec3(1,0,0));
    RadianceCache b=a;
    b.maxent=radiance_to_rgb_maxent(vec3(3),vec3(-1,0,0));
    RadianceCache r=resampleTemporalRadiance(a,b,ivec3(i,-int(i),17),i);
    RadianceCache first=resampleTemporalRadiance(a,emptyCache(),ivec3(i),i);
    RadianceCache black=a; black.maxent=radiance_to_rgb_maxent(vec3(0),vec3(0));
    RadianceCache dark=resampleTemporalRadiance(black,black,ivec3(i),i);
    RadianceCache fading=resampleTemporalRadiance(black,a,ivec3(i),i);
    testOutput.values[3u*i]=vec4(r.maxent.maxentR.w*r.W,
        r.maxent.maxentR.x,first.maxent.maxentR.w*first.W,
        radianceCacheRisRandom(ivec3(i,-int(i),17),i));
    testOutput.values[3u*i+1u]=vec4(dark.M,dark.W,
        fading.maxent.maxentR.w*fading.W,r.M);
    testOutput.values[3u*i+2u]=vec4(TonyMcMapface_Tiny(
        apply_shadow_toe(testInput.values[i].rgb,EXPOSURE_CURVE_K)),1);
}
''',encoding='utf-8')
    rng=np.random.default_rng(387)
    hdr=np.exp(rng.uniform(-15,18,(count,4))).astype('f4')
    src=ctx.buffer(hdr.tobytes());src.bind_to_storage_buffer(8)
    outputs=[]
    for tree in trees:
        shader=ctx.compute_shader(gl_source(tree,fixture))
        dst=ctx.buffer(reserve=count*3*16);dst.bind_to_storage_buffer(7)
        # The production cache pass has a 4x4x4 local workgroup.
        shader.run(64,64,1);ctx.memory_barrier()
        # The fixture rejects the other three z slices before any SSBO access.
        outputs.append(np.frombuffer(dst.read(),dtype='f4').reshape(count,3,4).copy())
        shader.release();dst.release()
    a,b=outputs
    assert np.max(np.abs(b[:,0,0]-2)) < 1e-6
    assert np.all(b[:,0,2] == 1)
    assert np.all((b[:,0,3]>=0)&(b[:,0,3]<1))
    assert np.all(b[:,1] == [2,1,0.5,2])
    selection=float(np.mean(b[:,0,1]<0))
    assert abs(selection-0.75)<0.007
    tone_error=float(np.max(np.abs(a[:,2]-b[:,2])))
    assert tone_error<3e-6
    src.release()
    row={'cases':count,'RIS_history_selection_frequency':selection,
         'RIS_expected_frequency':0.75,'first_black_and_normalization':'passed',
         'tonemap_max_absolute_error':tone_error}
    print(json.dumps(row),flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT/'temp/allpass/baseline_shaders')
    parser.add_argument('--width', type=int, default=257)
    parser.add_argument('--height', type=int, default=145)
    parser.add_argument('--label', default='allpass', help='Report directory beneath temp/')
    parser.add_argument('--signal-half-ulp', type=int, choices=(0,1), default=0,
                        help='Explicit FP16 lighting-only ULP bound; all metadata remains exact')
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error('image dimensions must be positive')
    if not (args.baseline_dir/'lib/settings.glsl').is_file():
        parser.error('--baseline-dir must point to a complete pre-edit shader tree')
    output = (ROOT/'temp'/args.label).resolve()
    if not output.is_relative_to((ROOT/'temp').resolve()): parser.error('label must stay under temp/')
    output.mkdir(parents=True, exist_ok=True)
    (ROOT/'temp/allpass').mkdir(parents=True, exist_ok=True)  # temporary GLSL fixtures
    ctx = moderngl.create_standalone_context(require=430)
    trees = [args.baseline_dir.resolve(), ROOT/'shaders']
    started = time.time()
    report = {'renderer': ctx.info['GL_RENDERER'], 'size': [args.width, args.height],
              'baseline_dir': str(trees[0]),
              'atrous': atrous(ctx, trees, args.width, args.height),
              'bloom': bloom(ctx, trees, args.width, args.height),
              'exposure': exposure(ctx, trees),
              'denoiser_chain': denoiser_chain(ctx, trees, args.width, args.height,
                                               signal_half_ulp_tolerance=args.signal_half_ulp),
              'variance_preparation': [denoiser_chain(ctx, trees, args.width, args.height, stage,
                                                     signal_half_ulp_tolerance=args.signal_half_ulp)
                                       for stage in (50,65)],
              'history_resolve': [denoiser_chain(ctx, trees, args.width, args.height, stage,
                                                signal_half_ulp_tolerance=args.signal_half_ulp)
                                  for stage in (58,72)],
              'auxiliary_math': auxiliary_math(ctx, trees)}
    report['elapsed_seconds'] = time.time()-started
    (output/f'gpu_{args.width}x{args.height}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
