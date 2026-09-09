"""Time consecutive production spatial passes with real signal ping-pong.

The two source trees receive identical synthetic inputs. GPU copies restore the
inputs outside each query; one query encloses all five or six dispatches. Query
results and signal readbacks are collected only after the complete AB/BA batch.
This is a synthetic chain benchmark, not an in-game frame-time measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import moderngl
import numpy as np

import audit_post_pipeline as audit


ROOT = Path(__file__).resolve().parents[1]
IRIS_BARRIER = 0x2028  # SHADER_STORAGE | SHADER_IMAGE_ACCESS | TEXTURE_FETCH


def oct_words(normals):
    p = normals[..., :2] / np.abs(normals).sum(axis=-1, keepdims=True)
    folded = (1.0 - np.abs(p[..., ::-1])) * np.where(p >= 0, 1.0, -1.0)
    p = np.where(normals[..., 2:3] < 0, folded, p)
    uv = np.rint(np.clip(p * 0.5 + 0.5, 0, 1) * 65535).astype('u4')
    return uv[..., 0] | (uv[..., 1] << np.uint32(16))


def fixture(width, height, kind):
    rng = np.random.default_rng(108011)
    y, x = np.indices((height, width))
    u, v = x / width, y / height
    yaw, pitch = 0.47, -0.31
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rotation = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    camera = np.eye(4, dtype='f4')
    camera[:3, :3] = rotation
    frame = audit.frame_state(width, height)
    frame[:64] = camera.T.copy().tobytes()  # GLSL matrices are column-major.
    view_ray = np.stack(((2*u-1+0.001)/1.2, (2*v-1-0.002)/1.7,
                         -np.ones_like(u)), axis=-1)
    view_ray /= np.linalg.norm(view_ray, axis=-1, keepdims=True)
    normals_view = np.broadcast_to(np.array([0.14, -0.07, 1.0]), view_ray.shape).copy()
    normals_view /= np.linalg.norm(normals_view, axis=-1, keepdims=True)
    distance = np.where((u > 0.38) & (u < 0.68) & (v > 0.2), -4.0, -8.0)
    distance /= np.sum(normals_view * view_ray, axis=-1)
    normals_world = normals_view @ rotation
    direction = np.stack((0.16*np.sin(3*u), 0.13*np.cos(4*v),
                          0.5+0.03*np.sin(5*u+v)), axis=-1)
    energy = (1.2+5*u+3*v) * np.where(u < 0.5, 1.0, 1.8)
    energy *= rng.uniform(0.9, 1.1, (height, width))
    if kind == 'mixed':
        normals_world = rng.normal(size=(height, width, 3))
        normals_world /= np.linalg.norm(normals_world, axis=-1, keepdims=True)
        direction = rng.normal(size=(height, width, 3))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        direction *= rng.uniform(0.0, 0.9, (height, width, 1))
        energy = np.exp(rng.uniform(-5, 7, (height, width)))
    proposal = np.empty((height, width, 8), dtype='f4')
    proposal[..., :3] = direction * energy[..., None]
    proposal[..., 3] = energy
    proposal[..., 4] = 0.06 * energy * np.sin(3*u)
    proposal[..., 5] = -0.04 * energy * np.cos(2*v)
    proposal[..., 6] = np.sqrt(energy) * 0.7
    proposal[..., 7] = distance * (1.0+0.03*np.sin(4*u+v))
    independent = proposal.copy()
    independent[..., :6] *= (0.75+0.07*np.sin(5*u))[..., None]
    independent[..., 6] *= 1.4
    independent[..., 7] *= 1.15
    proposal[(x+3*y) % 71 == 0, 6] = -2.0
    independent[(2*x+y) % 53 == 0, 6] = -2.0
    proposal[(3*x+y) % 149 == 0, 6] = -1.0
    independent[(x+5*y) % 173 == 0, 6] = -1.0
    geometry = np.zeros((height, width, 4), dtype='u4')
    geometry[..., 0] = distance.astype('f4').view('u4')
    geometry[..., 1] = oct_words(normals_world)
    roughness = np.where(u < 1/3, 0.0, np.where(u < 2/3, 0.35, 1.0))
    if kind == 'mixed':
        roughness = np.choose((x+y) % 3, [0.0, 0.35, 1.0])
    geometry[..., 3] = audit.half_words(np.stack((roughness,
        1.0+(x+y) % 12), axis=-1))[..., 0]
    geometry[v < 0.04, 0] = np.float32(-1.0).view('u4')
    return frame, geometry, audit.half_words(proposal), audit.half_words(independent)


def populate_fixture_rays(ctx, tree, inputs):
    """Fill c3.z with the production GPU encoder, outside all timing queries.

    Both trees receive identical geometry. Older spatial passes ignore z; newer
    passes consume it. CPU oct encoding can differ at rounding boundaries, so
    use the actual ray reconstruction, frame layout and codec here.
    """
    frame, geometry, proposal, independent = inputs
    height, width = geometry.shape[:2]
    source = audit.gl_source(tree, 'composite51.csh')
    source, count = re.subn(r'\bvoid\s+main\s*\(\s*\)',
                           'void fixtureUnusedSpatialMain()', source)
    assert count >= 1, ('missing entry point', count)
    source += '''
layout(binding=6, rgba32ui) uniform uimage2D fixtureGeometry;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(fixtureGeometry)))) return;
    uvec4 words = imageLoad(fixtureGeometry, p);
    words.z = encodeNormalU(reconstructPrimaryRay(uvec2(p)));
    imageStore(fixtureGeometry, p, words);
}
'''
    if (tree/'lib/lighting/denoiser/ray_scale.glsl').exists():
        source = source.replace('words.z = encodeNormalU(reconstructPrimaryRay(uvec2(p)));',
            'words.z = floatBitsToUint(denoiserPrimaryRayScale(p));')
    shader = ctx.compute_shader(source)
    state = ctx.buffer(frame)
    texture = ctx.texture((width, height), 4, geometry.tobytes(), dtype='u4')
    try:
        state.bind_to_storage_buffer(1)
        texture.bind_to_image(6, read=True, write=True)
        gx, gy, _ = audit.compute_group_size(shader)
        shader.run((width+gx-1)//gx, (height+gy-1)//gy, 1)
        ctx.memory_barrier(IRIS_BARRIER)
        result = np.frombuffer(texture.read(), dtype='u4').reshape(geometry.shape).copy()
    finally:
        texture.release()
        state.release()
        shader.release()
    return frame, result, proposal, independent


class SpatialChain:
    def __init__(self, ctx, tree, domain, steps, width, height, inputs):
        self.ctx, self.resources = ctx, []
        self.width, self.height, self.steps = width, height, steps
        self.stride = ((width+7)//8) * ((height+7)//8) * 64
        self.indices = audit.tiled_indices(width, height)
        frame, geometry, proposal, independent = inputs
        self.frame = self.keep(ctx.buffer(frame))
        self.geometry = self.keep(ctx.texture((width, height), 4, geometry.tobytes(), dtype='u4'))
        self.proposal = [self.keep(ctx.texture((width, height), 4, dtype='u4')) for _ in range(2)]
        seed_proposal = self.keep(ctx.texture((width, height), 4, proposal.tobytes(), dtype='u4'))
        self.seed_proposal_fbo = self.keep(ctx.framebuffer([seed_proposal]))
        self.proposal_fbo = self.keep(ctx.framebuffer([self.proposal[0]]))
        first = 51 if domain == 'diffuse' else 66
        self.entries = [f'composite{i}.csh' for i in range(first, first+steps)]
        sources = [audit.gl_source(tree, entry) for entry in self.entries]
        self.trimmed = 'struct DenoiserSpatialCurrentAccumulator' in sources[0]
        self.source_hashes = {entry: hashlib.sha256(source.encode('utf-8')).hexdigest()
                              for entry, source in zip(self.entries, sources)}
        self.shaders = [self.keep(ctx.compute_shader(source)) for source in sources]
        # A diagnostic can eliminate all scratch reads while retaining stores.
        # Linked output images still require the image ABI in that case.
        self.image_scratch = any('bloomAtlas_Sampler' in s or 'bloomBlur_Sampler' in s
                                 or 'bloomAtlas' in s or 'bloomBlur' in s
                                 for s in self.shaders)
        self.scratch = None
        self.buffer = None
        self.debug = None
        if self.image_scratch:
            self.scratch = [self.keep(ctx.texture((width, height), 4, dtype='f4')) for _ in range(2)]
            encoded = (independent + np.uint32(0x00800000)).tobytes()
            seed_scratch = self.keep(ctx.texture((width, height), 4, encoded, dtype='f4'))
            self.seed_scratch_fbo = self.keep(ctx.framebuffer([seed_scratch]))
            self.scratch_fbo = self.keep(ctx.framebuffer([self.scratch[0]]))
        else:
            self.buffer = self.keep(ctx.buffer(reserve=10*self.stride*16))
            initial_plane = np.zeros((self.stride, 4), dtype='u4')
            initial_plane[self.indices] = independent.reshape(-1, 4)
            self.seed_buffer = self.keep(ctx.buffer(initial_plane.tobytes()))
        if any('DebugBuffer' in shader for shader in self.shaders):
            self.debug = self.keep(ctx.buffer(reserve=2*self.stride*16))
        self.groups = []
        self.outputs = []
        for index, shader in enumerate(self.shaders):
            shader['colortex3'].value = 0
            signal_name = ('colortex4' if domain == 'diffuse' else 'colortex5')
            if index % 2:
                signal_name = 'colortex5' if signal_name == 'colortex4' else 'colortex4'
            shader[signal_name].value = 1
            output_name = 'colorimg5' if signal_name == 'colortex4' else 'colorimg4'
            self.outputs.append(int(output_name[8:]) if output_name in shader else None)
            if self.scratch:
                for n, name in enumerate(('bloomAtlas', 'bloomBlur')):
                    if name in shader:
                        shader[name].value = n
                    if name+'_Sampler' in shader:
                        shader[name+'_Sampler'].value = 7+n
            gx, gy, gz = audit.compute_group_size(shader)
            if gz != 1:
                raise ValueError(('unexpected z workgroup', self.entries[index], gz))
            self.groups.append(((width+gx-1)//gx, (height+gy-1)//gy, 1))

    def keep(self, resource):
        self.resources.append(resource)
        return resource

    def prepare(self):
        self.ctx.copy_framebuffer(self.proposal_fbo, self.seed_proposal_fbo)
        if self.scratch:
            self.ctx.copy_framebuffer(self.scratch_fbo, self.seed_scratch_fbo)
        else:
            self.ctx.copy_buffer(self.buffer, self.seed_buffer, size=self.stride*16,
                                 write_offset=8*self.stride*16)
        self.ctx.memory_barrier(IRIS_BARRIER)

    def run(self):
        self.frame.bind_to_storage_buffer(1)
        self.geometry.use(0)
        if self.buffer:
            self.buffer.bind_to_storage_buffer(2)
        if self.debug:
            self.debug.bind_to_storage_buffer(6)
        for index, (shader, groups) in enumerate(zip(self.shaders, self.groups)):
            self.proposal[index % 2].use(1)
            if self.outputs[index] is not None:
                self.proposal[1-index % 2].bind_to_image(self.outputs[index], read=True, write=True)
            if self.scratch:
                for n, texture in enumerate(self.scratch):
                    texture.use(7+n)
                target = 1-index % 2
                self.scratch[target].bind_to_image(target, read=True, write=True)
            # Match Iris ComputeProgram (default concurrent-compute disabled)
            # and CompositeRenderer: the same visibility barrier on both sides.
            self.ctx.memory_barrier(IRIS_BARRIER)
            shader.run(*groups)
            self.ctx.memory_barrier(IRIS_BARRIER)

    def read_outputs(self):
        final_index = self.steps % 2
        if self.scratch:
            independent = np.frombuffer(self.scratch[final_index].read(), dtype='u4')
            independent = independent - np.uint32(0x00800000)
        else:
            raw = self.buffer.read(size=self.stride*16, offset=(8+final_index)*self.stride*16)
            independent = np.frombuffer(raw, dtype='u4').reshape(-1, 4)[self.indices]
        result = {'independent': independent.reshape(-1, 4).copy()}
        if self.outputs[-1] is not None:
            result['proposal'] = np.frombuffer(self.proposal[final_index].read(), dtype='u4').reshape(-1, 4).copy()
        return result

    def release(self):
        for resource in reversed(self.resources):
            resource.release()


def differences(a, b):
    ah, bh = a.view('<f2').astype('f4'), b.view('<f2').astype('f4')
    finite = np.isfinite(ah) & np.isfinite(bh)
    absolute = np.abs(ah[finite]-bh[finite])
    scaled = absolute / np.maximum(1.0, np.abs(ah[finite]))
    return {'bitwise_equal': bool(np.array_equal(a, b)),
            'changed_u32_lanes': int(np.count_nonzero(a != b)),
            'changed_half_lanes': int(np.count_nonzero(ah != bh)),
            'nonfinite_half_lanes': [int(np.count_nonzero(~np.isfinite(ah))),
                                     int(np.count_nonzero(~np.isfinite(bh)))],
            'max_absolute_error': float(absolute.max(initial=0)),
            'max_scaled_error': float(scaled.max(initial=0)),
            'mean_scaled_error': float(scaled.mean()) if scaled.size else 0.0,
            'sha256': [hashlib.sha256(v.tobytes()).hexdigest() for v in (a, b)]}


def benchmark(ctx, trees, domain, steps, width, height, inputs, trials):
    inputs = populate_fixture_rays(ctx, trees[1], inputs)
    chains = [SpatialChain(ctx, tree, domain, steps, width, height, inputs) for tree in trees]
    try:
        deadline = time.perf_counter()+0.5
        while time.perf_counter() < deadline:
            for chain in chains:
                chain.prepare()
                chain.run()
            ctx.finish()
        queries = [[None, None] for _ in range(trials)]
        for trial in range(trials):
            for index in ([0, 1] if trial % 2 == 0 else [1, 0]):
                chains[index].prepare()
                query = ctx.query(time=True)
                with query:
                    chains[index].run()
                queries[trial][index] = query
        # No query elapsed access, texture read, or buffer read occurs before
        # this point. Each variant owns separate resources throughout the batch.
        ctx.finish()
        paired_ms = np.array([[q.elapsed/1e6 for q in pair] for pair in queries])
        outputs = [chain.read_outputs() for chain in chains]
        comparison = {name: differences(outputs[0][name], outputs[1][name])
                      for name in outputs[0].keys() & outputs[1].keys()}
        consumed = {}
        for name in comparison:
            values = [audit.spatial_consumed_words(output[name].tobytes(), name, chain.trimmed)
                      if any(c.trimmed for c in chains) else output[name].tobytes()
                      for output, chain in zip(outputs, chains)]
            consumed[name] = differences(*[np.frombuffer(value, dtype='u4').reshape(-1,4)
                                          for value in values])
        return {'domain': domain, 'steps': steps, 'entries': chains[0].entries,
                'baseline_ms': float(np.median(paired_ms[:, 0])),
                'candidate_ms': float(np.median(paired_ms[:, 1])),
                'paired_speedup_median': float(np.median(paired_ms[:, 0]/paired_ms[:, 1])),
                'p10_p90_ms': np.percentile(paired_ms, [10, 90], axis=0).T.tolist(),
                'paired_ms': paired_ms.tolist(), 'trials': trials,
                'scratch_storage': ['images' if c.image_scratch else 'ssbo_10_planes' for c in chains],
                'source_sha256': [c.source_hashes for c in chains],
                'groups': [c.groups for c in chains], 'comparison': comparison,
                'consumed_comparison': consumed,
                'unpaired_outputs': [sorted(set(outputs[n])-set(outputs[1-n])) for n in range(2)]}
    finally:
        for chain in chains:
            chain.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, default=ROOT/'shaders')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fixture', choices=('coherent', 'mixed'), default='coherent')
    parser.add_argument('--steps', type=int, choices=(5, 6), default=5)
    parser.add_argument('--trials', type=int, default=31)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.trials) < 1:
        parser.error('width, height, and trials must be positive')
    destination = (args.output or ROOT/'temp/bench11'/
        f'spatial_chain_{args.fixture}_{args.steps}_{args.width}x{args.height}.json').resolve()
    if not destination.is_relative_to((ROOT/'temp').resolve()):
        parser.error('all output artifacts must stay under temp/')
    trees = [args.baseline_dir.resolve(), args.candidate_dir.resolve()]
    ctx = moderngl.create_standalone_context(require=430)
    inputs = fixture(args.width, args.height, args.fixture)
    report = {'dimensions': [args.width, args.height], 'fixture': args.fixture,
              'trees': list(map(str, trees)), 'gpu': ctx.info['GL_RENDERER'],
              'driver': ctx.info['GL_VERSION'], 'barrier_bits': hex(IRIS_BARRIER),
              'image_binding': 'READ_WRITE', 'query_scope': 'consecutive spatial chain',
              'query_readback': 'after complete AB/BA batch; no output readbacks before timing',
              'camera_rotation': {'yaw': 0.47, 'pitch': -0.31},
              'validation_policy': 'report raw numerical differences; no acceptance threshold',
              'results': []}
    for domain in ('diffuse', 'reflection'):
        row = benchmark(ctx, trees, domain, args.steps, args.width, args.height, inputs, args.trials)
        report['results'].append(row)
        print(json.dumps({k: row[k] for k in ('domain', 'steps', 'baseline_ms',
                                            'candidate_ms', 'paired_speedup_median', 'comparison')}), flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(destination), flush=True)
    ctx.release()


if __name__ == '__main__':
    main()
