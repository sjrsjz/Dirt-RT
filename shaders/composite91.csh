#version 430 core

// One-dispatch 3x3 bloom pyramid. Each group builds an L0..L3 tile locally;
// the last completed group builds the small L4..L8 tail.
layout(local_size_x = 16, local_size_y = 16) in;
// L0 is half resolution and one 16x16 physical group owns a 32x32 L0 tile.
// A quarter-resolution dispatch therefore launches only potentially useful
// groups (the final partial group is rejected by groupCount when necessary).
const vec2 workGroupsRender = vec2(0.25, 0.25);
layout(rgba32f) uniform coherent image2D bloomAtlas;
uniform sampler2D colortex0;

#include "/lib/post_processing/bloom.glsl"
#include "/lib/buffers/frame_data.glsl"

const int BLOOM_TILE = 32;
const int BLOOM_LOCAL_LEVELS = 4;
// Proven maxima for a 32x32 logical tile and the native i/N mapping, including
// odd resolutions. RGB is stored as separate FP32 scalar/vector arrays so
// vec3 array padding cannot push shared memory beyond the 32 KiB budget.
const int BLOOM_L0_W = 46;
const int BLOOM_L1_W = 22;
const int BLOOM_L2_W = 10;

shared vec2 smL0RG[BLOOM_L0_W * BLOOM_L0_W];
shared float smL0B[BLOOM_L0_W * BLOOM_L0_W];
shared vec2 smL1RG[BLOOM_L1_W * BLOOM_L1_W];
shared float smL1B[BLOOM_L1_W * BLOOM_L1_W];
shared vec2 smL2RG[BLOOM_L2_W * BLOOM_L2_W];
shared float smL2B[BLOOM_L2_W * BLOOM_L2_W];
shared uint smTailOwner;

bool rectEmpty(ivec4 r) { return r.x > r.z || r.y > r.w; }
ivec4 emptyRect() { return ivec4(1, 1, 0, 0); }

ivec4 ownedRect(int level, ivec2 group, ivec2 size) {
    int tile = BLOOM_TILE >> level;
    ivec2 lo = group * tile;
    if (any(greaterThanEqual(lo, size)) || any(lessThanEqual(size, ivec2(0)))) return emptyRect();
    return ivec4(lo, min(lo + tile - 1, size - 1));
}

ivec4 unionRect(ivec4 a, ivec4 b) {
    if (rectEmpty(a)) return b;
    if (rectEmpty(b)) return a;
    return ivec4(min(a.xy, b.xy), max(a.zw, b.zw));
}

// Native ray-grid mapping. Primary rays are generated at pixel / resolution
// (without a +0.5 texel-centre offset), so a bloom sample j represents j/N as
// well. Endpoint mapping through (N-1) accumulates a half-texel error at high
// LODs. A one-sample dimension represents the whole parent and uses its centre.
vec2 mappedSource(vec2 p, ivec2 srcSize, ivec2 dstSize) {
    return vec2(
        dstSize.x > 1 ? p.x * float(srcSize.x) / float(dstSize.x) : 0.5 * float(max(srcSize.x - 1, 0)),
        dstSize.y > 1 ? p.y * float(srcSize.y) / float(dstSize.y) : 0.5 * float(max(srcSize.y - 1, 0)));
}

// Width, in source texels, represented by one destination texel.
vec2 mappedFootprint(ivec2 srcSize, ivec2 dstSize) {
    return vec2(
        dstSize.x > 1 ? float(srcSize.x) / float(dstSize.x) : float(srcSize.x),
        dstSize.y > 1 ? float(srcSize.y) / float(dstSize.y) : float(srcSize.y));
}

// Three-tap, linear-phase downsampling kernel. The two side weights carry the
// unrounded subpixel phase delta, so sum(w)=1 and sum(w*x)=centre exactly.
// This prevents the periodic half-texel drift produced by rounding the mapped
// coordinate before filtering. A wider final 5 -> 2 footprint uses stride 2
// while retaining exactly three samples and the same first moment.
ivec2 phaseStride(vec2 footprint) {
    return max(ivec2(1), ivec2(floor(0.5 * footprint)));
}

vec3 phaseWeights(float centre, int base, int stride) {
    float delta = centre - float(base);
    float phase = delta / max(float(stride), 1.0);
    return vec3(0.25 - 0.5 * phase, 0.5, 0.25 + 0.5 * phase);
}

ivec4 sourceFootprint(ivec4 dst, ivec2 srcSize, ivec2 dstSize) {
    if (rectEmpty(dst) || any(lessThanEqual(srcSize, ivec2(0))) ||
        any(lessThanEqual(dstSize, ivec2(0)))) return emptyRect();
    vec2 footprint = mappedFootprint(srcSize, dstSize);
    ivec2 stride = phaseStride(footprint);
    ivec2 lo = ivec2(floor(
        mappedSource(vec2(dst.xy), srcSize, dstSize) + 0.5)) - stride;
    ivec2 hi = ivec2(floor(
        mappedSource(vec2(dst.zw), srcSize, dstSize) + 0.5)) + stride;
    lo = max(lo, ivec2(0));
    hi = min(hi, srcSize - 1);
    return any(greaterThan(lo, hi)) ? emptyRect() : ivec4(lo, hi);
}

bool rectContains(ivec4 r, ivec2 p) {
    return !rectEmpty(r) && all(greaterThanEqual(p, r.xy)) && all(lessThanEqual(p, r.zw));
}

vec3 sampleScene3x3(ivec2 dst, ivec2 srcSize, ivec2 dstSize) {
    vec2 sp = mappedSource(vec2(dst), srcSize, dstSize);
    vec2 footprint = mappedFootprint(srcSize, dstSize);
    ivec2 stride = phaseStride(footprint);
    ivec2 base = ivec2(floor(sp + 0.5));
    vec3 wx = phaseWeights(sp.x, base.x, stride.x);
    vec3 wy = phaseWeights(sp.y, base.y, stride.y);
    vec3 sum = vec3(0.0);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        ivec2 p = base + ivec2(x * stride.x, y * stride.y);
        if (all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, srcSize))) {
            float w = wx[x + 1] * wy[y + 1];
            sum += bloomSafeFloat(texelFetch(colortex0, p, 0).rgb) * w;
        }
    }
    return sum;
}

vec3 loadL0(ivec2 p, ivec4 r) {
    ivec2 q = p - r.xy, extent = r.zw - r.xy + 1;
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, extent)) ||
        q.x >= BLOOM_L0_W || q.y >= BLOOM_L0_W) return vec3(0.0);
    int i = q.y * BLOOM_L0_W + q.x;
    return vec3(smL0RG[i], smL0B[i]);
}
vec3 loadL1(ivec2 p, ivec4 r) {
    ivec2 q = p - r.xy, extent = r.zw - r.xy + 1;
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, extent)) ||
        q.x >= BLOOM_L1_W || q.y >= BLOOM_L1_W) return vec3(0.0);
    int i = q.y * BLOOM_L1_W + q.x;
    return vec3(smL1RG[i], smL1B[i]);
}
vec3 loadL2(ivec2 p, ivec4 r) {
    ivec2 q = p - r.xy, extent = r.zw - r.xy + 1;
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, extent)) ||
        q.x >= BLOOM_L2_W || q.y >= BLOOM_L2_W) return vec3(0.0);
    int i = q.y * BLOOM_L2_W + q.x;
    return vec3(smL2RG[i], smL2B[i]);
}

#define SAMPLE_SHARED(result, loadFn, dst, srcSize, dstSize, srcRect) do { \
    vec2 _sp = mappedSource(vec2(dst), srcSize, dstSize); \
    vec2 _footprint = mappedFootprint(srcSize, dstSize); \
    ivec2 _stride = phaseStride(_footprint); \
    ivec2 _base = ivec2(floor(_sp + 0.5)); \
    vec3 _wx = phaseWeights(_sp.x, _base.x, _stride.x); \
    vec3 _wy = phaseWeights(_sp.y, _base.y, _stride.y); \
    vec3 _sum = vec3(0.0); \
    for (int _y = -1; _y <= 1; ++_y) for (int _x = -1; _x <= 1; ++_x) { \
        ivec2 _p = _base + ivec2(_x * _stride.x, _y * _stride.y); \
        if (all(greaterThanEqual(_p, ivec2(0))) && all(lessThan(_p, srcSize))) { \
            float _w = _wx[_x + 1] * _wy[_y + 1]; \
            _sum += loadFn(_p, srcRect) * _w; \
        } \
    } \
    result = _sum; \
} while (false)

vec3 sampleAtlas3x3(ivec2 dst, int srcLevel, int dstLevel, ivec2 atlasSize) {
    ivec2 srcSize = bloomSize(srcLevel, atlasSize);
    ivec2 dstSize = bloomSize(dstLevel, atlasSize);
    ivec2 srcOrigin = bloomOrigin(srcLevel, atlasSize);
    vec2 sp = mappedSource(vec2(dst), srcSize, dstSize);
    vec2 footprint = mappedFootprint(srcSize, dstSize);
    ivec2 stride = phaseStride(footprint);
    ivec2 base = ivec2(floor(sp + 0.5));
    vec3 wx = phaseWeights(sp.x, base.x, stride.x);
    vec3 wy = phaseWeights(sp.y, base.y, stride.y);
    vec3 sum = vec3(0.0);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        ivec2 p = base + ivec2(x * stride.x, y * stride.y);
        if (all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, srcSize))) {
            float w = wx[x + 1] * wy[y + 1];
            sum += bloomSafeFloat(imageLoad(bloomAtlas, srcOrigin + p).rgb) * w;
        }
    }
    return sum;
}

void main() {
    ivec2 atlasSize = imageSize(bloomAtlas);
    ivec2 levelSize[4] = ivec2[4](
        bloomSize(0, atlasSize), bloomSize(1, atlasSize),
        bloomSize(2, atlasSize), bloomSize(3, atlasSize));
    ivec2 group = ivec2(gl_WorkGroupID.xy);
    ivec2 groupCount = (levelSize[0] + BLOOM_TILE - 1) / BLOOM_TILE;

    // Iris dispatches over the full render size; only L0-owning groups count.
    if (any(greaterThanEqual(group, groupCount))) return;

    ivec4 own[4], needed[4];
    for (int l = 0; l < 4; ++l) own[l] = ownedRect(l, group, levelSize[l]);
    needed[3] = own[3];
    for (int l = 2; l >= 0; --l)
        needed[l] = unionRect(own[l], sourceFootprint(needed[l + 1], levelSize[l], levelSize[l + 1]));

    uint lane = gl_LocalInvocationIndex;
    ivec2 extent0 = needed[0].zw - needed[0].xy + 1;
    int count0 = rectEmpty(needed[0]) ? 0 : extent0.x * extent0.y;
    for (int i = int(lane); i < count0; i += 256) {
        ivec2 q = ivec2(i % extent0.x, i / extent0.x);
        ivec2 p = needed[0].xy + q;
        vec3 v = sampleScene3x3(p, textureSize(colortex0, 0), levelSize[0]);
        if (q.x < BLOOM_L0_W && q.y < BLOOM_L0_W)
        {
            int si = q.y * BLOOM_L0_W + q.x;
            smL0RG[si] = v.rg;
            smL0B[si] = v.b;
        }
        if (rectContains(own[0], p)) imageStore(bloomAtlas, bloomOrigin(0, atlasSize) + p, vec4(bloomSafeFloat(v), 1.0));
    }
    barrier();

    ivec2 extent1 = needed[1].zw - needed[1].xy + 1;
    int count1 = rectEmpty(needed[1]) ? 0 : extent1.x * extent1.y;
    for (int i = int(lane); i < count1; i += 256) {
        ivec2 q = ivec2(i % extent1.x, i / extent1.x);
        ivec2 p = needed[1].xy + q;
        vec3 v; SAMPLE_SHARED(v, loadL0, p, levelSize[0], levelSize[1], needed[0]);
        if (q.x < BLOOM_L1_W && q.y < BLOOM_L1_W)
        {
            int si = q.y * BLOOM_L1_W + q.x;
            smL1RG[si] = v.rg;
            smL1B[si] = v.b;
        }
        if (rectContains(own[1], p)) imageStore(bloomAtlas, bloomOrigin(1, atlasSize) + p, vec4(bloomSafeFloat(v), 1.0));
    }
    barrier();

    ivec2 extent2 = needed[2].zw - needed[2].xy + 1;
    int count2 = rectEmpty(needed[2]) ? 0 : extent2.x * extent2.y;
    for (int i = int(lane); i < count2; i += 256) {
        ivec2 q = ivec2(i % extent2.x, i / extent2.x);
        ivec2 p = needed[2].xy + q;
        vec3 v; SAMPLE_SHARED(v, loadL1, p, levelSize[1], levelSize[2], needed[1]);
        if (q.x < BLOOM_L2_W && q.y < BLOOM_L2_W)
        {
            int si = q.y * BLOOM_L2_W + q.x;
            smL2RG[si] = v.rg;
            smL2B[si] = v.b;
        }
        if (rectContains(own[2], p)) imageStore(bloomAtlas, bloomOrigin(2, atlasSize) + p, vec4(bloomSafeFloat(v), 1.0));
    }
    barrier();

    ivec2 extent3 = needed[3].zw - needed[3].xy + 1;
    int count3 = rectEmpty(needed[3]) ? 0 : extent3.x * extent3.y;
    for (int i = int(lane); i < count3; i += 256) {
        ivec2 q = ivec2(i % extent3.x, i / extent3.x);
        ivec2 p = needed[3].xy + q;
        vec3 v; SAMPLE_SHARED(v, loadL2, p, levelSize[2], levelSize[3], needed[2]);
        imageStore(bloomAtlas, bloomOrigin(3, atlasSize) + p, vec4(bloomSafeFloat(v), 1.0));
    }

    // Publish all image stores before the leader publishes group completion.
    memoryBarrierImage();
    barrier();
    if (lane == 0u) {
        uint before = atomicAdd(bloomCompletedGroups, 1u);
        uint expected = uint(groupCount.x * groupCount.y);
        smTailOwner = (before + 1u == expected) ? 1u : 0u;
    }
    barrier();
    if (smTailOwner == 0u) return;

    // The elected group acquires L3 and finishes only the small pyramid tail.
    memoryBarrierImage();
    for (int dstLevel = BLOOM_LOCAL_LEVELS; dstLevel <= 8; ++dstLevel) {
        ivec2 dstSize = bloomSize(dstLevel, atlasSize);
        int count = max(dstSize.x, 0) * max(dstSize.y, 0);
        for (int i = int(lane); i < count; i += 256) {
            ivec2 p = ivec2(i % dstSize.x, i / dstSize.x);
            vec3 v = sampleAtlas3x3(p, dstLevel - 1, dstLevel, atlasSize);
            imageStore(bloomAtlas, bloomOrigin(dstLevel, atlasSize) + p, vec4(bloomSafeFloat(v), 1.0));
        }
        memoryBarrierImage();
        barrier();
    }
}
