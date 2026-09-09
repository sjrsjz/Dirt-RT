// Separable bloom Gaussian. Axis and image names are compile-time policies.
// Every invocation reaches the shared-memory barrier, including edge lanes.
#include "/lib/post_processing/bloom.glsl"

const int BLOOM_BLUR_GROUP = 256;
const int BLOOM_BLUR_RADIUS = 16;
shared vec3 bloomBlurTile[BLOOM_BLUR_GROUP + 2 * BLOOM_BLUR_RADIUS];

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 atlasSize = imageSize(BLOOM_BLUR_INPUT);
    int lane = int(gl_LocalInvocationIndex);
    ivec2 origin = coord - BLOOM_BLUR_AXIS * (lane + BLOOM_BLUR_RADIUS);
    for (int i = lane; i < BLOOM_BLUR_GROUP + 2 * BLOOM_BLUR_RADIUS;
            i += BLOOM_BLUR_GROUP) {
        ivec2 source = origin + BLOOM_BLUR_AXIS * i;
        bloomBlurTile[i] = all(greaterThanEqual(source, ivec2(0)))
                && all(lessThan(source, atlasSize))
            ? bloomSafeFloat(imageLoad(BLOOM_BLUR_INPUT, source).rgb) : vec3(0.0);
    }
    // barrier() includes visibility of shared writes; no second memory fence.
    barrier();
    if (any(greaterThanEqual(coord, atlasSize))) return;

    int level;
    ivec2 regionMin, regionMax;
    bloomFindLOD(coord, atlasSize, level, regionMin, regionMax);
    if (level < 0) {
        imageStore(BLOOM_BLUR_OUTPUT, coord, vec4(0.0));
        return;
    }
    vec3 sum = bloomBlurTile[lane + BLOOM_BLUR_RADIUS];
    if (level == 0) {
        imageStore(BLOOM_BLUR_OUTPUT, coord, vec4(sum, 1.0));
        return;
    }

    float greenSigma = 0.5 * float(level);
    vec3 sigmaScale = bloomDiffusionSigmaScale();
    vec3 rgbSigma = greenSigma * sigmaScale;
    vec3 coefficient = vec3(LOG2_E / (2.0 * greenSigma * greenSigma))
        / (sigmaScale * sigmaScale);
    int radius = clamp(int(ceil(3.0 * max(rgbSigma.r,
        max(rgbSigma.g, rgbSigma.b)))), 1, BLOOM_BLUR_RADIUS);
    vec3 weightSum = vec3(1.0);
    // The even Gaussian uses one exponential for both sides. The denominator
    // includes out-of-region weights: preserve the atlas's zero-extension ABI.
    for (int offset = 1; offset <= radius; ++offset) {
        vec3 w = exp2(-float(offset * offset) * coefficient);
        weightSum += 2.0 * w;
        ivec2 delta = BLOOM_BLUR_AXIS * offset;
        vec3 pair = vec3(0.0);
        if (all(greaterThanEqual(coord - delta, regionMin)))
            pair += bloomBlurTile[lane + BLOOM_BLUR_RADIUS - offset];
        if (all(lessThanEqual(coord + delta, regionMax)))
            pair += bloomBlurTile[lane + BLOOM_BLUR_RADIUS + offset];
        sum += pair * w;
    }
    imageStore(BLOOM_BLUR_OUTPUT, coord,
        vec4(bloomSafeFloat(sum / weightSum), 1.0));
}
