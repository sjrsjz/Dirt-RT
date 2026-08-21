#ifndef MAXENT_VARIANCE_PREPARE_GLSL
#define MAXENT_VARIANCE_PREPARE_GLSL

#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"

// Canonical variance-preparation contract shared by diffuse and specular.
//
// Source signal:
//   maxEntY      = (E[Y * direction], E[Y])
//   CoCg         = chroma paired with E[Y]
//   meanY2       = raw temporal second moment E[Y^2]
//   historyLength = Kish effective temporal sample count N_eff
//
// Outputs:
//   colortex3 = canonical RGBA32UI denoiser geometry documented in
//               maxent_spatial_geometry.glsl
//   colortex4 = canonical RGBA32UI MaxEnt signal documented in
//               maxent_spatial_signal.glsl
// The variance carried by colortex4 is a local squared-Bures uncertainty
// estimate for the temporally accumulated MaxEnt state. It is obtained by
// pulling the Gaussian Bures/W2 metric back to (v, omega), contracting that
// metric with the covariance information recoverable from E[Y^2], and is
// computed in F32 then stored as an FP16 square root. First-surface plane
// consistency constrains both variance pooling stages. The signal's upper
// metadata half contains the already constructed radial virtual distance, not
// reflection hit distance.
// Iris owns the physical colortex4 ping-pong backing between spatial passes.

#if !defined(MAXENT_VARIANCE_DIFFUSE) && !defined(MAXENT_VARIANCE_SPECULAR)
#error "Select one MaxEnt variance source"
#endif
#if defined(MAXENT_VARIANCE_DIFFUSE) && defined(MAXENT_VARIANCE_SPECULAR)
#error "Select only one MaxEnt variance source"
#endif

layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg3;

struct DenoiserVarianceSource {
    vec4 maxEntY;
    vec2 CoCg;
    float meanY2;
    float historyLength;
    float hitDistance;
};

struct DenoiserVarianceGeometry {
    vec3 surfaceNormal;
    vec3 primaryRay;
    float surfaceDistance;
    float virtualScale;
    float ggxAlpha;
    uint materialID;
    bool valid;
};

DenoiserVarianceSource denoiserVarianceEmptySource() {
    DenoiserVarianceSource source;
    source.maxEntY = vec4(0.0);
    source.CoCg = vec2(0.0);
    source.meanY2 = 0.0;
    source.historyLength = 1.0;
    source.hitDistance = 0.0;
    return source;
}

DenoiserVarianceGeometry denoiserVarianceDecodeGeometry(uvec4 words, ivec2 pixel) {
    DenoiserVarianceGeometry geometry;
    float distance = uintBitsToFloat(words.w);
    float ggxAlpha = unpackHalf2x16(words.y).x;
    geometry.primaryRay = reconstructPrimaryRay(uvec2(pixel));
    geometry.surfaceNormal = decodeNormalU(words.x);
    geometry.surfaceDistance = distance;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    geometry.virtualScale = 0.0;
    geometry.ggxAlpha = 1.0;
    #else
    ggxAlpha = clamp(ggxAlpha, 0.0, 1.0);
    float perceptualRoughness = sqrt(ggxAlpha);
    geometry.virtualScale = denoiserSpatialSpecularVirtualScale(
            geometry.primaryRay, geometry.surfaceNormal, perceptualRoughness);
    geometry.ggxAlpha = ggxAlpha;
    #endif
    geometry.materialID = words.y >> 16u;
    geometry.valid = distance >= 0.0 && !isnan(distance)
            && !isinf(distance);
    return geometry;
}

DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel) {
    return denoiserVarianceDecodeGeometry(readPrimaryGeometryWords(uvec2(pixel)), pixel);
}

uvec4 denoiserVariancePackSpatialGeometry(uvec4 primaryWords, DenoiserVarianceGeometry geometry) {
    float ggxAlpha = geometry.ggxAlpha;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    float signalRoughness = 1.0;
    #else
    float signalRoughness = sqrt(ggxAlpha);
    #endif
    uint packedRoughness = packHalf2x16(vec2(signalRoughness, 0.0));
    return uvec4(primaryWords.w, encodeNormalU(geometry.surfaceNormal),
        encodeNormalU(geometry.primaryRay), packedRoughness);
}

DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel) {
    DenoiserVarianceSource source;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    uvec4 words = readDiffuseSwapRaw(uvec2(pixel));
    source.maxEntY = vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
    source.CoCg = unpackHalf2x16(words.z);
    vec2 historyRootM2 = unpackHalf2x16(words.w);
    source.historyLength = historyRootM2.x;
    source.meanY2 = historyRootM2.y * historyRootM2.y;
    source.hitDistance = 0.0;
    #else
    uvec4 words = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, uvec2(pixel))];
    SpecularMaxEnt signal = unpackSpecularMaxEnt(words.xyz);
    source.maxEntY = signal.maxEntY;
    source.CoCg = signal.CoCg;
    vec2 rootM2History = unpackHalf2x16(words.w);
    source.meanY2 = rootM2History.x * rootM2History.x;
    source.historyLength = rootM2History.y;
    source.hitDistance = unpackHalf2x16(reflectBuffer.data[addr(SPEC_N_HISTGEO, uvec2(pixel))].w).x;
    #endif
    return source;
}

void denoiserVarianceStore(ivec2 pixel, DenoiserMaxEntSignal signal,
    uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry) {
    imageStore(colorimg3, pixel, denoiserVariancePackSpatialGeometry(primaryGeometryWords, geometry));
    imageStore(colorimg4, pixel, denoiserPackMaxEntSignal(signal));
}

void denoiserVarianceStoreInvalid(ivec2 pixel) {
    imageStore(colorimg3, pixel, uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u));
    imageStore(colorimg4, pixel, denoiserInvalidMaxEntSignalWords());
}

float denoiserVarianceSanitizeNonnegative(float value) {
    return isnan(value) || isinf(value) ? 0.0 : max(value, 0.0);
}

vec4 denoiserVarianceCanonicalMean(vec4 meanState) {
    if (any(isnan(meanState)) || any(isinf(meanState)))
        return vec4(0.0);
    vec3 directionalMean = meanState.xyz;
    float meanY = max(meanState.w, 0.0);
    float directionalLength2 = dot(directionalMean, directionalMean);
    if (directionalLength2 > meanY * meanY
            && directionalLength2 > 0.0)
        directionalMean *= meanY * inversesqrt(directionalLength2);
    return vec4(directionalMean, meanY);
}

float denoiserVarianceCanonicalMeanY2(vec4 meanState, float meanY2) {
    meanY2 = denoiserVarianceSanitizeNonnegative(meanY2);
    return max(meanY2, max(meanState.w * meanState.w, dot(meanState.xyz, meanState.xyz)));
}

// Local pullback of the Gaussian Bures/W2 metric to the 4-DOF
// MaxEnt parameterization theta = (v, omega).
//
// The Gaussian proxy is axisymmetric around n = v / |v|:
//   Sigma = p^2 (I - nn^T) + q^2 nn^T,
// with the same p/q construction used by the spatial Bures distance.
// For an infinitesimal perturbation dv = n dr + dv_perp and dOmega,
//
//   d_B^2 ~= Grr dr^2 + Gperp |dv_perp|^2
//              + 2 GrOmega dr dOmega + GOmegaOmega dOmega^2.
//
// Only E[Y^2] is stored, so the full covariance of (v, omega) is not
// identifiable. We therefore use the following closure:
//   * the exact scalar magnitudes available from the stored moments,
//       tr Cov[Y u] = E[Y^2] - |v|^2,
//       Var[Y]      = E[Y^2] - omega^2;
//   * the MaxEnt Gaussian proxy only to split tr Cov[Y u] into the radial
//     and two perpendicular components;
//   * Cov(dr, dOmega) = 0, because E[Y^2 u] is not stored.
//
// The returned value is in squared-Bures units.  It is the population-form
// uncertainty proxy, i.e. it carries the same (N - 1) / N bias as the raw
// second moments.  denoiserVarianceOfTemporalMean() converts it to the
// uncertainty of the temporal mean by dividing by N - 1.
float denoiserVarianceBuresPopulation(vec4 meanState, float meanY2) {
    meanState = denoiserVarianceCanonicalMean(meanState);
    meanY2 = denoiserVarianceCanonicalMeanY2(meanState, meanY2);

    vec3 v = meanState.xyz;
    float omega = meanState.w;
    float r2 = dot(v, v);
    float omega2 = omega * omega;

    // These two scalar covariance magnitudes are determined exactly by the
    // stored moments.  They are population-form (1/N-normalized) scatters.
    float vectorPopulationTrace = max(meanY2 - r2, 0.0);
    float energyPopulationVariance = max(meanY2 - omega2, 0.0);

    if (vectorPopulationTrace <= 0.0 && energyPopulationVariance <= 0.0)
        return 0.0;

    // A nonnegative Y with E[Y] == 0 must be identically zero.  This branch
    // is therefore only a numerical-degeneracy fallback.  At v -> 0 the
    // exact pullback metric tends to G_v = 1 and G_omega = 4/3.
    if (omega2 <= 1e-30)
        return vectorPopulationTrace
            + (4.0 / 3.0) * energyPopulationVariance;

    // Same closed-form axisymmetric covariance as denoiserSpatialMakeBuresData:
    //   s  = sqrt(4 omega^2 - 3/4 |v|^2)
    //   p2 = (2 omega^2 + omega s) / 9 - |v|^2 / 3
    //   q2 = p2 + |v|^2 / 2
    float rootArgument = max(4.0 * omega2 - 0.75 * r2, 0.0);
    float traceRoot = sqrt(rootArgument);
    float common_ = (2.0 * omega2 + omega * traceRoot) / 9.0;
    float p2 = max(common_ - r2 / 3.0, 0.0);
    float q2 = max(common_ + r2 / 6.0, 0.0);
    float shapeTrace = 2.0 * p2 + q2;

    if (traceRoot <= 0.0 || p2 <= 0.0 || q2 <= 0.0 || shapeTrace <= 0.0)
        return vectorPopulationTrace
            + (4.0 / 3.0) * energyPopulationVariance;

    // The true second directional moment E[Y^2 uu^T] is unavailable.  Use
    // the MaxEnt proxy only for its axisymmetric shape, while preserving the
    // observed total vector variance magnitude from E[Y^2].
    float radialPopulationVariance = vectorPopulationTrace * q2 / shapeTrace;
    float perpendicularPopulationTrace = max(
            vectorPopulationTrace - radialPopulationVariance, 0.0);

    float p = sqrt(p2);
    float q = sqrt(q2);
    float r = sqrt(r2);

    // Derivatives of p^2 and q^2 with respect to r = |v| and omega.
    // traceRoot is safely nonzero for realizable |v| <= omega, omega > 0.
    float dp2Dr = -omega * r / (12.0 * traceRoot) - (2.0 / 3.0) * r;
    float dq2Dr = -omega * r / (12.0 * traceRoot) + (1.0 / 3.0) * r;
    float dCommonDOmega = (4.0 * omega + traceRoot
            + 4.0 * omega2 / traceRoot) / 9.0;

    float dpDr = dp2Dr / (2.0 * p);
    float dqDr = dq2Dr / (2.0 * q);
    float dpDOmega = dCommonDOmega / (2.0 * p);
    float dqDOmega = dCommonDOmega / (2.0 * q);

    // Local Bures metric coefficients.  The perpendicular term contains the
    // covariance-orientation contribution in addition to the Euclidean mean
    // displacement.  Since q^2 - p^2 = |v|^2 / 2, it simplifies to r^2 /
    // (4 (p^2 + q^2)).
    float Grr = 1.0 + 2.0 * dpDr * dpDr + dqDr * dqDr;
    float Gperp = 1.0 + r2 / (4.0 * (p2 + q2));
    float GOmegaOmega = 2.0 * dpDOmega * dpDOmega
            + dqDOmega * dqDOmega;

    // The pullback also has
    //   GrOmega = 2 dp/dr dp/domega + dq/dr dq/domega,
    // but its covariance contraction requires Cov(dr, domega), equivalently
    // E[Y^2 u], which is not present in the compact signal.  The zero-cross
    // closure avoids inventing an unobservable sign/correlation.

    float buresPopulationVariance =
          Grr * radialPopulationVariance
        + Gperp * perpendicularPopulationTrace
        + GOmegaOmega * energyPopulationVariance;

    return denoiserVarianceSanitizeNonnegative(buresPopulationVariance);
}

float denoiserVarianceOfTemporalMean(vec4 meanState, float meanY2, float historyLength) {
    // Every population covariance term above carries the usual (N - 1) / N
    // bias.  Dividing the contracted Bures population uncertainty by N - 1
    // therefore gives the local squared-Bures uncertainty of the temporal
    // sample mean, under the covariance-shape closure above.
    if (historyLength <= 1.5) return 0.0;
    return denoiserVarianceBuresPopulation(meanState, meanY2)
        / (historyLength - 1.0);
}

DenoiserVarianceSource denoiserVarianceSanitizeSource(DenoiserVarianceSource source) {
    source.maxEntY = denoiserVarianceCanonicalMean(source.maxEntY);
    if (any(isnan(source.CoCg)) || any(isinf(source.CoCg)))
        source.CoCg = vec2(0.0);
    if (source.maxEntY.w <= 0.0) source.CoCg = vec2(0.0);
    source.meanY2 = denoiserVarianceCanonicalMeanY2(source.maxEntY, source.meanY2);
    source.historyLength = clamp(denoiserVarianceSanitizeNonnegative(source.historyLength), 1.0, 65504.0);
    source.hitDistance = clamp(denoiserVarianceSanitizeNonnegative(source.hitDistance),
            0.0, DENOISER_SPATIAL_FP16_MAX);
    return source;
}

const uint MAXENT_VARIANCE_TILE_SIZE = 16u;
const uint MAXENT_VARIANCE_HALO = 3u;
const uint MAXENT_VARIANCE_SHARED_WIDTH = 22u;
const uint MAXENT_VARIANCE_SHARED_AREA = 22u * 22u;

// First-surface distance and primary-ray direction reconstruct the sample
// position used by the plane test. Validity/material metadata is separate so
// the geometry domain occupies 12 bytes per tile entry instead of a uvec4.
shared uvec2 denoiserVarianceSurfaceTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMetadataTile[MAXENT_VARIANCE_SHARED_AREA];
shared uvec2 denoiserVarianceMeanTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMomentTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVariancePoolRequired;

// The exposed sigma range starts at 0.5, so the denominator is >= 0.25.
const float MAXENT_VARIANCE_KERNEL_DENOM = MAXENT_VARIANCE_KERNEL_SIGMA * MAXENT_VARIANCE_KERNEL_SIGMA;
const float MAXENT_VARIANCE_KERNEL_1D[4] = {
    1.0,
    exp(-0.5 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-2.0 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-4.5 / MAXENT_VARIANCE_KERNEL_DENOM)
    };

void denoiserVarianceWriteTile(uint index, DenoiserVarianceGeometry geometry,
    DenoiserVarianceSource source, bool valid) {
    source = denoiserVarianceSanitizeSource(source);
    denoiserVarianceSurfaceTile[index] = uvec2(floatBitsToUint(geometry.surfaceDistance),
            encodeNormalU(geometry.primaryRay));
    denoiserVarianceMetadataTile[index] = valid
        ? (0x80000000u | (geometry.materialID & 0x7fffffffu)) : 0u;
    denoiserVarianceMeanTile[index] = uvec2(
            packHalf2x16(clamp(source.maxEntY.xy, vec2(-65504.0),
                    vec2(65504.0))),
            packHalf2x16(clamp(source.maxEntY.zw, vec2(-65504.0),
                    vec2(65504.0))));
    denoiserVarianceMomentTile[index] = packHalf2x16(vec2(source.historyLength,
                min(sqrt(source.meanY2), 65504.0)));
}

void denoiserVarianceLoadTile(uint index, ivec2 pixel, ivec2 imageMax) {
    ivec2 clampedPixel = clamp(pixel, ivec2(0), imageMax);
    bool inBounds = all(equal(pixel, clampedPixel));
    DenoiserVarianceGeometry geometry = denoiserVarianceLoadGeometry(clampedPixel);
    bool valid = inBounds && geometry.valid;
    DenoiserVarianceSource source = valid
        ? denoiserVarianceLoadSource(clampedPixel) : denoiserVarianceEmptySource();
    denoiserVarianceWriteTile(index, geometry, source, valid);
}

bool denoiserVarianceTileValid(uint index) {
    return (denoiserVarianceMetadataTile[index] & 0x80000000u) != 0u;
}

uint denoiserVarianceTileMaterial(uint index) {
    return denoiserVarianceMetadataTile[index] & 0x7fffffffu;
}

float denoiserVarianceTileSurfacePlaneDepthExponent(uint index, float centerPlaneOffset,
    vec3 centerSurfaceNormal, float surfaceRejectionScale) {
    uvec2 words = denoiserVarianceSurfaceTile[index];
    return denoiserSpatialSurfacePlaneDepthExponent(centerPlaneOffset, centerSurfaceNormal,
        decodeNormalU(words.y), uintBitsToFloat(words.x), surfaceRejectionScale);
}

vec4 denoiserVarianceTileMean(uint index) {
    return vec4(unpackHalf2x16(denoiserVarianceMeanTile[index].x), unpackHalf2x16(denoiserVarianceMeanTile[index].y));
}

vec2 denoiserVarianceTileMomentHistory(uint index) {
    vec2 historyRootM2 = unpackHalf2x16(denoiserVarianceMomentTile[index]);
    return vec2(historyRootM2.y * historyRootM2.y, historyRootM2.x);
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 localPixel = gl_LocalInvocationID.xy;
    uint lane = gl_LocalInvocationIndex;
    ivec2 imageSize = ivec2(resolution_global);
    ivec2 imageMax = imageSize - 1;
    // Begin is nonnegative by its setting range. End is an independent
    // setting, so preserve an ordered smoothstep interval.
    float varianceHistoryBegin = MAXENT_VARIANCE_HISTORY_BEGIN;
    float varianceHistoryEnd = max(MAXENT_VARIANCE_HISTORY_END, varianceHistoryBegin + 1e-3);

    if (lane == 0u) denoiserVariancePoolRequired = 0u;
    barrier();

    uint centerX = localPixel.x + MAXENT_VARIANCE_HALO;
    uint centerY = localPixel.y + MAXENT_VARIANCE_HALO;
    uint centerIndex = centerY * MAXENT_VARIANCE_SHARED_WIDTH + centerX;
    ivec2 centerPixel = ivec2(pixel);
    ivec2 clampedCenter = clamp(centerPixel, ivec2(0), imageMax);
    bool centerInBounds = all(equal(centerPixel, clampedCenter));
    uvec4 centerGeometryWords = readPrimaryGeometryWords(uvec2(clampedCenter));
    DenoiserVarianceGeometry centerGeometry = denoiserVarianceDecodeGeometry(centerGeometryWords, clampedCenter);
    bool centerValid = centerInBounds && centerGeometry.valid;
    float surfaceRejectionScale = denoiserSpatialSurfaceRejectionScale(
            centerGeometry.surfaceDistance, float(imageSize.y));
    float centerPlaneOffset = centerGeometry.surfaceDistance
            * dot(centerGeometry.surfaceNormal, centerGeometry.primaryRay);
    DenoiserVarianceSource centerSource = centerValid
        ? denoiserVarianceLoadSource(clampedCenter) : denoiserVarianceEmptySource();
    centerSource = denoiserVarianceSanitizeSource(centerSource);
    denoiserVarianceWriteTile(centerIndex, centerGeometry, centerSource, centerValid);

    if (centerValid && centerSource.historyLength < varianceHistoryEnd)
        atomicOr(denoiserVariancePoolRequired, 1u);
    barrier();

    // Stable workgroups need only their center samples. A group containing
    // any short-history pixel cooperatively loads the 3-pixel halo once.
    if (denoiserVariancePoolRequired != 0u) {
        ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy
                    * MAXENT_VARIANCE_TILE_SIZE) - ivec2(MAXENT_VARIANCE_HALO);
        for (uint index = lane; index < MAXENT_VARIANCE_SHARED_AREA;
            index += MAXENT_VARIANCE_TILE_SIZE
                    * MAXENT_VARIANCE_TILE_SIZE) {
            uint tileX = index % MAXENT_VARIANCE_SHARED_WIDTH;
            uint tileY = index / MAXENT_VARIANCE_SHARED_WIDTH;
            bool interior = tileX >= MAXENT_VARIANCE_HALO
                    && tileX < MAXENT_VARIANCE_HALO
                            + MAXENT_VARIANCE_TILE_SIZE
                    && tileY >= MAXENT_VARIANCE_HALO
                    && tileY < MAXENT_VARIANCE_HALO
                            + MAXENT_VARIANCE_TILE_SIZE;
            if (!interior)
                denoiserVarianceLoadTile(index, tileOrigin + ivec2(tileX, tileY), imageMax);
        }
        barrier();
    }

    float estimatorVariance = 0.0;
    if (centerValid) {
        vec4 centerMean = denoiserVarianceTileMean(centerIndex);
        vec2 centerMomentHistory = denoiserVarianceTileMomentHistory(centerIndex);
        float centerMeanY2 = centerMomentHistory.x;
        float centerHistory = centerMomentHistory.y;
        float temporalVariance = denoiserVarianceOfTemporalMean(centerMean, centerMeanY2, centerHistory);
        float spatialVariance = temporalVariance;

        if (centerHistory < varianceHistoryEnd
                && denoiserVariancePoolRequired != 0u) {
            float sumMass = 0.0;
            vec4 sumMean = vec4(0.0);
            float sumMeanY2 = 0.0;

            for (int offsetY = -3; offsetY <= 3; ++offsetY) {
                for (int offsetX = -3; offsetX <= 3; ++offsetX) {
                    uint sampleIndex = uint(int(centerY) + offsetY)
                            * MAXENT_VARIANCE_SHARED_WIDTH
                            + uint(int(centerX) + offsetX);
                    if (!denoiserVarianceTileValid(sampleIndex)
                            || denoiserVarianceTileMaterial(sampleIndex)
                                != centerGeometry.materialID)
                        continue;

                    float spatialWeight = MAXENT_VARIANCE_KERNEL_1D[abs(offsetX)]
                            * MAXENT_VARIANCE_KERNEL_1D[abs(offsetY)]
                            * exp(-denoiserVarianceTileSurfacePlaneDepthExponent(sampleIndex, centerPlaneOffset,
                                    centerGeometry.surfaceNormal, surfaceRejectionScale));
                    vec2 sampleMomentHistory = denoiserVarianceTileMomentHistory(sampleIndex);
                    float mass = spatialWeight * sampleMomentHistory.y;

                    sumMass += mass;
                    sumMean += mass
                            * denoiserVarianceTileMean(sampleIndex);
                    sumMeanY2 += mass * sampleMomentHistory.x;
                }
            }

            if (sumMass > 1e-8) {
                float inverseMass = 1.0 / sumMass;
                vec4 pooledMean = denoiserVarianceCanonicalMean(sumMean * inverseMass);
                float pooledMeanY2 = denoiserVarianceCanonicalMeanY2(pooledMean, sumMeanY2 * inverseMass);
                float pooledPopulationVariance = denoiserVarianceBuresPopulation(
                        pooledMean, pooledMeanY2);
                // Preserve the legacy cold-start spatial treatment exactly:
                // the neighborhood estimates a per-observation uncertainty
                // scale, while only the center pixel's actual temporal history
                // reduces the estimator variance.  No Kish/Bessel correction
                // is introduced in this version.
                spatialVariance = pooledPopulationVariance
                        / centerHistory;
            }
        }

        float temporalTrust = smoothstep(varianceHistoryBegin, varianceHistoryEnd, centerHistory);
        if (centerHistory <= 1.5) temporalTrust = 0.0;
        // Both paths are expressed in the same local squared-Bures units; the
        // sigma-domain transition therefore keeps the existing cold-start
        // behavior while changing only the uncertainty geometry.
        float estimatorSigma = mix(sqrt(spatialVariance), sqrt(temporalVariance), temporalTrust);
        estimatorVariance = denoiserVarianceSanitizeNonnegative(estimatorSigma * estimatorSigma);
    }
    if (!centerInBounds) return;
    if (!centerValid) {
        denoiserVarianceStoreInvalid(centerPixel);
        return;
    }

    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = centerSource.maxEntY;
    outputSignal.CoCg = centerSource.CoCg;
    outputSignal.variance = estimatorVariance;
    outputSignal.virtualDistance = centerGeometry.surfaceDistance + centerGeometry.virtualScale * centerSource.hitDistance;
    denoiserVarianceStore(centerPixel, outputSignal, centerGeometryWords, centerGeometry);
}

#endif // MAXENT_VARIANCE_PREPARE_GLSL
