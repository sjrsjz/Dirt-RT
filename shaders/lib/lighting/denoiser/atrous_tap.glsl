// Shared loop body, included inside the eight-tap spatial loop after bounds
// rejection and the kernelWeight declaration. Uses the center guide prepared
// by the schedule; updates both accumulators with the same proposal weights.
// Keep rejection as loop continue: a function with early returns changes the
// driver's control-flow lowering and regresses the large kernel on NVIDIA.
// Kernel-specific expression ordering preserves the existing FP32 rounding.

uvec4 sampleGeometryWords = denoiserSpatialLoadGeometryWords(samplePixel);
uvec4 sampleSignalWords = denoiserSpatialLoadSignalWords(samplePixel);
uvec4 sampleCurrentWords =
    denoiserSpatialLoadIndependentCurrentWords(samplePixel);
if (!denoiserSpatialGeometryWordsValid(sampleGeometryWords)
        || !denoiserSpatialPreparedSignalWordsValid(sampleSignalWords)
        || !denoiserSpatialPreparedSignalWordsValid(sampleCurrentWords))
    continue;

vec3 samplePrimaryRay;
float sampleSurfaceDistance;
float sampleEffectiveSamples;
denoiserSpatialDecodeSampleGeometry(sampleGeometryWords, samplePixel,
    samplePrimaryRay, sampleSurfaceDistance,
    sampleEffectiveSamples);
if (!statisticsValidEffectiveSampleCount(sampleEffectiveSamples))
    continue;
#ifdef MAXENT_ATROUS_SMALL_KERNEL
vec3 sampleSurfacePosition = samplePrimaryRay
        * sampleSurfaceDistance;
float surfaceGeometryExponent = surfaceRejectionScale
        * abs(dot(centerGeometry.geometryNormal, sampleSurfacePosition)
                - centerGeometry.surfacePlaneOffset);
#else
float surfaceGeometryExponent =
    denoiserSpatialAxialDistanceExponent(
        centerGeometry.surfacePlaneOffset,
        centerGeometry.geometryNormal, samplePrimaryRay,
        sampleSurfaceDistance, surfaceRejectionScale);
#endif

DenoiserMaxEntSignal sampleSignal = denoiserUnpackMaxEntSignalTrusted(sampleSignalWords);
DenoiserMaxEntSignal sampleCurrent =
    denoiserUnpackMaxEntSignalTrusted(sampleCurrentWords);
float virtualDistanceWeight;
float weight = denoiserSpatialWeight(centerMetric, centerSignal,
        sampleSignal, samplePrimaryRay,
        surfaceGeometryExponent,
        kernelWeight,
        lightDifferenceScale,
        virtualDistanceAlpha, centerVirtualPosition,
        centerVirtualNormal, virtualRejectionScale,
        virtualDistanceWeight);
#if DENOISER_SPATIAL_ACCUMULATE_PROPOSAL
denoiserSpatialAccumulate(accum, sampleSignal, weight,
    virtualDistanceWeight);
#endif
denoiserSpatialAccumulate(currentAccum, sampleCurrent, weight,
    virtualDistanceWeight);
