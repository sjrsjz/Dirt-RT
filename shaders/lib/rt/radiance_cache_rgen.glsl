// #version 460 core — declared by ray5.rgen.
#define RADIANCE_CACHE_TRACE
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#include "/lib/rt/raytrace_rgen.glsl"

const float RC_UNIFORM_SPHERE_PDF = 1.0 / (4.0 * PI);
const float RC_TRACE_T_MIN = RADIANCE_CACHE_SURFACE_EPSILON;
const float RC_PROBE_JITTER_SCALE = 0.95;
const float RC_MAX_GUIDED_PROBABILITY = 0.9;

struct RadianceCacheGuideInfo {
    GuideInfo alice;
    vec3 risAxis;
    float risKappa;
    float risProb;
    bool risValid;
};

RadianceCache samplePreviousRadianceCache(vec3 worldPos) {
    vec3 voxelCoord = radianceCacheWorldToVoxel(worldPos, prevRaytracingCamPos);
    if (!isRadianceCacheSampleInBounds(voxelCoord)) return emptyCache();
    RadianceCacheAddress address = findRadianceCacheAddress(worldPos);
    if (!radianceCacheAddressHasHistory(address, cam.frameId)) return emptyCache();
    return loadRadianceCachePlanes(address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
}

RadianceCacheGuideInfo computeRadianceCacheGuide(vec3 worldPos) {
    RadianceCacheGuideInfo guide;
    guide.alice.axis = vec3(0.0, 1.0, 0.0);
    guide.alice.kappa = 0.0;
    guide.alice.prob = 0.0;
    guide.alice.valid = false;
    guide.risAxis = vec3(0.0, 1.0, 0.0);
    guide.risKappa = 0.0;
    guide.risProb = 0.0;
    guide.risValid = false;

    vec3 voxelCoord = radianceCacheWorldToVoxel(worldPos, prevRaytracingCamPos);
    if (!isRadianceCacheSampleInBounds(voxelCoord)) return guide;
    RadianceCacheAddress address = findRadianceCacheAddress(worldPos);
    if (!radianceCacheAddressHasHistory(address, cam.frameId)) return guide;

    RadianceCache previous = loadRadianceCachePlanes(
        address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
    if (!radianceCacheValueValid(previous)) return guide;
    vec4 luminanceAlice = rgb_alice_luminance(previous.alice) * previous.W;
    vec3 directionalEnergy = luminanceAlice.xyz;
    float directionalLength = length(directionalEnergy);
    float totalEnergy = max(luminanceAlice.w, directionalLength);

    if (directionalLength <= 1e-8 || totalEnergy <= 1e-8) {
        return guide;
    }

    guide.alice.axis = directionalEnergy / directionalLength;
    float rho = clamp(directionalLength / totalEnergy, 0.0, 1.0);
    guide.alice.kappa = alice_kappa(directionalLength, totalEnergy);
    guide.alice.prob = min(PATH_GUIDING_STRENGTH * rho,
        RC_MAX_GUIDED_PROBABILITY);
    guide.alice.valid = guide.alice.prob > 1e-6;
    if (RADIANCE_CACHE_RIS_GUIDING_STRENGTH <= 0.0) return guide;

    // A temporal RIS reservoir is a proposal, never the light-field estimate.
    // Reinforce it only when its selected incoming direction and normalized
    // energy agree with the stable multi-frame ALICE moments.
    RadianceCache reservoir = loadRadianceCachePlanes(
        address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
    if (!radianceCacheValueValid(reservoir)) return guide;
    vec4 risAlice = rgb_alice_luminance(reservoir.alice);
    float risDirectionalLength = length(risAlice.xyz);
    float risMeanEnergy = max(risAlice.w, risDirectionalLength) * reservoir.W;
    if (!(risDirectionalLength > 1e-8) || !(risMeanEnergy > 1e-8)) return guide;

    vec3 risAxis = risAlice.xyz / risDirectionalLength;
    float directionalAgreement = smoothstep(0.25, 0.9,
        max(dot(guide.alice.axis, risAxis), 0.0));
    float energyAgreement = min(totalEnergy, risMeanEnergy)
        / max(max(totalEnergy, risMeanEnergy), 1e-8);
    energyAgreement = smoothstep(0.05, 0.5, energyAgreement);
    float confidenceDenominator = max(
        min(float(RADIANCE_CACHE_MAX_HIST), 16.0) - 1.0, 1.0);
    float historyConfidence = clamp((reservoir.M - 1.0)
        / confidenceDenominator, 0.0, 1.0);
    float risShare = clamp(RADIANCE_CACHE_RIS_GUIDING_STRENGTH, 0.0, 1.0)
        * directionalAgreement * energyAgreement * historyConfidence;

    guide.risProb = guide.alice.prob * risShare;
    guide.alice.prob -= guide.risProb;
    guide.risAxis = risAxis;
    guide.risKappa = clamp(RADIANCE_CACHE_RIS_GUIDING_KAPPA, 0.0, 0.98);
    guide.risValid = guide.risProb > 1e-6 && guide.risKappa > 1e-4;
    return guide;
}

vec3 sampleProbeJitter(ivec3 voxelCoord, uint frameId) {
    // 与方向采样 RNG 解耦的每帧 3D hash。输出均匀覆盖中心体素，
    // 每侧保留 RC_PROBE_JITTER_SCALE 安全边距，避免 jitter 后重新贴到体素边界。
    vec3 key = vec3(voxelCoord)
            + float(frameId) * vec3(0.754877666, 0.569840296, 0.438579021);
    vec3 xi = hash33(key);
    return (xi - 0.5) * (RC_PROBE_JITTER_SCALE * VOXEL_SIZE);
}

vec3 sampleRadianceCacheDirection(RadianceCacheGuideInfo guide,
        out float estimatorWeight) {
    float mixtureSelector = getRandom();
    vec2 xi = vec2(getRandom(), getRandom());
    float aliceProb = guide.alice.valid ? max(guide.alice.prob, 0.0) : 0.0;
    float risProb = guide.risValid ? max(guide.risProb, 0.0) : 0.0;
    float uniformProb = max(1.0 - aliceProb - risProb, 0.0);

    bool useRis = mixtureSelector < risProb;
    bool useAlice = !useRis && mixtureSelector < risProb + aliceProb;
    vec3 sampleAxis = useRis ? guide.risAxis
        : (useAlice ? guide.alice.axis : vec3(0.0, 1.0, 0.0));
    float sampleKappa = useRis ? guide.risKappa
        : (useAlice ? guide.alice.kappa : 0.0);
    vec3 direction = sample_alice_guiding(sampleAxis, sampleKappa, xi);

    float alicePdf = guide.alice.valid
        ? alice_guiding_pdf(direction, guide.alice.axis, guide.alice.kappa) : 0.0;
    float risPdf = guide.risValid
        ? alice_guiding_pdf(direction, guide.risAxis, guide.risKappa) : 0.0;
    float mixturePdf = uniformProb * RC_UNIFORM_SPHERE_PDF
        + aliceProb * alicePdf + risProb * risPdf;
    estimatorWeight = RC_UNIFORM_SPHERE_PDF / max(mixturePdf, 1e-20);
    return normalize(direction);
}

vec3 evaluateRadianceCacheHit(vec3 rayOrigin, vec3 rayDirection, vec3 hitPos, float hitDistance) {
    vec3 payloadNormal = payload_unpackGeomNormal(tmp_Payload.data);
    int blockID;
    payload_unpackShadow(tmp_Payload.data, blockID);

    // 体素中心探针仍可能位于实心方块内部。从内部命中的不透明背面
    // 是有效的完全遮挡样本，不能翻转法线后当作正常受光表面评估。
    if (dot(payloadNormal, rayDirection) > 0.0 && !isTransmissiveBlock(blockID)) {
        return vec3(0.0);
    }

    Material evaluated = evaluateMaterial(tmp_Payload, rayDirection, 1u);
    vec3 geometryNormal = faceforward(payloadNormal, payloadNormal, rayDirection);
    vec3 macroNormal = evaluated.macroNormal;
    material surf = materialFromEvaluated(evaluated, blockID);

    // 入射辐射率定义在几何表面的空气侧。沿几何法线偏移 epsilon，
    // 避免浮点误差把表面查询归入实体内部中心 probe。
    vec3 recursiveSamplePos = hitPos + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    RadianceCache recursiveCache = samplePreviousRadianceCache(recursiveSamplePos);
    vec3 recursiveDiffuseIncident = radianceCacheValueValid(recursiveCache)
        ? radianceCacheDiffuseIncident(recursiveCache, macroNormal) : vec3(0.0);
    vec3 diffuseAlbedo = evaluateDiffuseAlbedo(surf, rayDirection, macroNormal);

    vec3 traceLightDir = -lightDir_global;
    vec3 directIncident = vec3(0.0);
    if (dot(geometryNormal, traceLightDir) < 0.0 && !isDarkened) {
        directIncident = evalDirectDiffuseIncident(
                hitPos,
                geometryNormal,
                macroNormal,
                rayDirection,
                traceLightDir,
                false
            );
    }
    // Both terms are E/pi; apply the diffuse albedo exactly once.
    vec3 diffuseRadiance = (recursiveDiffuseIncident + directIncident) * diffuseAlbedo;

    MediumResult medium = evalMedium(
            hitDistance,
            rayDirection,
            rayOrigin.y,
            false,
            vec4(0.0),
            vec3(0.0)
        );

    // 对齐现有首段合成语义：介质透射作用于表面项，段内介质发射单独加入。
    // 更高阶间接光由上一帧缓存的固定点反馈提供。
    return medium.absorption * (surf.light + diffuseRadiance) + medium.emission;
}

void main() {
    vec3 currentCameraPosition = cam.viewInverse[3].xyz;
    uint linearIndex = gl_LaunchIDEXT.x
        + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
        + gl_LaunchIDEXT.z * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y;
    ivec3 worldVoxel;
    RadianceCacheAddress poolAddress = radianceCacheAddressForPoolVoxel(
        linearIndex, currentCameraPosition, worldVoxel);
    if (poolAddress.token >= RC_LOCKED_TOKEN) return;
    uint frameStamp = cam.frameId;
    // The allocator pins every resident/new brick requested by ray0/ray1 in
    // this frame. Retain off-screen cache data, but do not spend rays updating
    // stale slots until visible geometry requests them again.
    if (rcLoad(rcMetaAddress(poolAddress.slot, RC_META_PIN_FRAME)) !=
            rcFrameTag(frameStamp)) return;
    bool hasHistory = radianceCacheResolvedPoolAddressHasHistory(
        poolAddress, frameStamp);
    if (hasHistory && !radianceCacheShouldUpdate(worldVoxel, frameStamp)) return;

    setFrame(cam.frameId);
    vec3 seedCoord = vec3(worldVoxel)
            + float(cam.frameId) * vec3(0.61803398875, 0.41421356237, 0.73205080757);
    wseed = floatBitsToUint(hash13(seedCoord));
    wseed3 = uvec3(
            floatBitsToUint(randcore4()),
            floatBitsToUint(randcore4()),
            floatBitsToUint(randcore4())
        );

    setSkyVars();
    #if END_SKYBOX == 1
    isDarkened = world_type_global != WORLD_OVERWORLD && world_type_global != WORLD_THE_NETHER;
    #else
    isDarkened = world_type_global != WORLD_OVERWORLD
            && world_type_global != WORLD_THE_END
            && world_type_global != WORLD_THE_NETHER;
    #endif

    vec3 probeCenter = (vec3(worldVoxel) + 0.5) * VOXEL_SIZE;
    vec3 probePosition = probeCenter + sampleProbeJitter(worldVoxel, cam.frameId);
    // 引导场属于中心 voxel；jitter 只改变实际射线原点，不改变缓存寻址。
    RadianceCacheGuideInfo guide = computeRadianceCacheGuide(probeCenter);
    float estimatorWeight;
    vec3 rayDirection = sampleRadianceCacheDirection(guide, estimatorWeight);

    vec3 hitPos, hitDirection;
    float hitDistance = raycastMin(
            probePosition,
            rayDirection,
            hitPos,
            hitDirection,
            true,
            false,
            RC_TRACE_T_MIN
        );

    vec3 radiance = hitDistance < -0.5
        ? sampleSkyNoSun(probePosition.y, rayDirection, -lightDir_global) : evaluateRadianceCacheHit(probePosition, rayDirection, hitPos, hitDistance);

    radiance *= estimatorWeight;
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);
    radiance = clamp(radiance, vec3(0.0), vec3(32000.0));

    RadianceCache result;
    result.alice = radiance_to_rgb_alice(radiance, rayDirection);
    result.W = 1.0;
    result.M = 1.0;
    storeRadianceCachePlanes(
        poolAddress, RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1, result);
}
