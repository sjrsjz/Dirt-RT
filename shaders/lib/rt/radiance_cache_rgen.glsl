// #version 460 core — declared by ray3.rgen.
#define RADIANCE_CACHE_TRACE
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#include "/lib/rt/raytrace_rgen.glsl"

const float RC_UNIFORM_SPHERE_PDF = 1.0 / (4.0 * PI);
const float RC_TRACE_T_MIN = RADIANCE_CACHE_SURFACE_EPSILON;
const float RC_PROBE_JITTER_SCALE = 0.95;

RadianceCache samplePreviousRadianceCache(vec3 worldPos) {
    vec3 voxelCoord = radianceCacheWorldToVoxel(worldPos, prevRaytracingCamPos);
    if (!isRadianceCacheSampleInBounds(voxelCoord)) return emptyCache();
    return sampleRadianceCacheHist(voxelCoord);
}

GuideInfo computeRadianceCacheGuide(vec3 worldPos) {
    GuideInfo guide;
    guide.axis = vec3(0.0, 1.0, 0.0);
    guide.kappa = 0.0;
    guide.prob = 0.0;
    guide.valid = false;

    RadianceCache previous = samplePreviousRadianceCache(worldPos);
    vec3 directionalEnergy = previous.alice.aliceY.xyz;
    float directionalLength = length(directionalEnergy);
    float totalEnergy = max(previous.alice.aliceY.w, directionalLength);

    if (previous.weight <= 0.0 || directionalLength <= 1e-8 || totalEnergy <= 1e-8) {
        return guide;
    }

    guide.axis = directionalEnergy / directionalLength;
    float rho = clamp(directionalLength / totalEnergy, 0.0, 1.0);
    guide.kappa = alice_kappa(directionalLength, totalEnergy);
    guide.prob = min(PATH_GUIDING_STRENGTH * rho, 0.999);
    guide.valid = true;
    return guide;
}

vec3 sampleProbeJitter(uvec3 voxelCoord, uint frameId) {
    // 与方向采样 RNG 解耦的每帧 3D hash。输出均匀覆盖中心体素的 95%，
    // 每侧保留 2.5% 安全边距，避免 jitter 后重新贴到体素边界。
    vec3 key = vec3(voxelCoord)
            + float(frameId) * vec3(0.754877666, 0.569840296, 0.438579021);
    vec3 xi = hash33(key);
    return (xi - 0.5) * (RC_PROBE_JITTER_SCALE * VOXEL_SIZE);
}

vec3 sampleRadianceCacheDirection(GuideInfo guide, out float estimatorWeight) {
    float mixtureSelector = getRandom();
    vec2 xi = vec2(getRandom(), getRandom());
    bool useGuide = guide.valid && mixtureSelector < guide.prob;

    // 统一使用 ALICE 的球面映射。无引导分支等价于 axis=Y、kappa=0
    // 的均匀球面，避免在 prob=0 时切换到另一套与 RC 随机场相关的参数化。
    vec3 sampleAxis = useGuide ? guide.axis : vec3(0.0, 1.0, 0.0);
    float sampleKappa = useGuide ? guide.kappa : 0.0;
    vec3 direction = sample_alice_guiding(sampleAxis, sampleKappa, xi);

    float guidePdf = guide.valid
        ? alice_guiding_pdf(direction, guide.axis, guide.kappa) : 0.0;
    float mixturePdf = (1.0 - guide.prob) * RC_UNIFORM_SPHERE_PDF
            + guide.prob * guidePdf;
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
    vec3 macroNormal = normalize(faceforward(
                evaluated.macroNormal,
                evaluated.macroNormal,
                rayDirection
            ));
    material surf = materialFromEvaluated(evaluated, blockID);

    // 入射辐射率定义在几何表面的空气侧。沿几何法线偏移 epsilon，
    // 避免浮点误差把表面查询归入实体内部中心 probe。
    vec3 recursiveSamplePos = hitPos + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    RadianceCache recursiveCache = samplePreviousRadianceCache(recursiveSamplePos);
    vec3 recursiveIrradiance = recursiveCache.weight > 0.0
        ? project_alice_irradiance(recursiveCache.alice, macroNormal) : vec3(0.0);
    vec3 diffuseAlbedo = evaluateDiffuseAlbedo(surf, rayDirection, macroNormal);

    vec3 traceLightDir = -lightDir_global;
    vec3 directIncident = vec3(0.0);
    if (dot(geometryNormal, traceLightDir) < 0.0 && !isDarkened) {
        directIncident = evalDirectDiffuseIncident(
                hitPos,
                geometryNormal,
                surf,
                rayDirection,
                traceLightDir,
                false
            );
    }
    vec3 diffuseRadiance = (recursiveIrradiance + directIncident) * diffuseAlbedo / PI;

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
    uvec3 voxelCoord = gl_LaunchIDEXT.xyz;
    if (any(greaterThanEqual(voxelCoord, uvec3(
                    RADIANCE_CACHE_W,
                    RADIANCE_CACHE_H,
                    RADIANCE_CACHE_D
                )))) return;

    setFrame(cam.frameId);
    vec3 seedCoord = vec3(voxelCoord)
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

    vec3 probeCenter = radianceCacheVoxelWorldPos(voxelCoord, cam.viewInverse[3].xyz);
    vec3 probePosition = probeCenter + sampleProbeJitter(voxelCoord, cam.frameId);
    // 引导场属于中心 voxel；jitter 只改变实际射线原点，不改变缓存寻址。
    GuideInfo guide = computeRadianceCacheGuide(probeCenter);
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
    result.alice = radiance_to_alice(radiance, rayDirection);
    result.weight = 1.0;
    storeRadianceCacheSwap(voxelCoord, result);
}
