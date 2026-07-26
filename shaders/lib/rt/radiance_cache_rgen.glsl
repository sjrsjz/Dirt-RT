// #version 460 core — declared by ray3.rgen.
#define RADIANCE_CACHE_TRACE
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#include "/lib/rt/raytrace_rgen.glsl"

const float RC_UNIFORM_SPHERE_PDF = 1.0 / (4.0 * PI);
const float RC_TRACE_T_MIN = 0.001;

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
    guide.valid = guide.prob > 0.0;
    return guide;
}

vec3 sampleUniformSphere(vec2 xi) {
    float y = 1.0 - 2.0 * xi.x;
    float radius = sqrt(max(0.0, 1.0 - y * y));
    float phi = 2.0 * PI * xi.y;
    return vec3(radius * cos(phi), y, radius * sin(phi));
}

vec3 sampleRadianceCacheDirection(GuideInfo guide, out float estimatorWeight) {
    bool useGuide = guide.valid && getRandom() < guide.prob;
    vec2 xi = vec2(getRandom(), getRandom());
    vec3 direction = useGuide
        ? sample_alice_guiding(guide.axis, guide.kappa, xi)
        : sampleUniformSphere(xi);

    float guidePdf = guide.valid
        ? alice_guiding_pdf(direction, guide.axis, guide.kappa)
        : 0.0;
    float mixturePdf = (1.0 - guide.prob) * RC_UNIFORM_SPHERE_PDF
        + guide.prob * guidePdf;
    estimatorWeight = RC_UNIFORM_SPHERE_PDF / max(mixturePdf, 1e-20);
    return normalize(direction);
}

vec3 evaluateRadianceCacheHit(vec3 rayOrigin, vec3 rayDirection, vec3 hitPos, float hitDistance) {
    Material evaluated = evaluateMaterial(tmp_Payload, rayDirection, 1u);
    vec3 payloadNormal = payload_unpackGeomNormal(tmp_Payload.data);
    vec3 geometryNormal = faceforward(payloadNormal, payloadNormal, rayDirection);
    vec3 macroNormal = normalize(faceforward(
        evaluated.macroNormal,
        evaluated.macroNormal,
        rayDirection
    ));

    int blockID;
    payload_unpackShadow(tmp_Payload.data, blockID);
    material surf = materialFromEvaluated(evaluated, blockID);

    RadianceCache recursiveCache = samplePreviousRadianceCache(hitPos);
    vec3 recursiveIrradiance = recursiveCache.weight > 0.0
        ? project_alice_irradiance(recursiveCache.alice, macroNormal)
        : vec3(0.0);
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
    vec3 diffuseRadiance = (recursiveIrradiance + directIncident) * diffuseAlbedo;

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

    vec3 probePosition = radianceCacheVoxelWorldPos(voxelCoord, cam.viewInverse[3].xyz);
    GuideInfo guide = computeRadianceCacheGuide(probePosition);
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
        ? sampleSkyNoSun(probePosition.y, rayDirection, -lightDir_global)
        : evaluateRadianceCacheHit(probePosition, rayDirection, hitPos, hitDistance);

    radiance *= estimatorWeight;
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);
    radiance = clamp(radiance, vec3(0.0), vec3(32000.0));

    RadianceCache result;
    result.alice = radiance_to_alice(radiance, rayDirection);
    result.weight = 1.0;
    storeRadianceCacheSwap(voxelCoord, result);
}
