// #version 460 core is declared by each enclosing rayN.rgen entry. ray0
// defines PRIMARY_GBUFFER_PASS; ray1..ray3 define one dedicated first lobe.
// This file is never compiled alone.
#extension GL_EXT_ray_query : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_scalar_block_layout : require

#define PREV_DIFFUSE_BUFFER

#include "/lib/rt/payload.glsl"
#define FRAGMENT_INFO_NO_PRIMITIVE
#include "/lib/rt/fragment_info.glsl"
#include "/lib/common/bicubic.glsl"
// constants.glsl includes settings.glsl — must precede pom.glsl for option overrides
#include "/lib/constants.glsl"
#include "/lib/settings.glsl"
#include "/lib/rt/pom.glsl"
#include "/lib/rt/mipmap.glsl"
#include "/lib/sky.glsl"
#include "/lib/math/quaternions.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/radiance_cache.glsl"
#include "/lib/pbr/material.glsl"
#include "/lib/common.glsl"

// ray0 builds the primary-surface cache. Continuation entries define exactly
// one of FIRST_LOBE_DIFFUSE / FIRST_LOBE_REFLECTION / FIRST_LOBE_REFRACTION.
#if !defined(PRIMARY_GBUFFER_PASS) && !defined(FIRST_LOBE_DIFFUSE) && !defined(FIRST_LOBE_REFLECTION) && !defined(FIRST_LOBE_REFRACTION)
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#endif

// ---------------------------------------------------------------------------
// PSR (Primary Surface Replacement) — refraction virtual-image reprojection
// ---------------------------------------------------------------------------
const float PSR_ROUGHNESS_THRESHOLD = 0.15; // first-surface roughness above this → disable PSR, fall back to first-surface temporal accumulation
const float PATH_ROUGHNESS_TERMINATE = 0.8; // accumulated sqrt(Σ r_i²) above this → terminate refractive chain early (too diffuse)
const int MAX_REFRACTIVE_BOUNCES = 4; // max refractive surfaces to trace through before stopping

layout(std430, binding = 0) uniform CameraInfo {
    vec3 corners[4];
    mat4 viewInverse;
    uint frameId;
    uint flags;
    uint world_type;
} cam;

layout(binding = 1) uniform accelerationStructureEXT acc;
layout(binding = 3) uniform sampler2D blockTex;
layout(binding = 4) uniform sampler2D blockTexNormal;
layout(binding = 5) uniform sampler2D blockTexSpecular;
layout(binding = 6) uniform sampler2D entityTextures[256];
layout(location = 6) rayPayloadEXT Payload payload;

#include "/lib/rt/blue_noise.glsl"

#define ENTITY_INSTANCE_FLAG 0x800000u

struct EntityMotionVertex {
    f16vec4 deltaAndValid;
};

layout(std430, set = 0, binding = 2, scalar) readonly buffer EntityMotionBuffer {
    EntityMotionVertex vertices[];
} entityMotionBuffer;

#if !defined(RADIANCE_CACHE_TRACE)
void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir);
void TracePrimaryGBuffer(uvec2 coord, vec3 ro, vec3 rd);
#endif

bool isDarkened = false;

#if defined(PRIMARY_GBUFFER_PASS) || defined(FIRST_LOBE_DIFFUSE)
void markRadianceCacheGeometryHit(uvec2 pixel, vec3 hitPosition, vec3 geometryNormal) {
    uvec2 tileMask = uvec2(RADIANCE_CACHE_MARK_TILE_SIZE - 1u);
    if (any(notEqual(pixel & tileMask, uvec2(0u)))) return;
    vec3 cameraPosition = cam.viewInverse[3].xyz;
    vec3 cacheCoord = radianceCacheWorldToVoxel(hitPosition, cameraPosition);
    if (!isRadianceCacheSampleInBounds(cacheCoord)) return;
    vec3 solidPosition = hitPosition - geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    vec3 airPosition = hitPosition + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    ivec3 solidBrick = radianceCacheWorldBrick(radianceCacheWorldVoxel(solidPosition));
    ivec3 airBrick = radianceCacheWorldBrick(radianceCacheWorldVoxel(airPosition));
    markRadianceCacheBlockHasGeometry(solidBrick, cameraPosition);
    if (any(notEqual(airBrick, solidBrick)))
        markRadianceCacheBlockHasGeometry(airBrick, cameraPosition);
}
#endif

#if !defined(RADIANCE_CACHE_TRACE)

#if !defined(PRIMARY_GBUFFER_PASS)
// Continuation passes only need the primary-distance word to reject sky.
// Keep this before camera-ray reconstruction, RNG setup, setSkyVars(), and
// material decoding: all of those are dead work when ray0 reported a miss.
bool clearSkyContinuation(uvec2 pixel) {
    float primaryDistance = uintBitsToFloat(
            geomBuffer.data[addr(GEO_N_GEO, pixel)].w);
    if (primaryDistance >= -0.5) return false;

    #if defined(FIRST_LOBE_DIFFUSE)
    diffuseBuffer.data[addr(DIF_N_LIGHT, pixel)] = uvec4(0u);
    diffuseBuffer.data[addr(DIF_N_GEO, pixel)] = uvec4(0u);
    #elif defined(FIRST_LOBE_REFLECTION)
    reflectBuffer.data[addr(SPEC_N_GEO, pixel)] = uvec4(0u);
    reflectBuffer.data[addr(SPEC_N_LIGHT, pixel)] = uvec4(0u);
    #else
    refractBuffer.data[addr(SPEC_N_GEO, pixel)] = uvec4(0u);
    refractBuffer.data[addr(SPEC_N_LIGHT, pixel)] = uvec4(0u);
    geomBuffer.data[addr(GEO_N_NORMALS, pixel)].w = 0u;
    #endif
    return true;
}
#endif

void main() {
    uvec2 pixel = uvec2(gl_LaunchIDEXT.xy);
    #if !defined(PRIMARY_GBUFFER_PASS)
    if (clearSkyContinuation(pixel)) return;
    #endif

    vec2 px = vec2(gl_LaunchIDEXT.xy);
    vec2 p = px / vec2(gl_LaunchSizeEXT.xy);

    vec3 origin = cam.viewInverse[3].xyz;
    vec3 target = mix(mix(cam.corners[0], cam.corners[2], p.y), mix(cam.corners[1], cam.corners[3], p.y), p.x);
    vec3 direction = normalize((cam.viewInverse * vec4(target.xyz, 0.0)).xyz);

    setFrame(cam.frameId);
    #if END_SKYBOX == 1
    isDarkened = world_type_global != WORLD_OVERWORLD && world_type_global != WORLD_THE_NETHER;
    #else
    isDarkened = world_type_global != WORLD_OVERWORLD && world_type_global != WORLD_THE_END && world_type_global != WORLD_THE_NETHER;
    #endif

    wseed = floatBitsToUint(hash12(px + fract(cam.frameId * vec2(0.6180339887498949, 0.4142135623730950))));

    vec3 seed0 = vec3(randcore4(), randcore4(), randcore4());
    wseed3.x = floatBitsToUint(seed0.x);
    wseed3.y = floatBitsToUint(seed0.y);
    wseed3.z = floatBitsToUint(seed0.z);

    setSkyVars();
    #if defined(PRIMARY_GBUFFER_PASS)
    TracePrimaryGBuffer(pixel, origin, direction);
    #else
    Trace(pixel, origin, direction, -lightDir_global);
    #endif

    // Per-frame-once global state: only primary pass pixel (0,0). Updating the
    // previous matrices from any continuation pass would break reprojection.
    #if defined(PRIMARY_GBUFFER_PASS)
    if (gl_LaunchIDEXT.xy == vec2(0)) {
        world_type_global = int(cam.world_type);
        frame_id = int(cam.frameId);
        camPos = origin;
        camY_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[2]), 0)).xyz;
        camX_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[1]), 0)).xyz;

        // Save the already-composed transform for next-frame reprojection.
        rtPrevViewProjection = rtViewProjection;

        // ModelView: pure rotation (transpose of viewInverse), no translation
        rtModelView = mat4(transpose(mat3(cam.viewInverse)));

        // Projection: asymmetric frustum (with TAA jitter offset)
        float zNear = -cam.corners[0].z;
        float w = cam.corners[1].x - cam.corners[0].x;
        float h = cam.corners[2].y - cam.corners[0].y;
        float farD = 2048.0;
        mat4 projection = mat4(0.0);
        projection[0][0] = (2.0 * zNear) / w;
        projection[1][1] = (2.0 * zNear) / h;
        projection[2][0] = (cam.corners[1].x + cam.corners[0].x) / w;
        projection[2][1] = (cam.corners[2].y + cam.corners[0].y) / h;
        projection[2][2] = -(farD + zNear) / (farD - zNear);
        projection[2][3] = -1.0;
        projection[3][2] = -(2.0 * farD * zNear) / (farD - zNear);
        rtViewProjection = projection * rtModelView;
        rtInverseViewProjection = inverse(rtViewProjection);
    }
    #endif
}
#endif

Payload tmp_Payload;

float raycastMin(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o,
    bool inverse_0, bool isNEE, float tMin) {
    bool inside = !inverse_0;
    payload_packFlags(payload.data, 0.0, inside, false, isNEE);
    payload_packShadow(payload.data, vec3(1.0), 0);
    float tMax = 2048.0;
    traceRayEXT(acc, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, ro, tMin, rd, tMax, 6);
    Payload hitPayload = payload;
    float t;
    ro_o = payload_unpackHitPos(hitPayload.data, t);
    rd_o = rd;
    tmp_Payload = hitPayload;
    return t;
}

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0, bool isNEE) {
    return raycastMin(ro, rd, ro_o, rd_o, inverse_0, isNEE, 0.0);
}

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0) {
    return raycast(ro, rd, ro_o, rd_o, inverse_0, false);
}

vec4 getPrimarySurfaceMotion(Payload hitPayload) {
    uint instanceIdx, geometryId, primitiveId;
    payload_unpackQuadIDs(hitPayload.data, instanceIdx, geometryId, primitiveId);
    if ((instanceIdx & ENTITY_INSTANCE_FLAG) == 0u) {
        // w=2 distinguishes static scene geometry from a tracked entity in
        // motion debug views. Temporal passes only test w >= 0.5.
        return vec4(0.0, 0.0, 0.0, 2.0);
    }

    vec2 bary = payload_unpackBarycentrics(hitPayload.data);
    float w0 = 1.0 - bary.x - bary.y;
    uint baseVertex = (primitiveId >> 1u) * 4u;
    uint secondVertex = (primitiveId & 1u) == 0u ? 1u : 2u;
    uint thirdVertex = (primitiveId & 1u) == 0u ? 2u : 3u;
    vec4 m0 = vec4(entityMotionBuffer.vertices[baseVertex].deltaAndValid);
    vec4 m1 = vec4(entityMotionBuffer.vertices[baseVertex + secondVertex].deltaAndValid);
    vec4 m2 = vec4(entityMotionBuffer.vertices[baseVertex + thirdVertex].deltaAndValid);
    vec3 motion = m0.xyz * w0 + m1.xyz * bary.x + m2.xyz * bary.y;
    float valid = min(m0.w, min(m1.w, m2.w));
    return vec4(motion, valid);
}

struct material {
    vec3 Cs;
    vec3 Cd;
    vec2 S;
    vec4 R;
    vec3 light;
};

material newMaterial(vec3 Cs, vec3 Cd, vec2 S, vec4 R, vec3 light) {
    material a;
    a.Cs = Cs;
    a.Cd = Cd;
    a.S = S;
    a.R = R;
    a.light = light;
    return a;
}

Material evaluateMaterial(Payload pld, vec3 rd_i, uint bounce) {
    // --- Unpack quad data from payload (packed by rchit, no geometryBuffers needed) ---
    vec2 uv = payload_unpackQuadUV(pld.data);
    vec4 atlas = payload_unpackAtlasBox(pld.data);
    vec3 geomN = payload_unpackGeomNormal(pld.data);
    vec3 tangent = payload_unpackTangent(pld.data);
    vec3 tint;
    float skylight;
    payload_unpackQuadExtras(pld.data, tint, skylight);
    int blockID;
    // vec3 _shadow = payload_unpackShadow(pld.data, blockID);
    bool _inside, handedness, _isNEE;
    payload_unpackFlags(pld.data, _inside, handedness, _isNEE);
    float bitangentSign = handedness ? 1.0 : -1.0;
    uint instanceIdx, entityTextureId, primitiveId;
    payload_unpackQuadIDs(pld.data, instanceIdx, entityTextureId, primitiveId);

    // --- TBN ---
    // Interpolated/compressed tangents are not guaranteed to remain
    // perpendicular to the geometric normal.  Gram-Schmidt here prevents the
    // tangent-space XY terms from pushing a normal map below the surface.
    tangent = normalize(tangent - geomN * dot(geomN, tangent));
    // tangent and geomN are now orthonormal, hence their cross is unit length.
    vec3 bitangent = cross(tangent, geomN) * bitangentSign;
    mat3 tbn = mat3(tangent, bitangent, geomN);

    float hitDistance;
    payload_unpackHitPos(pld.data, hitDistance);
    vec2 mipResolution = max(vec2(resolution_global),
            vec2(gl_LaunchSizeEXT.xy));
    float pixelConeSpread = rtPixelConeSpread(cam.corners[0],
            cam.corners[1], cam.corners[2], mipResolution);

    vec2 sampleUV = uv;
    vec4 albedoTex;
    vec4 specularTex;
    vec4 normalTex;

    if (entityTextureId != 0u) {
        uint textureIndex = entityTextureId - 1u;
        ivec2 textureResolution = textureSize(
                entityTextures[nonuniformEXT(textureIndex)], 0);
        float mipLevel = rtTextureLod(textureResolution,
                vec4(0.0, 0.0, 1.0, 1.0), hitDistance, rd_i, geomN,
                bounce, pixelConeSpread);
        albedoTex = textureLod(
                entityTextures[nonuniformEXT(textureIndex)], uv, mipLevel);
        specularTex = vec4(0.0, 0.04, 0.0, 1.0);
        normalTex = vec4(0.5, 0.5, 1.0, 1.0);
    } else {
        vec2 localCoord = getRelativeUV(uv, atlas);

        // --- POM — first hit only ---
        ivec2 textureResolution = textureSize(blockTex, 0);
        float mipLevel = rtTextureLod(textureResolution, atlas,
                hitDistance, rd_i, geomN, bounce, pixelConeSpread);

        #if POM_ENABLED == 1
        if (bounce == 0u) {
            sampleUV = computeParallaxUV(blockTexNormal, localCoord, atlas,
                    rd_i, tbn, mipLevel);
        } else {
            sampleUV = uv;
        }
        #else
        sampleUV = uv;
        #endif

        // Explicit LOD is mandatory in RT: implicit texture() derivatives are not
        // available in ray stages and otherwise collapse to mip 0.
        albedoTex = textureLod(blockTex, sampleUV, mipLevel);
        specularTex = textureLod(blockTexSpecular, sampleUV, mipLevel);

        #if POM_ENABLED == 1
        if (bounce == 0u && mipLevel < 0.5) {
            normalTex = textureBicubic(blockTexNormal, sampleUV, atlas,
                    vec2(textureResolution));
        } else {
            normalTex = textureLod(blockTexNormal, sampleUV, mipLevel);
        }
        #else
        normalTex = textureLod(blockTexNormal, sampleUV, mipLevel);
        #endif
    }

    albedoTex.rgb = pow(albedoTex.rgb * tint, vec3(2.2));

    Material evaluated = getMaterial(albedoTex, normalTex, specularTex, tbn,
        wetStrength_global, wetness_global, skylight, geomN);
    vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
    evaluated.macroNormal = constrainMappedNormal(evaluated.macroNormal,
        geometryNormal, -rd_i);
    return evaluated;
}

// Check if a block is a transmissive/refractive surface (water, glass).
// LabPBR standard: translucent blocks are identified by the rendering layer,
// not by any material channel. In our ray-tracing pipeline we conservatively
// treat only water and glass as transmissive for the PSR chain.
bool isTransmissiveBlock(int blockID) {
    return blockID == BLOCK_WATER || blockID == BLOCK_GLASS;
}

// Convert Material (from getMaterial) to BSDF material struct
material materialFromEvaluated(Material mat, int blockID) {
    // Precompute block-type flags once (each was compared 3-5× before)
    bool isWater = blockID == BLOCK_WATER;
    bool isGlass = blockID == BLOCK_GLASS;
    bool isPortal = blockID == BLOCK_PORTAL;

    float metallic = mat.metallic;

    // albedo.a is coverage for alpha-tested geometry. The any-hit shader has
    // already rejected uncovered texels, so using the remaining filtered alpha
    // as physical transmission turns leaf/vine/lily-pad edges into glass.
    // Only explicitly classified blocks may enter the transmission branch.
    float opaqueFraction = (isWater || isGlass) ? 0.0 : 1.0;
    opaqueFraction = isPortal ? 0.25 : opaqueFraction;
    float roughness = (isWater || isPortal) ? 0.0 : mat.roughness;
    vec3 albedo = isWater ? vec3(1.0) : mat.albedo;
    vec3 emission = isPortal ? albedo * (1.0 - opaqueFraction) : mat.emission;
    float specSelector = (isWater || isGlass)
        ? 1.0 : mix(opaqueFraction, 1.0, metallic);

    return newMaterial(clamp(mat.F0, 0.0, 1.0), albedo,
        vec2(specSelector, 1.0 - opaqueFraction),
        vec4(roughness > 0.01 ? max(roughness, 0.0125) : 0.0, opaqueFraction,
            isWater, mat.subsurface_scattering),
        emission);
}

// The compact shadow payload reserves its block-ID byte for three special
// transport classes, so ordinary blocks arrive as ID 0. ReLAX still needs a
// stable discriminator. The atlas rectangle identifies the sampled sprite
// without another payload slot, SSBO, or image.
uint hashRelaxMaterialWord(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

int getRelaxMaterialID(Payload pld, int transportBlockID) {
    vec4 atlas = payload_unpackAtlasBox(pld.data);
    uvec4 a = floatBitsToUint(atlas);
    uint h = hashRelaxMaterialWord(a.x ^ (a.y * 0x9e3779b9u));
    h = hashRelaxMaterialWord(h ^ a.z ^ (a.w * 0x85ebca6bu));

    uint transportClass = transportBlockID == BLOCK_WATER ? 1u : (transportBlockID == BLOCK_GLASS ? 2u : (transportBlockID == BLOCK_PORTAL ? 3u : 0u));
    h = hashRelaxMaterialWord(h ^ (transportClass * 0x27d4eb2du));
    return int(h & 0xffffu);
}

vec3 evaluateNonSpecularAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    // Disney's diffuse/base-color lobe is not pre-multiplied by a view-angle
    // Fresnel term. Its grazing response is part of the Burley diffuse factor.
    return surf.Cd;
}

vec3 evaluateDiffuseAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    return evaluateNonSpecularAlbedo(surf, rd_i, macroNormal)
        * (1.0 - clamp(surf.S.y, 0.0, 1.0));
}

// Exact GLSL port of NRD.hlsli's material-factor front end. ReLAX consumes
// demodulated specular radiance, so both the fit and its deliberately biased
// stability floors are part of the denoiser contract, not optional tuning.
const float RELAX_NRD_EPS = 1e-6;
const float RELAX_NRD_MATERIAL_FACTOR_MIN_SCALE = 0.02;
const float RELAX_NRD_ROUGHNESS_FACTOR_MIN_SCALE = 0.1;

vec3 relaxNrdEnvironmentTermRtg(vec3 Rf0, float NoV,
    float perceptualRoughness) {
    // Ray Tracing Gems, Chapter 32, Equation 4. The official fit consumes
    // perceptual roughness and squares it internally to obtain GGX alpha.
    float m = clamp(perceptualRoughness * perceptualRoughness, 0.0, 1.0);

    vec4 X = vec4(1.0, NoV, NoV * NoV, NoV * NoV * NoV);
    vec4 Y = vec4(1.0, m, m * m, m * m * m);

    // Written explicitly to preserve HLSL mul(matrix, columnVector) semantics
    // and avoid a row/column-major transpose while porting the NRD constants.
    vec2 m1x = vec2(
            0.99044 * X.x - 1.28514 * X.y,
            1.29678 * X.x - 0.755907 * X.y);
    vec3 m2x = vec3(
            1.0 * X.x + 2.92338 * X.y + 59.4188 * X.w,
            20.3225 * X.x - 27.0302 * X.y + 222.592 * X.w,
            121.563 * X.x + 626.13 * X.y + 316.627 * X.w);
    vec2 m3x = vec2(
            0.0365463 * X.x + 3.32707 * X.y,
            9.0632 * X.x - 9.04756 * X.y);
    vec3 m4x = vec3(
            1.0 * X.x + 3.59685 * X.z - 1.36772 * X.w,
            9.04401 * X.x - 16.3174 * X.z + 9.22949 * X.w,
            5.56589 * X.x + 19.7886 * X.z - 20.2123 * X.w);

    float bias = dot(m1x, Y.xy)
            / max(dot(m2x, Y.xyw), RELAX_NRD_EPS);
    float scale = dot(m3x, Y.xy)
            / max(dot(m4x, Y.xyw), RELAX_NRD_EPS);
    return clamp(Rf0 * scale + vec3(bias), vec3(0.0), vec3(1.0));
}

vec3 evaluateSpecularAlbedo(
    material surf, vec3 rd_i, vec3 macroNormal, float etaRatio
) {
    float NoV = clamp(abs(dot(rd_i, macroNormal)), 0.0, 1.0);
    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float etaDenominator = max(abs(1.0 + etaRatio), 1e-4);
    float dielectricF0 = (1.0 - etaRatio) / etaDenominator;
    dielectricF0 *= dielectricF0;

    // Match evaluateSurfaceFresnel(): opaque/metallic lobes use LabPBR F0,
    // while transmissive interfaces use IOR Fresnel.
    vec3 Rf0 = mix(surf.Cs * surf.S.x, vec3(dielectricF0),
            transmissionSelector);

    // Dirt RT stores GGX alpha; NRD_MaterialFactors takes perceptual roughness.
    float perceptualRoughness = sqrt(clamp(surf.R.x, 0.0, 1.0));
    vec3 specFactor = relaxNrdEnvironmentTermRtg(
            clamp(Rf0, vec3(0.0), vec3(1.0)), NoV, perceptualRoughness);

    // These two lines intentionally reproduce NRD_MaterialFactors exactly.
    // They bias very dark/rough reflectors, preventing unstable demodulation.
    specFactor *= mix(RELAX_NRD_ROUGHNESS_FACTOR_MIN_SCALE,
            1.0, perceptualRoughness);
    specFactor = mix(vec3(RELAX_NRD_MATERIAL_FACTOR_MIN_SCALE),
            vec3(1.0), specFactor);

    if (any(isnan(specFactor)) || any(isinf(specFactor)))
        return vec3(RELAX_NRD_MATERIAL_FACTOR_MIN_SCALE);
    return clamp(specFactor,
        vec3(RELAX_NRD_MATERIAL_FACTOR_MIN_SCALE), vec3(1.0));
}

vec3 reproject(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - prevRaytracingCamPos;
    vec4 clipPos = rtPrevViewProjection * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

// -----------------------------------------------------------------------------------
// Direct-lighting helpers are defined next to the sun sampler below.
// ---------------------------------------------------------------------------

vec3 GetSpecularDominantDirection(vec3 N, vec3 V, float R) {
    float f = (1.0 - R) * (sqrt(1.0 - R) + R);
    vec3 R0 = reflect(V, N);
    return normalize(mix(N, R0, f));
}

// ===========================================================================
// Data Structures
// ===========================================================================

struct HalfVector {
    vec3 H;
    bool valid;
};

struct LobeProbs {
    float P_spec, P_refr, P_diff;
    vec3 specWeight, refrWeight, diffWeight;
    float F;
};

const float PT_DELTA_ROUGHNESS = 1e-5;

float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-30);
}

float sunSolidAngle() {
    return max(2.0 * PI * (1.0 - cosD_S), 1e-12);
}

float sunDirectionPdf(vec3 wi, vec3 lightDir) {
    vec3 sunDirection = -normalize(lightDir);
    return dot(wi, sunDirection) >= cosD_S
    ? 1.0 / sunSolidAngle() : 0.0;
}

bool isDeltaSpecular(float roughness) {
    return roughness <= PT_DELTA_ROUGHNESS;
}

struct MediumResult {
    vec3 absorption;
    vec3 emission;
};

struct GuideInfo {
    vec3 axis;
    float kappa;
    float prob;
    bool valid;
};

struct PSRResult {
    float virtualDist;
    float pathRoughness;
    vec3 refrDir;
};

struct FirstBounceData {
    vec3 p, macro_n, geometry_n, micro_n, rd_o, rd_i, refr_dir;
    vec3 specularAlbedo, diffuseAlbedo, transmissionAlbedo;
    vec3 emission_val, light_surf, absorption;
    float t, roughness, n_i, n_o, t2_ior_adjusted, pathRoughness;
    float reflectionHitDistance;
    vec3 reflectionEndpointOffset;
    vec3 surfaceMotion;
    float motionValid;
    int type, materialID;
};

// ===========================================================================
// Half-Vector Computation
// ===========================================================================

HalfVector computeHalfVector(vec3 wo, vec3 wi) {
    vec3 Hsum = wo + wi;
    float Hlen2 = dot(Hsum, Hsum);
    float valid = float(Hlen2 > 1e-12);
    HalfVector hv;
    hv.H = Hsum * inversesqrt(max(Hlen2, 1e-12)) * valid;
    hv.valid = valid > 0.5;
    return hv;
}

// ===========================================================================
// Disney Cook-Torrance BRDF Evaluation
// ===========================================================================

// Evaluate GGX microfacet BRDF: f(wo, wi) × NoL, and VNDF sampling PDF.
// Returns false if any degenerate angle; caller treats as bsdf_weight = 0.
// The caller applies MIS weighting externally (pdfMix or pdfNDF×P_spec).
vec3 evaluateSurfaceFresnel(vec3 wo, vec3 H, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio) {
    vec3 schlickF = reflectanceColor(Cs, abs(dot(wo, H))).rgb * Sx;
    float dielectricF = fresnel(wo, H, etaRatio);
    return mix(schlickF, vec3(dielectricF),
        clamp(transmissionSelector, 0.0, 1.0));
}

bool evaluateSpecularBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 Cs, float Sx,
    float transmissionSelector, float etaRatio, float roughness,
    out vec3 fSpecTimesNoL, out float pdfNDF
) {
    float NoV = dot(macroNormal, wo);
    float NoL = dot(macroNormal, wi);

    if (NoV <= 1e-5 || NoL <= 1e-5) {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    HalfVector hv = computeHalfVector(wo, wi);
    float NoH = dot(macroNormal, hv.H);
    float VoH = abs(dot(wo, hv.H));

    bool valid = hv.valid && NoH > 1e-5 && VoH > 1e-5;

    if (valid) {
        float rough = max(roughness, 1e-4);
        float D = GGX_D(NoH, rough);
        float G2 = GGX_G2_standard(NoV, NoL, rough);
        vec3 Fh = evaluateSurfaceFresnel(
                wo, hv.H, Cs, Sx, transmissionSelector, etaRatio);

        fSpecTimesNoL = Fh * D * G2 * NoL / max(4.0 * NoV * NoL, 1e-8);
        pdfNDF = GGX_vndf_pdf(wo, wi, macroNormal, rough);
    } else {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
    }

    return valid;
}

// Burley's Disney diffuse term. surf.R.x stores GGX alpha, while the Disney
// fit is parameterized by perceptual roughness, hence the square root.
float evaluateDisneyDiffuseFactor(
    vec3 wo, vec3 wi, vec3 macroNormal, float roughness
) {
    float NoV = clamp(dot(macroNormal, wo), 0.0, 1.0);
    float NoL = clamp(dot(macroNormal, wi), 0.0, 1.0);
    HalfVector hv = computeHalfVector(wo, wi);
    if (!hv.valid || NoV <= 0.0 || NoL <= 0.0) return 0.0;

    float LoH = clamp(dot(wi, hv.H), 0.0, 1.0);
    float perceptualRoughness = sqrt(clamp(roughness, 0.0, 1.0));
    float Fd90 = 0.5 + 2.0 * perceptualRoughness * LoH * LoH;
    float lightScatter = mix(1.0, Fd90, pow(1.0 - NoL, 5.0));
    float viewScatter = mix(1.0, Fd90, pow(1.0 - NoV, 5.0));
    return max(lightScatter * viewScatter, 0.0);
}

bool evaluateDisneyDiffuseBRDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 geometryNormal,
    vec3 diffuseColor, float roughness,
    out vec3 fDiffuseTimesNoL, out float pdfCosine
) {
    float NoL = dot(macroNormal, wi);
    float NoV = dot(macroNormal, wo);
    if (NoL <= 1e-6 || NoV <= 1e-6
            || dot(geometryNormal, wi) <= 0.0) {
        fDiffuseTimesNoL = vec3(0.0);
        pdfCosine = 0.0;
        return false;
    }

    float Fd = evaluateDisneyDiffuseFactor(
            wo, wi, macroNormal, roughness);
    fDiffuseTimesNoL = diffuseColor * (Fd * NoL / PI);
    pdfCosine = NoL / PI;
    return Fd > 0.0;
}

// Walter/PBRT rough-dielectric BTDF evaluated for a VNDF-sampled microfacet.
// etaRatio is eta_i / eta_t, matching GLSL refract().  The returned value is
// f_t * abs(NoL), and pdfNDF is the corresponding solid-angle PDF of wi.
bool evaluateTransmissionBSDF(
    vec3 wo, vec3 wi, vec3 macroNormal, vec3 microNormal,
    vec3 transmissionColor, float etaRatio, float roughness,
    out vec3 fTransmissionTimesNoL, out float pdfNDF
) {
    float NoV = dot(macroNormal, wo);
    float NoL = -dot(macroNormal, wi);
    if (NoV <= 1e-5 || NoL <= 1e-5 || etaRatio <= 1e-5) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    vec3 H = dot(microNormal, macroNormal) >= 0.0
        ? microNormal : -microNormal;
    float NoH = dot(macroNormal, H);
    float VoH = dot(wo, H);
    float LiH = dot(wi, H);
    if (NoH <= 1e-5 || VoH <= 1e-5 || LiH >= -1e-5) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    float rough = max(roughness, 1e-4);
    float etaP = 1.0 / etaRatio;
    float denom = LiH + VoH / etaP;
    float denom2 = denom * denom;
    if (denom2 <= 1e-12) {
        fTransmissionTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
        return false;
    }

    float D = GGX_D(NoH, rough);
    float G2 = GGX_G2_standard(NoV, NoL, rough);
    float T = 1.0 - clamp(fresnel(wo, H, etaRatio), 0.0, 1.0);
    float btdf = T * D * G2
            * abs(LiH * VoH / max(NoL * NoV * denom2, 1e-20));
    // Radiance transport across media is non-symmetric by eta^2.
    btdf /= max(etaP * etaP, 1e-8);

    float dH_dWi = abs(LiH) / denom2;
    pdfNDF = GGX_vndf_half_pdf(wo, H, macroNormal, rough) * dH_dWi;
    fTransmissionTimesNoL = transmissionColor * btdf * NoL;
    bool finiteResult = !isnan(pdfNDF) && !isinf(pdfNDF)
            && !any(isnan(fTransmissionTimesNoL))
            && !any(isinf(fTransmissionTimesNoL));
    if (!finiteResult) {
        pdfNDF = 0.0;
        fTransmissionTimesNoL = vec3(0.0);
    }
    return finiteResult && pdfNDF > 0.0;
}

// ===========================================================================
// BSDF Lobe Probabilities
// ===========================================================================

LobeProbs computeLobeProbs(material surf, vec3 rd_i, vec3 macroNormal, float rs) {
    LobeProbs p;
    // Lobe selection must be independent of the subsequently sampled GGX
    // half-vector. This gives the continuation strategy a well-defined PDF
    // that can be evaluated for an arbitrary NEE direction.
    p.F = clamp(fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
    vec4 rC = reflectanceColor(surf.Cs, dot(rd_i, macroNormal));

    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float diffuseSelector = 1.0 - transmissionSelector;

    vec3 interfaceF = mix(
            rC.rgb * surf.S.x,
            vec3(p.F),
            transmissionSelector);
    // Probabilities choose a non-zero transport lobe; they are not Fresnel
    // absorption probabilities. In particular, a metal has Cd == 0 and must
    // never spend samples on a zero-valued diffuse branch.
    float interfaceEnergy = clamp(luma(interfaceF), 0.0, 1.0);
    float baseEnergy = max(luma(max(surf.Cd, vec3(0.0))), 0.0);
    float remainingEnergy = max(1.0 - interfaceEnergy, 0.0);
    float specImportance = interfaceEnergy;
    float refrImportance = remainingEnergy * transmissionSelector * baseEnergy;
    float diffImportance = remainingEnergy * diffuseSelector * baseEnergy;
    float importanceSum = specImportance + refrImportance + diffImportance;

    if (importanceSum > 1e-8) {
        p.P_spec = specImportance / importanceSum;
        p.P_refr = refrImportance / importanceSum;
        p.P_diff = diffImportance / importanceSum;
    } else {
        p.P_spec = 0.0;
        p.P_refr = 0.0;
        p.P_diff = 0.0;
    }

    vec3 nonSpecColor = surf.Cd;
    p.specWeight = interfaceF / max(p.P_spec, 1e-5);
    p.refrWeight = surf.Cd * transmissionSelector / max(p.P_refr, 1e-5);
    p.diffWeight = nonSpecColor * diffuseSelector / max(p.P_diff, 1e-5);

    return p;
}

// ===========================================================================
// Volumetric Medium
// ===========================================================================

MediumResult evalMedium(float t, vec3 rd_i, float ro_i_y, bool inside,
    vec4 fogColor, vec3 globalEmission) {
    MediumResult m;
    if (inside) {
        m.absorption = exp2(-t * fogColor.yzw * LOG2_E);
        m.emission = (1.0 - m.absorption) / (fogColor.yzw + 1e-5) * globalEmission;
    } else {
        m.absorption = exp2(-max(b_Q * (b_P.x - ro_i_y) * t
                        - 0.5 * b_Q * t * t * rd_i.y, 0.0) * LOG2_E);
        m.emission = vec3(0.0);
    }
    return m;
}

// ===========================================================================
// ALICE Path Guiding
// ===========================================================================

GuideInfo computeAliceGuide(vec3 ro_o, float strengthMultiplier) {
    GuideInfo g;
    g.axis = vec3(0.0, 1.0, 0.0);
    g.kappa = 0.0;
    g.prob = 0.0;
    g.valid = false;

    vec2 prev_coord = reproject(ro_o).xy;
    bool validPrev = all(greaterThanEqual(prev_coord, vec2(0.0)))
            && all(lessThanEqual(prev_coord, vec2(1.0)));
    if (!validPrev) return g;

    vec4 guideY = samplePathGuide(prev_coord * vec2(resolution_global));
    vec3 x = guideY.xyz;
    float omega = guideY.w;
    float length_x = max(length(x), 1e-20);
    omega = max(omega, length_x);
    g.axis = x / length_x;
    float rho = clamp(length_x / omega, 0.0, 1.0);
    g.kappa = alice_kappa(length_x, omega);
    g.valid = length_x > 1e-8;
    g.prob = float(g.valid) * strengthMultiplier * rho;
    return g;
}

// ===========================================================================
// Diffuse Direction Sampling with ALICE MIS
// ===========================================================================

vec3 sampleDiffuseWithGuide(vec3 geometryNormal, vec3 shadingNormal,
    vec3 ro_o, GuideInfo guide, vec2 xi,
    out vec3 next_rd, out float guideWeight, out float sampledPdf) {
    sampledPdf = 0.0;
    bool useGuide = getRandom() < guide.prob;
    if (useGuide) {
        next_rd = sample_alice_guiding(guide.axis, guide.kappa, xi);
    } else {
        next_rd = SampleUniformHemisphere(shadingNormal, xi);
    }

    float NoL = max(0.0, dot(shadingNormal, next_rd));
    float geometryNoL = dot(geometryNormal, next_rd);
    // 如果采样到了几何半球下方，直接裁切
    if (NoL <= 0.0 || geometryNoL <= 0.0) {
        guideWeight = 0.0;
        return vec3(0.0);
    }

    float pdfUniform = 1.0 / (2.0 * PI);
    float pdfAlice = guide.prob > 0.0 ? alice_guiding_pdf(next_rd, guide.axis, guide.kappa) : 0.0;
    float pdfMix = (1.0 - guide.prob) * pdfUniform + guide.prob * pdfAlice;
    sampledPdf = pdfMix;
    guideWeight = (pdfMix > 1e-20) ? (pdfUniform / pdfMix) : 0.0;
    return vec3(guideWeight);
}

// ===========================================================================
// PSR (Primary Surface Replacement) Refractive Chain
// ===========================================================================

PSRResult tracePSRChain(vec3 ro, vec3 rd, vec3 geometryNormal, float firstRoughness,
    bool wasInside, int baseDepth) {
    PSRResult result;
    result.virtualDist = 0.0;
    result.pathRoughness = 0.0;
    result.refrDir = rd;

    float r_accum2 = firstRoughness * firstRoughness;
    float n_camera = wasInside ? REFRACTIVE_INDEX : 1.0;
    float r_term2 = PATH_ROUGHNESS_TERMINATE * PATH_ROUGHNESS_TERMINATE; // compare squared

    vec3 ro_chain = ro;
    vec3 rd_chain = rd;
    bool inside_chain = !wasInside;
    vec3 departN = geometryNormal;

    for (int i = 0; i < MAX_REFRACTIVE_BOUNCES; i++) {
        float offsetSign = inside_chain ? -1.0 : 1.0;
        vec3 ro_next, rd_next;
        float t_next = raycast(ro_chain + departN * offsetSign * 0.00025,
                rd_chain, ro_next, rd_next, !inside_chain, false);

        if (t_next < -0.5) {
            result.virtualDist = VPROJDIST_SKY;
            break;
        }

        Material hitMat = evaluateMaterial(tmp_Payload, rd_chain, uint(baseDepth + 1 + i));
        int hitBlockID;
        payload_unpackShadow(tmp_Payload.data, hitBlockID);
        material hitSurf = materialFromEvaluated(hitMat, hitBlockID);

        float n_segment = inside_chain ? REFRACTIVE_INDEX : 1.0;
        result.virtualDist += t_next * n_camera / n_segment;

        r_accum2 += hitSurf.R.x * hitSurf.R.x;

        if (!isTransmissiveBlock(hitBlockID)) break;

        vec3 hitGeomN = payload_unpackGeomNormal(tmp_Payload.data);
        hitGeomN = faceforward(hitGeomN, hitGeomN, rd_chain);

        float n_from = inside_chain ? REFRACTIVE_INDEX : 1.0;
        float n_to = inside_chain ? 1.0 : REFRACTIVE_INDEX;
        vec3 next_refract = refract(rd_chain, hitGeomN, n_from / n_to);

        if (dot(next_refract, next_refract) <= 0.0) break;

        rd_chain = next_refract;
        ro_chain = ro_next;
        inside_chain = !inside_chain;
        departN = hitGeomN;

        if (r_accum2 > r_term2) break; // branchless: r_accum2 vs squared threshold
    }

    result.pathRoughness = sqrt(r_accum2);
    return result;
}

// ===========================================================================
// First-Bounce Handlers (compile-time dispatched via #if defined)
// ===========================================================================

void handleFirstBounce_Reflection(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes, float etaRatio,
    out vec3 bsdf_weight, out vec3 next_rd,
    out float sampledStrategyPdf, out bool sampledDelta
) {
    vec3 wo = -rd_i;
    sampledStrategyPdf = 0.0;
    sampledDelta = isDeltaSpecular(surf.R.x);

    if (sampledDelta) {
        next_rd = reflect(rd_i, macroNormal);
        // Match Alpha-Piscium's geometric-hemisphere repair.  Strong mapped
        // normals must not send a visible reflection through the surface.
        if (dot(next_rd, geometryNormal) < 0.0)
            next_rd = reflect(next_rd, geometryNormal);
        bsdf_weight = evaluateSurfaceFresnel(wo, macroNormal, surf.Cs,
            surf.S.x, surf.S.y, etaRatio);
        return;
    }

    // Keep the continuation strategy equal to the VNDF density returned by
    // evaluateSpecularBRDF. Mixing a separate guide here would require a
    // mixture PDF for both throughput and NEE MIS.
    next_rd = reflect(rd_i, microNormal);
    // Folding an invalid VNDF reflection about the geometry normal changes
    // the sampling density and biases f/pdf. Reject it instead.
    if (dot(next_rd, geometryNormal) <= 0.0) {
        bsdf_weight = vec3(0.0);
        return;
    }

    vec3 wi = next_rd;
    vec3 fSpecTimesNoL_val;
    float pdfNDF;
    if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
            surf.S.y, etaRatio, surf.R.x,
            fSpecTimesNoL_val, pdfNDF)) {
        sampledStrategyPdf = pdfNDF;
        bsdf_weight = (pdfNDF > 1e-8)
            ? (fSpecTimesNoL_val / pdfNDF) : vec3(0.0);
    } else {
        bsdf_weight = vec3(0.0);
    }
}

void handleFirstBounce_Refraction(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes, float rs,
    out vec3 bsdf_weight, out vec3 next_rd, inout bool inside_state,
    out PSRResult psr, bool wasInside, int baseDepth
) {
    vec3 refract_dir = refract(rd_i, microNormal, rs);
    vec3 psr_refract_dir = refract(rd_i, geometryNormal, rs);

    if (dot(refract_dir, refract_dir) > 0.0) {
        next_rd = refract_dir;
        bool was_inverse_0 = inside_state;
        inside_state = !inside_state;

        bool psrEnabled = surf.R.x < PSR_ROUGHNESS_THRESHOLD;
        if (psrEnabled) {
            vec3 chain_rd = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
            psr = tracePSRChain(ro_o, chain_rd, geometryNormal, surf.R.x, was_inverse_0, baseDepth);
        } else {
            psr.virtualDist = 0.0;
            psr.pathRoughness = surf.R.x;
            psr.refrDir = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
        }

        vec3 fTransmissionTimesNoL;
        float pdfTransmission;
        vec3 transmissionColor = surf.Cd * clamp(surf.S.y, 0.0, 1.0);
        if (isDeltaSpecular(surf.R.x)) {
            float F = clamp(fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
            // Radiance transport through a delta dielectric carries eta_i^2 /
            // eta_t^2. The inverse factor at the exit interface cancels it.
            bsdf_weight = transmissionColor * (1.0 - F) * (rs * rs);
        } else {
            bool validTransmission = evaluateTransmissionBSDF(
                    -rd_i, next_rd, macroNormal, microNormal,
                    transmissionColor, rs, surf.R.x,
                    fTransmissionTimesNoL, pdfTransmission);
            bsdf_weight = validTransmission && pdfTransmission > 1e-8
                ? fTransmissionTimesNoL / pdfTransmission : vec3(0.0);
        }
    } else {
        // TIR
        next_rd = reflect(rd_i, microNormal);
        psr.virtualDist = 0.0;
        psr.pathRoughness = surf.R.x;
        psr.refrDir = next_rd;
        bsdf_weight = vec3(0.0);
    }
}

void handleFirstBounce_Diffuse(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal,
    material surf, LobeProbs lobes, vec2 xi,
    out vec3 bsdf_weight, out vec3 next_rd, out float sampledStrategyPdf
) {
    GuideInfo guide = computeAliceGuide(ro_o, PATH_GUIDING_STRENGTH);
    float guideWeight;
    sampleDiffuseWithGuide(
        geometryNormal, macroNormal, ro_o, guide, xi, next_rd, guideWeight,
        sampledStrategyPdf);
    // Uniform-hemisphere sampling estimates (1 / 2pi) * integral L dOmega.
    // ALICE supplies the cosine in composite, so 2 converts it to 1/pi. The
    // Disney factor supplies the non-Lambertian angular response.
    float Fd = evaluateDisneyDiffuseFactor(
            -rd_i, next_rd, macroNormal, surf.R.x);
    bsdf_weight = vec3(2.0 * guideWeight * Fd);
}

// ===========================================================================
// Secondary Bounce Handler (stochastic mixture)
// ===========================================================================

void handleSecondaryBounce(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes,
    out vec3 bsdf_weight, out vec3 next_rd, out int lobeType,
    out bool sampledSpecularLobe, out float sampledStrategyPdf,
    out bool sampledDeltaLobe, out bool neeCompatible,
    inout bool inside_state
) {
    float rnd_lobe = getRandom();
    sampledSpecularLobe = false;
    sampledStrategyPdf = 0.0;
    sampledDeltaLobe = false;
    neeCompatible = false;
    float n_i = inside_state ? REFRACTIVE_INDEX : 1.0;
    float n_o = inside_state ? 1.0 : REFRACTIVE_INDEX;
    float rs = n_i / n_o;

    if (rnd_lobe < lobes.P_spec) {
        // Reflection
        sampledSpecularLobe = true;
        lobeType = REFLECTION;
        sampledDeltaLobe = isDeltaSpecular(surf.R.x);
        neeCompatible = true;
        next_rd = reflect(rd_i,
                sampledDeltaLobe ? macroNormal : microNormal);
        if (dot(next_rd, geometryNormal) < 0.0) {
            if (sampledDeltaLobe)
                next_rd = reflect(next_rd, geometryNormal);
            else {
                bsdf_weight = vec3(0.0);
                return;
            }
        }
        if (dot(next_rd, macroNormal) > 0.0) {
            vec3 wo = -rd_i;
            vec3 wi = next_rd;
            if (sampledDeltaLobe) {
                vec3 F = evaluateSurfaceFresnel(
                        wo, macroNormal, surf.Cs, surf.S.x, surf.S.y, rs);
                bsdf_weight = F / max(lobes.P_spec, 1e-8);
                return;
            }
            vec3 fSpecTimesNoL_val;
            float pdfNDF;
            if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
                    surf.S.y, rs, surf.R.x,
                    fSpecTimesNoL_val, pdfNDF)) {
                bsdf_weight = (pdfNDF > 1e-8)
                    ? (fSpecTimesNoL_val / (pdfNDF * lobes.P_spec)) : vec3(0.0);
                sampledStrategyPdf = pdfNDF * lobes.P_spec;
            } else {
                bsdf_weight = vec3(0.0);
            }
        } else {
            bsdf_weight = vec3(0.0);
        }
    } else if (rnd_lobe < lobes.P_spec + lobes.P_refr) {
        // Refraction
        lobeType = REFRACTION;
        vec3 refract_dir = refract(rd_i, microNormal, rs);
        if (dot(refract_dir, refract_dir) > 0.0) {
            next_rd = refract_dir;
            vec3 fTransmissionTimesNoL;
            float pdfTransmission;
            vec3 transmissionColor = surf.Cd * clamp(surf.S.y, 0.0, 1.0);
            bool validTransmission;
            if (isDeltaSpecular(surf.R.x)) {
                float F = clamp(
                        fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
                bsdf_weight = transmissionColor * (1.0 - F) * (rs * rs)
                        / max(lobes.P_refr, 1e-8);
                validTransmission = true;
                sampledDeltaLobe = true;
            } else {
                validTransmission = evaluateTransmissionBSDF(
                        -rd_i, next_rd, macroNormal, microNormal,
                        transmissionColor, rs, surf.R.x,
                        fTransmissionTimesNoL, pdfTransmission);
                bsdf_weight = validTransmission && pdfTransmission > 1e-8
                    ? fTransmissionTimesNoL
                        / (pdfTransmission * max(lobes.P_refr, 1e-8)) : vec3(0.0);
            }
            inside_state = validTransmission ? !inside_state : inside_state;
        } else {
            next_rd = reflect(rd_i, microNormal);
            lobeType = REFLECTION;
            bsdf_weight = vec3(0.0);
        }
    } else {
        // Diffuse
        lobeType = DIFFUSION;
        neeCompatible = true;
        next_rd = DiffuseNormal(macroNormal, ro_o);
        vec3 fDiffuseTimesNoL;
        float pdfCosine;
        vec3 diffuseColor = surf.Cd
                * (1.0 - clamp(surf.S.y, 0.0, 1.0));
        bool validDiffuse = evaluateDisneyDiffuseBRDF(
                -rd_i, next_rd, macroNormal, geometryNormal,
                diffuseColor, surf.R.x,
                fDiffuseTimesNoL, pdfCosine);
        sampledStrategyPdf = lobes.P_diff * pdfCosine;
        bsdf_weight = validDiffuse && sampledStrategyPdf > 1e-8
            ? fDiffuseTimesNoL / sampledStrategyPdf : vec3(0.0);
    }
}

// ===========================================================================
// NEE: Direct Sunlight (branch-free per lobe type)
// ===========================================================================

bool sampleDirectSun(vec3 ro, vec3 geometryNormal, vec3 lightDir, bool inside,
    vec2 xi,
    out vec3 wi, out vec3 Li, out float lightPdf) {
    wi = -lightDir;
    Li = vec3(0.0);
    lightPdf = 1.0 / sunSolidAngle();
    ro += (dot(lightDir, geometryNormal) > 0.15 ? lightDir : geometryNormal) * 0.001;

    vec3 X, Y, Z;
    XYZ(lightDir, X, Y, Z);
    float r1 = xi.x;
    float alpha = xi.y * 2.0 * PI;
    float cosbeta = 1.0 - r1 * (1.0 - cosD_S);
    vec3 sampleDir = cosbeta * Y + sqrt(1.0 - cosbeta * cosbeta) * (cos(alpha) * X + sin(alpha) * Z);

    vec3 ro_o, rd_o;
    float t = raycast(ro, -sampleDir, ro_o, rd_o, !inside, true);
    if (t > -0.5) return false;

    wi = -sampleDir;
    // NEE owns only the solar disc. Atmospheric scattering remains in the
    // BSDF-sampled no-disc environment and is therefore never double counted.
    Li = sampleSkySunDisc(ro.y, wi, lightDir).xyz
            * payload_unpackShadow(tmp_Payload.data);
    return !any(isnan(Li)) && !any(isinf(Li));
}

bool sampleDirectSun(vec3 ro, vec3 geometryNormal, vec3 lightDir, bool inside,
    out vec3 wi, out vec3 Li, out float lightPdf) {
    return sampleDirectSun(ro, geometryNormal, lightDir, inside,
        vec2(getRandom(), getRandom()), wi, Li, lightPdf);
}

vec3 evalDirectDiffuse(vec3 ro, vec3 geometryNormal, vec3 shadingNormal,
    vec3 diffuseAlbedo, vec3 rd_i, vec3 lightDir, bool inside) {
    vec3 wi, Li;
    float lightPdf;
    if (!sampleDirectSun(
            ro, geometryNormal, lightDir, inside, wi, Li, lightPdf))
        return vec3(0.0);

    float OiN = dot(wi, shadingNormal);
    float oiNWeight = float(OiN > 0.0 && dot(wi, geometryNormal) > 0.0);

    return max(vec3(0.0), diffuseAlbedo * Li
            * (OiN * oiNWeight / max(PI * lightPdf, 1e-20)));
}

vec3 evalDirectDiffuseIncident(vec3 ro, vec3 geometryNormal,
    vec3 shadingNormal, vec3 rd_i, vec3 lightDir, bool inside) {
    // Unit Lambertian albedo makes the result E/pi without a fragile
    // component-wise divide by the material base color.
    return evalDirectDiffuse(ro, geometryNormal, shadingNormal, vec3(1.0),
        rd_i, lightDir, inside);
}

vec3 misLightContribution(
    vec3 fTimesNoL, vec3 Li, float lightPdf, float bsdfStrategyPdf
) {
    float misWeight = powerHeuristic(lightPdf, bsdfStrategyPdf);
    vec3 result = fTimesNoL * Li
            * (misWeight / max(lightPdf, 1e-20));
    return (!any(isnan(result)) && !any(isinf(result)))
    ? max(result, vec3(0.0)) : vec3(0.0);
}

bool loadSecondaryRadianceCache(
    vec3 surfacePosition,
    vec3 geometryNormal,
    out RadianceCache cache
) {
    cache = emptyCache();
    vec3 samplePosition = surfacePosition
            + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    vec3 currentCameraPosition = cam.viewInverse[3].xyz;
    vec3 voxelCoord = radianceCacheWorldToVoxel(
            samplePosition, currentCameraPosition);
    if (!isRadianceCacheSampleInBounds(voxelCoord)) return false;

    RadianceCacheAddress address = findRadianceCacheAddress(samplePosition);
    if (!radianceCacheAddressHasHistory(address, cam.frameId)) return false;
    cache = loadRadianceCachePlanes(
            address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
    return radianceCacheValueValid(cache);
}

vec3 evaluateCachedDiffuseLighting(
    RadianceCache cache,
    vec3 macroNormal,
    LobeProbs lobes
) {
    // diffWeight already contains diffuseAlbedo / P_diff.
    return radianceCacheDiffuseIncident(cache, macroNormal) * lobes.diffWeight;
}

vec3 evaluateCachedRoughSpecularLighting(
    RadianceCache cache,
    vec3 rd_i,
    vec3 macroNormal,
    material surf,
    LobeProbs lobes
) {
    // Broad GGX is approximated by a cosine convolution around its dominant
    // direction, then modulated by the integrated GGX DFG response.
    vec3 dominantDirection = GetSpecularDominantDirection(
            macroNormal, rd_i, sqrt(clamp(surf.R.x, 0.0, 1.0)));
    vec3 incidentResponse = radianceCacheDiffuseIncident(
            cache, dominantDirection);
    vec3 specularAlbedo = evaluateSpecularAlbedo(
            surf, rd_i, macroNormal, 1.0 / REFRACTIVE_INDEX);
    return incidentResponse * specularAlbedo / max(lobes.P_spec, 1e-5);
}

// ===========================================================================
// G-Buffer & Output
// ===========================================================================

FirstBounceData initFirstBounceData(vec3 ro, vec3 rd) {
    FirstBounceData fb;
    fb.p = ro;
    fb.macro_n = -rd;
    fb.geometry_n = -rd;
    fb.micro_n = -rd;
    fb.rd_o = rd;
    fb.rd_i = rd;
    fb.refr_dir = rd;
    fb.specularAlbedo = vec3(0.0);
    fb.diffuseAlbedo = vec3(0.0);
    fb.transmissionAlbedo = vec3(0.0);
    fb.emission_val = vec3(0.0);
    fb.light_surf = vec3(0.0);
    fb.absorption = vec3(1.0);
    fb.t = -1.0;
    fb.roughness = 1.0;
    fb.n_i = 1.0;
    fb.n_o = 1.0;
    fb.t2_ior_adjusted = 0.0;
    fb.pathRoughness = 0.0;
    fb.reflectionHitDistance = 0.0;
    fb.reflectionEndpointOffset = vec3(0.0);
    fb.surfaceMotion = vec3(0.0);
    fb.motionValid = 0.0;
    fb.type = -1;
    fb.materialID = 0;
    return fb;
}

void recordFirstBounceGBuffer(
    vec3 ro_o, vec3 ro, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, int materialID, vec3 rd_i, vec3 next_rd, float t,
    float n_i, float n_o, int lobeType, vec3 segmentEmission,
    vec3 currentAbsorption, inout FirstBounceData fb
) {
    fb.p = ro_o;
    fb.macro_n = macroNormal;
    fb.geometry_n = geometryNormal;
    fb.micro_n = microNormal;
    fb.t = t;
    fb.type = lobeType;
    fb.materialID = materialID;
    fb.n_i = n_i;
    fb.n_o = n_o;
    fb.rd_i = rd_i;
    fb.rd_o = next_rd;
    fb.emission_val = segmentEmission;
    fb.light_surf = surf.light;
    fb.roughness = surf.R.x;
    fb.absorption = currentAbsorption;

    fb.specularAlbedo = evaluateSpecularAlbedo(
            surf, rd_i, macroNormal, n_i / max(n_o, 1e-5));
    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    vec3 nonSpecularAlbedo = evaluateNonSpecularAlbedo(surf, rd_i, macroNormal);
    fb.diffuseAlbedo = nonSpecularAlbedo * (1.0 - transmissionSelector);
    fb.transmissionAlbedo = nonSpecularAlbedo * transmissionSelector;
}

void writePrimarySurfaceGBuffer(uvec2 xy, FirstBounceData fb,
    material surf, vec3 ro) {
    vec3 posRel = fb.p - ro;
    writeGeo0(GEO_N_GEO, xy, posRel, fb.t);
    writeGeo1(GEO_N_NORMALS, xy, fb.geometry_n, fb.roughness,
        fb.materialID, fb.roughness);
    writeAlbedosPath(GEO_N_ALBEDOS, xy,
        fb.specularAlbedo, fb.diffuseAlbedo, fb.macro_n);
    writeMisc(GEO_N_MISC, xy,
        fb.transmissionAlbedo, fb.emission_val, fb.rd_i);
    writeLightAbs(GEO_N_LIGHTABS, xy, fb.light_surf, fb.absorption);
    writeSurfaceMotion(xy, fb.surfaceMotion, fb.motionValid);
    writePrimaryMaterial(xy, surf.Cs, surf.Cd, surf.S);
}

void loadPrimarySurfaceGBuffer(uvec2 xy, vec3 ro,
    out FirstBounceData fb, out material surf) {
    fb = initFirstBounceData(ro, vec3(0.0, 0.0, -1.0));

    vec3 posRel;
    readGeo0(GEO_N_GEO, xy, posRel, fb.t);
    fb.p = ro + posRel;
    readGeo1(GEO_N_NORMALS, xy, fb.geometry_n, fb.roughness,
        fb.materialID, fb.pathRoughness);
    #if defined(FIRST_LOBE_REFLECTION)
    fb.specularAlbedo = readPrimarySpecularAlbedoMicroNormal(xy,
            fb.macro_n);
    #else
    fb.macro_n = readMicroNormal(GEO_N_MICRONORMAL, xy);
    #endif
    fb.micro_n = fb.macro_n;
    #if defined(FIRST_LOBE_REFRACTION)
    readPrimaryTransmissionAndRay(xy,
        fb.transmissionAlbedo, fb.rd_i);
    #else
    fb.rd_i = readPrimaryRayDirection(xy);
    #endif
    fb.rd_o = fb.rd_i;
    fb.refr_dir = fb.rd_i;

    vec3 Cs, Cd;
    vec2 S;
    readPrimaryMaterial(xy, Cs, Cd, S);
    surf = newMaterial(Cs, Cd, S, vec4(fb.roughness, 0.0, 0.0, 0.0),
            vec3(0.0));
}

void writeDiffuseOutput(uvec2 xy, FirstBounceData fb, vec3 L_indirect,
    vec3 L_direct_0, vec3 L_direct_0_dir, vec3 ro) {
    vec3 pos_rel = fb.p - ro;

    AliceEncoding combinedAlice = init_alice();
    float mask = 0.0;
    if (fb.t > -0.5) {
        L_indirect = clamp(L_indirect, 0.0, GI_CLAMP_MAX);
        L_direct_0 = clamp(L_direct_0, 0.0, 32000.0);
        AliceEncoding indAlice = radiance_to_alice(L_indirect, fb.rd_o);
        AliceEncoding dirAlice = radiance_to_alice(
                L_direct_0, L_direct_0_dir);
        indAlice.CoCg += dirAlice.CoCg;
        indAlice.aliceY += dirAlice.aliceY;
        combinedAlice = indAlice;
        mask = 1.0;
    }
    float currentMeanY2 = combinedAlice.aliceY.w * combinedAlice.aliceY.w; // Y² for 1-spp
    writeDiffuseLightRT(xy, combinedAlice, currentMeanY2);
    writeDiffuseGeo(xy, pos_rel, mask);
}

void writeReflectionOutput(uvec2 xy, FirstBounceData fb, vec3 totalIllumination, vec3 ro) {
    vec3 pos_rel = fb.p - ro;
    vec3 refl_R = fb.rd_o;
    float refl_vprojdist = fb.reflectionHitDistance;
    vec3 refl_color = vec3(0.0);
    if (fb.t > -0.5) {
        vec3 demodulated = totalIllumination / max(fb.specularAlbedo,
                    vec3(RELAX_NRD_MATERIAL_FACTOR_MIN_SCALE));
        if (!any(isnan(demodulated)) && !any(isinf(demodulated))) {
            refl_color = clamp(demodulated, 0.0,
                    200.0 * div_avgExposure);
        }
    }
    writeReflGeo(xy, pos_rel, refl_R);
    writeReflLight(xy, refl_color, refl_vprojdist, 0.0);
    RelaxEndpointMoments endpoint = emptyRelaxEndpointMoments();
    float endpointScale = clamp(VPROJDIST_SKY, 1.0, 65504.0);
    if (fb.reflectionHitDistance > 0.0 &&
            fb.reflectionHitDistance < 0.5 * endpointScale &&
            !any(isnan(fb.reflectionEndpointOffset)) &&
            !any(isinf(fb.reflectionEndpointOffset))) {
        // A reflected feature moves as the virtual image behind the local
        // macro plane, not as the real secondary hit in front of it. Mirror
        // the hit displacement at the primary reflector before storing its
        // moments. This orthogonal transform preserves E[|X|^2] and makes the
        // delta-specular endpoint an exact virtual reprojection point.
        vec3 virtualEndpointOffset = reflect(
                fb.reflectionEndpointOffset, fb.macro_n);
        endpoint.mean = virtualEndpointOffset / endpointScale;
        endpoint.secondMoment = dot(endpoint.mean, endpoint.mean);
    }
    writeReflEndpointMoments(xy, endpoint);
}

void writeRefractionOutput(uvec2 xy, FirstBounceData fb, vec3 totalIllumination, vec3 ro) {
    vec3 pos_rel = fb.p - ro;
    vec3 refr_R = fb.refr_dir;
    float refr_vprojdist = 0.0;
    vec3 refr_color = vec3(0.0);
    if (fb.t > -0.5) {
        refr_vprojdist = fb.t2_ior_adjusted;
        refr_color = clamp(totalIllumination / max(fb.transmissionAlbedo, vec3(1e-6)), 0.0, 200.0 * div_avgExposure);
    }
    writeRefrGeo(xy, pos_rel, refr_R);
    writeRefrLight(xy, refr_color, refr_vprojdist, 0.0);
    writePathRoughness(GEO_N_NORMALS, xy, fb.pathRoughness);
}

#if defined(PRIMARY_GBUFFER_PASS)
void TracePrimaryGBuffer(uvec2 xy, vec3 ro, vec3 rd) {
    uint eyeMedium = cam.flags & 3u;
    bool inside = eyeMedium != 0u;
    vec4 fogColor = eyeMedium == 2u
        ? vec4(0.0, 0.05, 0.075, 0.1) * 5.0 : vec4(0.0, 0.325, 0.295, 0.3);
    vec3 globalEmission = eyeMedium == 2u
        ? vec3(1.0, 0.25, 0.05) * 10.0 : vec3(0.0);

    FirstBounceData fb = initFirstBounceData(ro, rd);
    material surf = newMaterial(vec3(0.0), vec3(0.0), vec2(0.0),
            vec4(1.0, 0.0, 0.0, 0.0), vec3(0.0));

    vec3 hitPosition, hitDirection;
    float t = raycast(ro, rd, hitPosition, hitDirection, !inside, false);
    if (t < -0.5) {
        fb.t = -1.0;
        fb.absorption = inside ? vec3(0.0) : vec3(1.0);
    } else {
        vec4 primaryMotion = getPrimarySurfaceMotion(tmp_Payload);
        fb.surfaceMotion = primaryMotion.xyz;
        fb.motionValid = primaryMotion.w;

        Material surfaceMat = evaluateMaterial(tmp_Payload, rd, 0u);
        vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
        vec3 geometryNormal = faceforward(geomN, geomN, rd);
        vec3 macroNormal = surfaceMat.macroNormal;
        int blockID;
        payload_unpackShadow(tmp_Payload.data, blockID);
        surf = materialFromEvaluated(surfaceMat, blockID);
        int materialID = getRelaxMaterialID(tmp_Payload, blockID);
        markRadianceCacheGeometryHit(xy, hitPosition, geometryNormal);

        float nI = inside ? REFRACTIVE_INDEX : 1.0;
        float nO = inside ? 1.0 : REFRACTIVE_INDEX;
        MediumResult medium = evalMedium(t, rd, ro.y, inside,
                fogColor, globalEmission);
        recordFirstBounceGBuffer(hitPosition, ro, macroNormal,
            geometryNormal, macroNormal, surf, materialID, rd, rd, t,
            nI, nO, -1, medium.emission, medium.absorption, fb);
    }

    writePrimarySurfaceGBuffer(xy, fb, surf, ro);
}
#endif

// -----------------------------------------------------------------------------------
// Core: Forward Path Tracing
// -----------------------------------------------------------------------------------
void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir) {
    // === SETUP ===
    uint isEyeInWater = cam.flags & 3u;
    uvec2 xy = coord;
    bool originalInside = isEyeInWater != 0;
    bool inside = originalInside;

    vec3 ro_i = ro;
    vec3 rd_i = rd;

    vec3 throughput = vec3(1.0);
    vec3 L_indirect = vec3(0.0);
    vec3 L_direct_0 = vec3(0.0);
    vec3 L_direct_0_dir = -lightDir;
    float cascadedRoughness2 = 0.0;
    // Sampling metadata for MIS if the current continuation ray reaches the
    // solar disc. Delta/refraction events have no competing sun-NEE strategy.
    float lastBsdfStrategyPdf = 0.0;
    bool lastBsdfDelta = false;
    bool lastNeeCompatible = false;

    vec4 fogColor = (isEyeInWater == 2u) ? vec4(0, 0.05, 0.075, 0.1) * 5.0 : vec4(0, 0.325, 0.295, 0.3);
    vec3 globalEmission = (isEyeInWater == 2u) ? vec3(1, 0.25, 0.05) * 10.0 : vec3(0);

    float fireflyCap = FIREFLY_SUPPRESSION_MULTIPLIER * div_avgExposure;

    FirstBounceData fb = initFirstBounceData(ro, rd);

    // ===== FIRST BOUNCE =====
    #if defined(FIRST_LOBE_DIFFUSE) || defined(FIRST_LOBE_REFLECTION) || defined(FIRST_LOBE_REFRACTION)
    material surf;
    loadPrimarySurfaceGBuffer(xy, ro, fb, surf);
    vec3 ro_o = fb.p;
    vec3 rd_o = fb.rd_i;

    if (fb.t < -0.5) {
        throughput = vec3(0.0);
    } else {
        vec3 geometryNormal = fb.geometry_n;
        vec3 macroNormal = fb.macro_n;
        #if defined(FIRST_LOBE_DIFFUSE)
        // The diffuse continuation never consumes a GGX micro-normal.
        vec3 microNormal = macroNormal;
        #else
        vec3 microNormal = isDeltaSpecular(surf.R.x) ? macroNormal
            : GGXVNDFNormal(macroNormal, -fb.rd_i, surf.R.x,
                rtBlueNoise2D(xy, 0u));
        #endif
        float n_i = inside ? REFRACTIVE_INDEX : 1.0;
        float n_o = inside ? 1.0 : REFRACTIVE_INDEX;
        float rs = n_i / n_o;
        LobeProbs lobes = computeLobeProbs(surf, fb.rd_i, macroNormal, rs);

        vec3 bsdf_weight = vec3(0.0);
        vec3 next_rd = fb.rd_i;
        int current_type = -1;
        PSRResult psr;
        psr.virtualDist = 0.0;
        psr.pathRoughness = 0.0;
        psr.refrDir = fb.rd_i;

        #if defined(FIRST_LOBE_REFLECTION)
        current_type = REFLECTION;
        bool firstDelta;
        handleFirstBounce_Reflection(fb.rd_i, ro_o, macroNormal,
            geometryNormal, microNormal, surf, lobes, rs, bsdf_weight,
            next_rd, lastBsdfStrategyPdf, firstDelta);
        lastBsdfDelta = firstDelta;
        lastNeeCompatible = true;
        #elif defined(FIRST_LOBE_REFRACTION)
        current_type = REFRACTION;
        bool wasInside = inside;
        handleFirstBounce_Refraction(fb.rd_i, ro_o, macroNormal,
            geometryNormal, microNormal, surf, lobes, rs, bsdf_weight,
            next_rd, inside, psr, wasInside, 0);
        lastBsdfStrategyPdf = 0.0;
        lastBsdfDelta = true;
        lastNeeCompatible = false;
        fb.refr_dir = psr.refrDir;
        fb.t2_ior_adjusted = psr.virtualDist;
        fb.pathRoughness = psr.pathRoughness;
        #else
        current_type = DIFFUSION;
        handleFirstBounce_Diffuse(fb.rd_i, ro_o, macroNormal,
            geometryNormal, surf, lobes, rtBlueNoise2D(xy, 0u),
            bsdf_weight, next_rd,
            lastBsdfStrategyPdf);
        lastBsdfDelta = false;
        lastNeeCompatible = true;
        #endif

        bool firstHasSunNee = current_type == DIFFUSION
                || (current_type == REFLECTION && !isDeltaSpecular(surf.R.x));
        if (!isDarkened && firstHasSunNee) {
            vec3 sunWi, sunLi;
            float lightPdf;
            if (sampleDirectSun(ro_o, geometryNormal, lightDir, inside,
                    rtBlueNoise2D(xy, 1u),
                    sunWi, sunLi, lightPdf)) {
                L_direct_0_dir = sunWi;
                if (current_type == DIFFUSION
                        && dot(sunWi, geometryNormal) > 0.0
                        && dot(sunWi, macroNormal) > 0.0) {
                    GuideInfo directGuide = computeAliceGuide(
                            ro_o, PATH_GUIDING_STRENGTH);
                    float proposalPdf = (1.0 - directGuide.prob)
                            * (1.0 / (2.0 * PI));
                    proposalPdf += directGuide.prob * alice_guiding_pdf(
                                sunWi, directGuide.axis, directGuide.kappa);
                    float Fd = evaluateDisneyDiffuseFactor(
                            -fb.rd_i, sunWi, macroNormal, surf.R.x);
                    float misWeight = powerHeuristic(lightPdf, proposalPdf);
                    L_direct_0 = max(vec3(0.0), sunLi
                                * (Fd * misWeight / max(PI * lightPdf, 1e-20)));
                } else if (current_type == REFLECTION
                        && !isDeltaSpecular(surf.R.x)
                        && dot(sunWi, geometryNormal) > 0.0) {
                    vec3 fSpecTimesNoL;
                    float pdfNDF;
                    if (evaluateSpecularBRDF(-fb.rd_i, sunWi, macroNormal,
                            surf.Cs, surf.S.x, surf.S.y, rs, surf.R.x,
                            fSpecTimesNoL, pdfNDF)) {
                        L_direct_0 = misLightContribution(
                                fSpecTimesNoL, sunLi, lightPdf, pdfNDF);
                    }
                }
            }
        }

        fb.rd_o = next_rd;
        fb.micro_n = microNormal;
        fb.type = current_type;
        fb.n_i = n_i;
        fb.n_o = current_type == REFRACTION ? n_o : n_i;
        cascadedRoughness2 = current_type == DIFFUSION
            ? 1.0 : surf.R.x * surf.R.x;
        throughput *= bsdf_weight;
        ro_i = ro_o + geometryNormal
                    * (current_type == REFRACTION ? -0.001 : 0.001);
        rd_i = next_rd;
    }
    #else
    vec3 ro_o, rd_o;
    float t = raycast(ro_i, rd_i, ro_o, rd_o, !inside, false);

    if (t < -0.5) {
        // Primary ray hit sky
        vec3 sky = sampleSky(ro_i.y, rd_i, lightDir).xyz;
        if (any(isnan(sky)) || any(isinf(sky))) sky = vec3(0.0);
        L_indirect += throughput * sky;
        throughput = vec3(0.0);
        fb.t = -1.0;
        fb.absorption = originalInside ? vec3(0.0) : vec3(1.0);
    } else {
        vec4 primaryMotion = getPrimarySurfaceMotion(tmp_Payload);
        fb.surfaceMotion = primaryMotion.xyz;
        fb.motionValid = primaryMotion.w;
        // --- Material evaluation ---
        Material surfaceMat = evaluateMaterial(tmp_Payload, rd_i, 0u);
        vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
        vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
        #if defined(FIRST_LOBE_DIFFUSE)
        markRadianceCacheGeometryHit(xy, ro_o, geometryNormal);
        #endif
        vec3 macroNormal = surfaceMat.macroNormal;
        int blockID;
        payload_unpackShadow(tmp_Payload.data, blockID);
        material surf = materialFromEvaluated(surfaceMat, blockID);
        // tmp_Payload is shared by every trace issued by this invocation.
        // Freeze the primary-surface signature before PSR or sun NEE can
        // replace it with a secondary/shadow-hit payload. Otherwise material
        // continuity follows sun visibility and RELAX rejects valid history.
        int relaxMaterialID = getRelaxMaterialID(tmp_Payload, blockID);
        #if defined(FIRST_LOBE_DIFFUSE)
        vec3 microNormal = macroNormal;
        #else
        vec3 microNormal = isDeltaSpecular(surf.R.x) ? macroNormal
            : GGXVNDFNormal(macroNormal, -rd_i, surf.R.x,
                rtBlueNoise2D(xy, 0u));
        #endif
        float n_i = inside ? REFRACTIVE_INDEX : 1.0;
        float n_o = inside ? 1.0 : REFRACTIVE_INDEX;
        float rs = n_i / n_o;

        // --- Medium absorption ---
        MediumResult medium = evalMedium(t, rd_i, ro_i.y, inside, fogColor, globalEmission);

        // --- Lobe probabilities ---
        LobeProbs lobes = computeLobeProbs(surf, rd_i, macroNormal, rs);

        // --- First bounce direction: compile-time dispatched ---
        vec3 bsdf_weight = vec3(0.0);
        vec3 next_rd = rd_i;
        int current_type = -1;
        PSRResult psr;
        psr.virtualDist = 0.0;
        psr.pathRoughness = 0.0;
        psr.refrDir = rd;

        #if defined(FIRST_LOBE_REFLECTION)
        {
            current_type = REFLECTION;
            bool firstDelta;
            handleFirstBounce_Reflection(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, rs, bsdf_weight, next_rd,
                lastBsdfStrategyPdf, firstDelta);
            lastBsdfDelta = firstDelta;
            lastNeeCompatible = true;
        }
        #elif defined(FIRST_LOBE_REFRACTION)
        {
            current_type = REFRACTION;
            bool wasInside = inside;
            handleFirstBounce_Refraction(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, rs, bsdf_weight, next_rd,
                inside, psr, wasInside, 0);
            lastBsdfStrategyPdf = 0.0;
            lastBsdfDelta = true;
            lastNeeCompatible = false;
            fb.refr_dir = psr.refrDir;
            fb.t2_ior_adjusted = psr.virtualDist;
            fb.pathRoughness = psr.pathRoughness;
        }
        #else
        {
            current_type = DIFFUSION;
            handleFirstBounce_Diffuse(rd_i, ro_o, macroNormal, geometryNormal,
                surf, lobes, rtBlueNoise2D(xy, 0u), bsdf_weight, next_rd,
                lastBsdfStrategyPdf);
            lastBsdfDelta = false;
            lastNeeCompatible = true;
        }
        #endif

        // --- NEE at depth 0 ---
        bool firstHasSunNee = current_type == DIFFUSION
                || (current_type == REFLECTION
                    && !isDeltaSpecular(surf.R.x));
        if (!isDarkened && firstHasSunNee) {
            vec3 sunWi, sunLi;
            float lightPdf;
            if (sampleDirectSun(ro_o, geometryNormal, lightDir, inside,
                    rtBlueNoise2D(xy, 1u),
                    sunWi, sunLi, lightPdf)) {
                L_direct_0_dir = sunWi;

                if (current_type == DIFFUSION
                        && dot(sunWi, geometryNormal) > 0.0
                        && dot(sunWi, macroNormal) > 0.0) {
                    GuideInfo directGuide = computeAliceGuide(
                            ro_o, PATH_GUIDING_STRENGTH);
                    float proposalPdf = (1.0 - directGuide.prob)
                            * (1.0 / (2.0 * PI));
                    proposalPdf += directGuide.prob * alice_guiding_pdf(
                                sunWi, directGuide.axis, directGuide.kappa);
                    float Fd = evaluateDisneyDiffuseFactor(
                            -rd_i, sunWi, macroNormal, surf.R.x);
                    float misWeight = powerHeuristic(lightPdf, proposalPdf);
                    // ALICE/composition supplies NoL and diffuse base color.
                    L_direct_0 = max(vec3(0.0), sunLi
                                * (Fd * misWeight / max(PI * lightPdf, 1e-20)));
                } else if (current_type == REFLECTION
                        && !isDeltaSpecular(surf.R.x)
                        && dot(sunWi, geometryNormal) > 0.0) {
                    vec3 fSpecTimesNoL;
                    float pdfNDF;
                    if (evaluateSpecularBRDF(
                            -rd_i, sunWi, macroNormal,
                            surf.Cs, surf.S.x, surf.S.y, rs, surf.R.x,
                            fSpecTimesNoL, pdfNDF)) {
                        L_direct_0 = misLightContribution(
                                fSpecTimesNoL, sunLi, lightPdf, pdfNDF);
                    }
                }
            }
        }

        // --- Record G-Buffer ---
        recordFirstBounceGBuffer(ro_o, ro, macroNormal, geometryNormal, microNormal,
            surf, relaxMaterialID, rd_i, next_rd, t, n_i,
            (current_type == REFRACTION) ? n_o : n_i,
            current_type, medium.emission,
            medium.absorption, fb);

        // PSR uses sqrt(sum(r_i^2)); a diffuse event is fully rough.
        cascadedRoughness2 = current_type == DIFFUSION
            ? 1.0 : surf.R.x * surf.R.x;

        // --- Update throughput ---
        throughput *= bsdf_weight;

        // --- Advance ray ---
        ro_i = ro_o + geometryNormal * ((current_type == REFRACTION) ? -0.001 : 0.001);
        rd_i = next_rd;
    }

    #endif

    // ===== SECONDARY LOOP =====
    if (max(throughput.r, max(throughput.g, throughput.b)) > 0.0
            && !any(isnan(throughput)) && !any(isinf(throughput))) {
        // Track whether we arrived via specular (not just current lobe).
        // Primary specular → secondary surface should be cache-eligible at bounce 2.
        bool arrivedViaSpecular = false;
        #if defined(FIRST_LOBE_REFLECTION)
        arrivedViaSpecular = true;
        #endif
        for (int depth = 1; depth < MaxRay; depth++) {
            // --- Ray cast ---
            float t2 = raycast(ro_i, rd_i, ro_o, rd_o, !inside, false);

            // Distance of the actual noisy specular sample. This replaces the
            // unrelated extra ray previously traced along a fitted direction.
            if (depth == 1) {
                fb.reflectionHitDistance = (t2 > -0.5) ? t2 : VPROJDIST_SKY;
                fb.reflectionEndpointOffset = t2 > -0.5
                    ? (ro_o - fb.p) : vec3(0.0);
            }

            // --- Miss -> sky ---
            if (t2 < -0.5) {
                vec3 sky = sampleSkyNoSun(ro_i.y, rd_i, lightDir).xyz;
                vec3 sunDisc = sampleSkySunDisc(
                        ro_i.y, rd_i, lightDir).xyz;
                float discWeight = 1.0;
                float lightPdf = sunDirectionPdf(rd_i, lightDir);
                if (lastNeeCompatible && !lastBsdfDelta
                        && lightPdf > 0.0) {
                    discWeight = powerHeuristic(
                            lastBsdfStrategyPdf, lightPdf);
                }
                sky += sunDisc * discWeight;
                if (any(isnan(sky)) || any(isinf(sky))) sky = vec3(0.0);
                vec3 skyContrib = throughput * sky;
                if (depth >= 2) {
                    float lum = dot(skyContrib, vec3(0.2126, 0.7152, 0.0722));
                    skyContrib *= fireflyCap / max(lum, fireflyCap);
                }
                L_indirect += skyContrib;
                break;
            }

            // --- Material ---
            Material surfaceMat = evaluateMaterial(tmp_Payload, rd_i, uint(depth));
            vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
            vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
            #if defined(FIRST_LOBE_DIFFUSE)
            markRadianceCacheGeometryHit(xy, ro_o, geometryNormal);
            #endif
            vec3 macroNormal = surfaceMat.macroNormal;
            int blockID;
            payload_unpackShadow(tmp_Payload.data, blockID);
            material surf = materialFromEvaluated(surfaceMat, blockID);
            vec3 microNormal = isDeltaSpecular(surf.R.x) ? macroNormal
                : GGXVNDFNormal(macroNormal, -rd_i, surf.R.x, ro_o);
            float n_i2 = inside ? REFRACTIVE_INDEX : 1.0;
            float n_o2 = inside ? 1.0 : REFRACTIVE_INDEX;
            float rs2 = n_i2 / n_o2;
            bool surfaceInside = inside;

            // --- Medium ---
            MediumResult medium = evalMedium(t2, rd_i, ro_i.y, inside, fogColor, globalEmission);
            // Segment emission is accumulated before applying this segment's
            // transmittance.  Surface terms and all following bounces use the
            // attenuated throughput.
            vec3 bounceEmission = throughput * medium.emission;
            throughput *= medium.absorption;
            bounceEmission += throughput * surf.light;
            if (depth >= 2) {
                float lum = dot(bounceEmission, vec3(0.2126, 0.7152, 0.0722));
                bounceEmission *= fireflyCap / max(lum, fireflyCap);
            }
            L_indirect += bounceEmission;

            // --- Lobe probabilities ---
            LobeProbs lobes = computeLobeProbs(surf, rd_i, macroNormal, rs2);

            // --- Secondary bounce: stochastic mixture ---
            vec3 bsdf_weight;
            vec3 next_rd;
            int current_type;
            bool sampledSpecularLobe;
            float sampledStrategyPdf;
            bool sampledDeltaLobe;
            bool neeCompatible;
            handleSecondaryBounce(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, bsdf_weight, next_rd,
                current_type, sampledSpecularLobe, sampledStrategyPdf,
                sampledDeltaLobe, neeCompatible, inside);

            float nextCascadedRoughness2 = current_type == DIFFUSION
                ? 1.0 : cascadedRoughness2 + surf.R.x * surf.R.x;

            // --- Sun NEE ---
            // One visibility ray evaluates every non-delta reflection lobe.
            // Lobe selection is solely a continuation strategy and must not
            // decide whether direct lighting exists at this vertex.
            bool sampledDiffuseLobe = current_type == DIFFUSION;
            bool hasDiffuseSunNee = lobes.P_diff > 1e-8;
            bool hasSpecularSunNee = lobes.P_spec > 1e-8
                    && !isDeltaSpecular(surf.R.x);
            if (!isDarkened && (hasDiffuseSunNee || hasSpecularSunNee)) {
                vec3 sunWi, sunLi;
                float lightPdf;
                vec3 sunL = vec3(0.0);
                if (sampleDirectSun(
                        ro_o, geometryNormal, lightDir, surfaceInside,
                        sunWi, sunLi, lightPdf)) {
                    if (hasDiffuseSunNee) {
                        vec3 diffuseColor = surf.Cd
                                * (1.0 - clamp(surf.S.y, 0.0, 1.0));
                        vec3 fDiffuseTimesNoL;
                        float pdfCosine;
                        if (evaluateDisneyDiffuseBRDF(
                                -rd_i, sunWi, macroNormal, geometryNormal,
                                diffuseColor, surf.R.x,
                                fDiffuseTimesNoL, pdfCosine)) {
                            sunL += misLightContribution(
                                    fDiffuseTimesNoL, sunLi, lightPdf,
                                    lobes.P_diff * pdfCosine);
                        }
                    }

                    if (hasSpecularSunNee
                            && dot(sunWi, geometryNormal) > 0.0) {
                        vec3 fSpecTimesNoL;
                        float pdfNDF;
                        if (evaluateSpecularBRDF(
                                -rd_i, sunWi, macroNormal,
                                surf.Cs, surf.S.x, surf.S.y, rs2, surf.R.x,
                                fSpecTimesNoL, pdfNDF)) {
                            sunL += misLightContribution(
                                    fSpecTimesNoL, sunLi, lightPdf,
                                    lobes.P_spec * pdfNDF);
                        }
                    }
                }

                vec3 neeContrib = throughput * sunL;
                if (depth >= 2) {
                    float lum = dot(neeContrib, vec3(0.2126, 0.7152, 0.0722));
                    neeContrib *= fireflyCap / max(lum, fireflyCap);
                }
                L_indirect += neeContrib;
            }

            // --- Radiance-cache path termination ---
            // Query only after NEE so direct sunlight is never replaced.
            int bounceNumber = depth + 1;
            float roughThreshold2 =
                RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD
                    * RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD;
            // Cache-eligible when:
            //   Diffuse + setting (≥3 avoids corner block artifacts), OR
            //   Specular lobe (roughness-based OR sharp after 1 reflection), OR
            //   Arrived via specular — secondary surface after mirror reflection.
            bool diffuseEligible = sampledDiffuseLobe
                    && bounceNumber >= RADIANCE_CACHE_DIFFUSE_MIN_BOUNCE;
            bool specularLobeEligible = sampledSpecularLobe
                    && (nextCascadedRoughness2 >= roughThreshold2 || bounceNumber >= 2);
            bool viaSpecularEligible = arrivedViaSpecular
                    && bounceNumber >= 2;
            bool cacheEligible = !inside
                    && (diffuseEligible || specularLobeEligible || viaSpecularEligible);
            if (cacheEligible) {
                RadianceCache cache;
                if (loadSecondaryRadianceCache(
                        ro_o, geometryNormal, cache)) {
                    vec3 cachedLighting = sampledDiffuseLobe
                        ? evaluateCachedDiffuseLighting(
                            cache, macroNormal, lobes) : evaluateCachedRoughSpecularLighting(
                            cache, rd_i, macroNormal, surf, lobes);
                    vec3 cacheContrib = throughput * cachedLighting;
                    if (depth >= 2) {
                        float lum = dot(
                                cacheContrib,
                                vec3(0.2126, 0.7152, 0.0722));
                        cacheContrib *=
                            fireflyCap / max(lum, fireflyCap);
                    }
                    L_indirect += cacheContrib;
                    break;
                }
            }

            // --- Throughput update ---
            throughput *= bsdf_weight;
            if (any(isnan(throughput)) || any(isinf(throughput))) {
                throughput = vec3(0.0);
                break;
            }
            if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.0) break;
            cascadedRoughness2 = nextCascadedRoughness2;

            // --- Russian Roulette ---
            if (depth >= 2) {
                float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
                if (getRandom() > p) break;
                throughput /= p;
            }

            // --- Advance ---
            ro_i = ro_o + geometryNormal * ((current_type == REFRACTION) ? -0.001 : 0.001);
            rd_i = next_rd;
            arrivedViaSpecular = (current_type == REFLECTION);
            lastBsdfStrategyPdf = sampledStrategyPdf;
            lastBsdfDelta = sampledDeltaLobe;
            lastNeeCompatible = neeCompatible;
        }
    }

    // ===== OUTPUT =====
    if (any(isnan(L_indirect)) || any(isinf(L_indirect)))
        L_indirect = vec3(0.0);
    if (any(isnan(L_direct_0)) || any(isinf(L_direct_0)))
        L_direct_0 = vec3(0.0);
    vec3 totalIllumination = clamp(
            L_indirect + L_direct_0, 0.0, 65504.0);

    #if defined(FIRST_LOBE_DIFFUSE)
    writeDiffuseOutput(
        xy, fb, L_indirect, L_direct_0, L_direct_0_dir, ro);
    #elif defined(FIRST_LOBE_REFLECTION)
    writeReflectionOutput(xy, fb, totalIllumination, ro);
    #else
    writeRefractionOutput(xy, fb, totalIllumination, ro);
    #endif
}
