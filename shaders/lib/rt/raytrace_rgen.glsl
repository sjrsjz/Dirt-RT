// #version 460 core — declared by each enclosing rayN.rgen entry, which must #define exactly one
// of FIRST_LOBE_DIFFUSE / FIRST_LOBE_REFLECTION / FIRST_LOBE_REFRACTION (and FIRST_LOBE_VAL).
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
#include "/lib/rt/pom.glsl"
#include "/lib/common/bicubic.glsl"
#include "/lib/constants.glsl"
#include "/lib/settings.glsl"
#include "/lib/sky.glsl"
#include "/lib/math/quaternions.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/radiance_cache.glsl"
#include "/lib/pbr/material.glsl"
#include "/lib/common.glsl"

// First-bounce lobe forced by the enclosing rayN.rgen entry.
// Each entry #defines exactly one of FIRST_LOBE_DIFFUSE / FIRST_LOBE_REFLECTION / FIRST_LOBE_REFRACTION
// and #defines FIRST_LOBE_VAL to the corresponding int (DIFFUSION=2 / REFLECTION=1 / REFRACTION=3).
// The body uses #if defined() to produce three distinct compiled passes with zero dead code.
#if !defined(FIRST_LOBE_DIFFUSE) && !defined(FIRST_LOBE_REFLECTION) && !defined(FIRST_LOBE_REFRACTION)
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
layout(location = 6) rayPayloadEXT Payload payload;

#if !defined(RADIANCE_CACHE_TRACE)
void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir);
#endif

bool isDarkened = false;

#if defined(FIRST_LOBE_DIFFUSE)
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
void main() {
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
    Trace(uvec2(gl_LaunchIDEXT.xy), origin, direction, -lightDir_global);

    // Per-frame-once global state: only diffuse pass (ray0) pixel (0,0).
    // rtPrev=rtModelView is non-idempotent — if all 3 passes run it,
    // rtPrev gets overwritten to the current frame's matrix → reprojection failure → ghosting.
    #if defined(FIRST_LOBE_DIFFUSE)
    if (gl_LaunchIDEXT.xy == vec2(0)) {
        world_type_global = int(cam.world_type);
        frame_id = int(cam.frameId);
        camPos = origin;
        camY_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[2]), 0)).xyz;
        camX_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[1]), 0)).xyz;

        // Save current frame matrices as "previous" (for next frame's temporal reprojection)
        rtPrevModelView = rtModelView;
        rtPrevProjection = rtProjection;

        // ModelView: pure rotation (transpose of viewInverse), no translation
        rtModelView = mat4(transpose(mat3(cam.viewInverse)));

        // Projection: asymmetric frustum (with TAA jitter offset)
        float zNear = -cam.corners[0].z;
        float w = cam.corners[1].x - cam.corners[0].x;
        float h = cam.corners[2].y - cam.corners[0].y;
        float farD = 2048.0;
        rtProjection = mat4(0.0);
        rtProjection[0][0] = (2.0 * zNear) / w;
        rtProjection[1][1] = (2.0 * zNear) / h;
        rtProjection[2][0] = (cam.corners[1].x + cam.corners[0].x) / w;
        rtProjection[2][1] = (cam.corners[2].y + cam.corners[0].y) / h;
        rtProjection[2][2] = -(farD + zNear) / (farD - zNear);
        rtProjection[2][3] = -1.0;
        rtProjection[3][2] = -(2.0 * farD * zNear) / (farD - zNear);
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

    // --- TBN ---
    vec3 bitangent = cross(tangent, geomN) * bitangentSign;
    mat3 tbn = mat3(tangent, bitangent, geomN);

    vec2 localCoord = getRelativeUV(uv, atlas);

    // --- POM — first hit only ---
    vec2 sampleUV;
    vec2 res = vec2(textureSize(blockTexNormal, 0));

#if POM_ENABLED == 1
    vec2 derivatives;
    if (bounce == 0u) {
        sampleUV = computeParallaxUV(blockTexNormal, localCoord, atlas, rd_i, tbn, derivatives);
    } else {
        sampleUV = uv;
    }
#else
    sampleUV = uv;
#endif

    // --- Sample textures (bicubic albedo+specular first hit, bilinear otherwise) ---
    vec4 albedoTex, specularTex, normalTex;
    albedoTex = texture(blockTex, sampleUV);
    specularTex = texture(blockTexSpecular, sampleUV);

#if POM_ENABLED == 1
    if (bounce == 0u) {
        normalTex = textureBicubic(blockTexNormal, sampleUV, atlas, res);
    } else {
        normalTex = texture(blockTexNormal, sampleUV);
    }
#else
    normalTex = texture(blockTexNormal, sampleUV);
#endif

    albedoTex.rgb = pow(albedoTex.rgb * tint, vec3(2.2));

    return getMaterial(albedoTex, normalTex, specularTex, tbn,
        wetStrength_global, wetness_global, skylight, geomN);
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
    float trans = float(!isWater && mat.translucent > 0.9 && !isGlass);
    trans = isPortal ? 0.25 : trans;
    float roughness = (isWater || isPortal) ? 0.0 : mat.roughness;
    vec3 albedo = isWater ? vec3(1.0) : mat.albedo;
    vec3 emission = isPortal ? albedo * (1.0 - trans) : mat.emission;
    float specSelector = (isWater || isGlass) ? 1.0 : mix(trans, 1.0, metallic);

    return newMaterial(clamp(mat.F0, 0.0, 1.0), albedo,
        vec2(specSelector, 1.0 - trans),
        vec4(roughness > 0.01 ? max(roughness, 0.0125) : 0.0, trans,
            isWater, mat.subsurface_scattering),
        emission);
}

vec3 evaluateNonSpecularAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    float NoV = clamp(abs(dot(rd_i, macroNormal)), 0.0, 1.0);
    vec3 reflected = reflectanceColor(surf.Cs, NoV).rgb * surf.S.x;
    return surf.Cd * max(vec3(0.0), vec3(1.0) - reflected);
}

vec3 evaluateDiffuseAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    return evaluateNonSpecularAlbedo(surf, rd_i, macroNormal)
        * (1.0 - clamp(surf.S.y, 0.0, 1.0));
}

vec3 evaluateSpecularAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    float NoV = clamp(abs(dot(rd_i, macroNormal)), 0.0, 1.0);
    vec3 F0 = surf.Cs * surf.S.x;
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.040, -0.040);
    // The split-sum fit is parameterized by perceptual roughness, while R.x
    // stores GGX alpha = perceptualRoughness^2.
    float perceptualRoughness = sqrt(clamp(surf.R.x, 0.0, 1.0));
    vec4 r = fma(vec4(perceptualRoughness), c0, c1);
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2 AB = fma(vec2(-1.04, 1.04), vec2(a004), r.zw);
    return max(F0 * AB.x + vec3(AB.y * surf.S.x), vec3(1e-5));
}

vec3 reproject(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - prevRaytracingCamPos;
    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

// -----------------------------------------------------------------------------------
// Direct-lighting (NEE) moved to evalDirectDiffuse() / evalDirectSpecular() below
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
// GGX Microfacet BRDF Evaluation
// ===========================================================================

// Evaluate GGX microfacet BRDF: f(wo, wi) × NoL, and NDF sampling PDF.
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
        pdfNDF = GGX_ndf_pdf(wo, wi, macroNormal, rough);
    } else {
        fSpecTimesNoL = vec3(0.0);
        pdfNDF = 0.0;
    }

    return valid;
}

// Walter/PBRT rough-dielectric BTDF evaluated for an NDF-sampled microfacet.
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
    pdfNDF = GGXpdf(NoH, 0.0, rough) * dH_dWi;
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

LobeProbs computeLobeProbs(material surf, vec3 rd_i, vec3 microNormal, float rs) {
    LobeProbs p;
    p.F = clamp(fresnel(-rd_i, microNormal, rs), 0.0, 1.0);
    vec4 rC = reflectanceColor(surf.Cs, dot(rd_i, microNormal));

    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float diffuseSelector = 1.0 - transmissionSelector;

    vec3 interfaceF = mix(
        rC.rgb * surf.S.x,
        vec3(p.F),
        transmissionSelector);
    p.P_spec = clamp(luma(interfaceF), 0.0, 1.0);
    p.P_refr = (1.0 - p.P_spec) * transmissionSelector;
    p.P_diff = (1.0 - p.P_spec) * diffuseSelector;

    vec3 nonSpecColor = surf.Cd * max(vec3(0.0), vec3(1.0) - rC.rgb * surf.S.x);
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
    vec3 ro_o, GuideInfo guide,
    out vec3 next_rd, out float guideWeight) {
    
    bool useGuide = getRandom() < guide.prob;
    if (useGuide) {
        next_rd = sample_alice_guiding(guide.axis, guide.kappa, rand2(ro_o));
    } else {
        next_rd = SampleUniformHemisphere(shadingNormal, ro_o);
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
    out vec3 bsdf_weight, out vec3 next_rd
) {
    GuideInfo guide = computeAliceGuide(ro_o, PATH_GUIDING_SPECULAR_STRENGTH * surf.R.x);
    float reflGuideProb = guide.prob; // already 0 when !valid (set in computeAliceGuide)

    vec3 reflGGX_wi = reflect(rd_i, microNormal);
    bool useReflGuide = surf.R.x > 0.01 && reflGuideProb > 0.0 && getRandom() < reflGuideProb;

    if (useReflGuide) {
        next_rd = sample_alice_guiding(guide.axis, guide.kappa, vec2(getRandom(), getRandom()));
    } else {
        next_rd = reflGGX_wi;
    }

    vec3 wo = -rd_i;
    vec3 wi = next_rd;
    vec3 fSpecTimesNoL_val;
    float pdfNDF;
    if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
            surf.S.y, etaRatio, surf.R.x,
            fSpecTimesNoL_val, pdfNDF)) {
        float pdfAlice = alice_guiding_pdf(wi, guide.axis, guide.kappa);
        float pdfMix = (1.0 - reflGuideProb) * pdfNDF + reflGuideProb * pdfAlice;
        bsdf_weight = (pdfMix > 1e-8) ? (fSpecTimesNoL_val / pdfMix) : vec3(0.0);
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
        bool validTransmission = evaluateTransmissionBSDF(
            -rd_i, next_rd, macroNormal, microNormal,
            transmissionColor, rs, surf.R.x,
            fTransmissionTimesNoL, pdfTransmission);
        bsdf_weight = validTransmission && pdfTransmission > 1e-8
            ? fTransmissionTimesNoL / pdfTransmission : vec3(0.0);
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
    material surf, LobeProbs lobes,
    out vec3 bsdf_weight, out vec3 next_rd
) {
    GuideInfo guide = computeAliceGuide(ro_o, PATH_GUIDING_STRENGTH);
    float guideWeight;
    sampleDiffuseWithGuide(
        geometryNormal, macroNormal, ro_o, guide, next_rd, guideWeight);
    // Uniform-hemisphere sampling estimates (1 / 2pi) * integral L dOmega.
    // ALICE supplies the cosine in composite, so 2 converts it to E / pi.
    bsdf_weight = vec3(2.0 * guideWeight);
}

// ===========================================================================
// Secondary Bounce Handler (stochastic mixture)
// ===========================================================================

void handleSecondaryBounce(
    vec3 rd_i, vec3 ro_o, vec3 macroNormal, vec3 geometryNormal, vec3 microNormal,
    material surf, LobeProbs lobes,
    out vec3 bsdf_weight, out vec3 next_rd, out int lobeType,
    out bool sampledSpecularLobe, inout bool inside_state
) {
    float rnd_lobe = getRandom();
    sampledSpecularLobe = false;
    float n_i = inside_state ? REFRACTIVE_INDEX : 1.0;
    float n_o = inside_state ? 1.0 : REFRACTIVE_INDEX;
    float rs = n_i / n_o;

    if (rnd_lobe < lobes.P_spec) {
        // Reflection
        sampledSpecularLobe = true;
        lobeType = REFLECTION;
        next_rd = reflect(rd_i, microNormal);
        if (dot(next_rd, macroNormal) > 0.0) {
            vec3 wo = -rd_i;
            vec3 wi = next_rd;
            vec3 fSpecTimesNoL_val;
            float pdfNDF;
            if (evaluateSpecularBRDF(wo, wi, macroNormal, surf.Cs, surf.S.x,
                    surf.S.y, rs, surf.R.x,
                    fSpecTimesNoL_val, pdfNDF)) {
                bsdf_weight = (pdfNDF > 1e-8)
                    ? (fSpecTimesNoL_val / (pdfNDF * lobes.P_spec)) : vec3(0.0);
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
            bool validTransmission = evaluateTransmissionBSDF(
                -rd_i, next_rd, macroNormal, microNormal,
                transmissionColor, rs, surf.R.x,
                fTransmissionTimesNoL, pdfTransmission);
            bsdf_weight = validTransmission && pdfTransmission > 1e-8
                ? fTransmissionTimesNoL
                    / (pdfTransmission * max(lobes.P_refr, 1e-8))
                : vec3(0.0);
            inside_state = validTransmission ? !inside_state : inside_state;
        } else {
            next_rd = reflect(rd_i, microNormal);
            lobeType = REFLECTION;
            bsdf_weight = vec3(0.0);
        }
    } else {
        // Diffuse
        lobeType = DIFFUSION;
        next_rd = DiffuseNormal(macroNormal, ro_o);
        bsdf_weight = dot(next_rd, geometryNormal) > 0.0
            ? lobes.diffWeight : vec3(0.0);
    }
}

// ===========================================================================
// NEE: Direct Sunlight (branch-free per lobe type)
// ===========================================================================

bool sampleDirectSun(vec3 ro, vec3 geometryNormal, vec3 lightDir, bool inside,
    out vec3 wi, out vec3 Li) {
    wi = -lightDir;
    Li = vec3(0.0);
    ro += (dot(lightDir, geometryNormal) > 0.15 ? lightDir : geometryNormal) * 0.001;

    vec3 X, Y, Z;
    XYZ(lightDir, X, Y, Z);
    float r1 = getRandom();
    float alpha = getRandom() * 2.0 * PI;
    float cosbeta = 1.0 - r1 * (1.0 - cosD_S);
    vec3 sampleDir = cosbeta * Y + sqrt(1.0 - cosbeta * cosbeta) * (cos(alpha) * X + sin(alpha) * Z);

    vec3 ro_o, rd_o;
    float t = raycast(ro, -sampleDir, ro_o, rd_o, !inside, true);
    if (t > -0.5) return false;

    wi = -sampleDir;
    Li = sampleSky(ro.y, wi, lightDir).xyz
        * payload_unpackShadow(tmp_Payload.data);
    return !any(isnan(Li)) && !any(isinf(Li));
}

vec3 evalDirectDiffuse(vec3 ro, vec3 geometryNormal, vec3 shadingNormal,
    vec3 diffuseAlbedo, vec3 rd_i, vec3 lightDir, bool inside) {
    vec3 wi, Li;
    if (!sampleDirectSun(ro, geometryNormal, lightDir, inside, wi, Li))
        return vec3(0.0);

    float OiN = dot(wi, shadingNormal);
    float oiNWeight = float(OiN > 0.0 && dot(wi, geometryNormal) > 0.0);

    return max(vec3(0.0), diffuseAlbedo * Li
        * (2.0 * OiN * oiNWeight * (1.0 - cosD_S)));
}

vec3 evalDirectDiffuseAliceSample(vec3 ro, vec3 geometryNormal,
    vec3 lightDir, bool inside, out vec3 sampledWi) {
    vec3 Li;
    if (!sampleDirectSun(
            ro, geometryNormal, lightDir, inside, sampledWi, Li))
        return vec3(0.0);
    if (dot(sampledWi, geometryNormal) <= 0.0) return vec3(0.0);

    // p_light = 1 / (2pi(1-cosD)).  ALICE applies cos(theta) and composite
    // applies the diffuse albedo, so encode Li / (pi*p_light) here.
    return max(vec3(0.0), Li * (2.0 * (1.0 - cosD_S)));
}

vec3 evalDirectDiffuseIncident(vec3 ro, vec3 geometryNormal,
    vec3 shadingNormal, vec3 rd_i, vec3 lightDir, bool inside) {
    // Unit Lambertian albedo makes the result E/pi without a fragile
    // component-wise divide by the material base color.
    return evalDirectDiffuse(ro, geometryNormal, shadingNormal, vec3(1.0),
        rd_i, lightDir, inside);
}

vec3 evalDirectSpecular(vec3 ro, vec3 macroNormal, vec3 Cs, float Sx,
    float transmissionSelector, float roughness, vec3 rd_i,
    vec3 lightDir, bool inside) {
    ro += (dot(lightDir, macroNormal) > 0.15 ? lightDir : macroNormal) * 0.001;

    vec3 X, Y, Z;
    XYZ(lightDir, X, Y, Z);
    float r1 = getRandom();
    float alpha = getRandom() * 2.0 * PI;
    float cosbeta = 1.0 - r1 * (1.0 - cosD_S);
    vec3 sampleDir = cosbeta * Y + sqrt(1.0 - cosbeta * cosbeta) * (cos(alpha) * X + sin(alpha) * Z);

    vec3 ro_o, rd_o;
    float t = raycast(ro, -sampleDir, ro_o, rd_o, !inside, true);
    if (t > -0.5) return vec3(0.0);

    vec3 wi = -sampleDir;
    vec3 Li = sampleSky(ro.y, wi, lightDir).xyz * payload_unpackShadow(tmp_Payload.data);

    float IoN = abs(dot(rd_i, macroNormal));
    float OiN = dot(wi, macroNormal);
    float oiNWeight = float(OiN > 0.0); // branchless: 0 if back-facing, 1 otherwise
    float safeOiN = max(OiN, 1e-6);

    float a = max(roughness, 1e-6);
    vec3 H = normalize(wi - rd_i);
    float cosThetaH = clamp(dot(H, macroNormal), 0.0, 1.0);
    float D = GGX_D(cosThetaH, a);
    float G2 = GGX_G2_standard(IoN, safeOiN, a);
    float etaRatio = inside ? REFRACTIVE_INDEX : (1.0 / REFRACTIVE_INDEX);
    vec3 F = evaluateSurfaceFresnel(
        -rd_i, H, Cs, Sx, transmissionSelector, etaRatio);

    return max(vec3(0.0), D * G2 * F * Li
        * (PI * 0.5 * (1.0 - cosD_S) / max(IoN, 1e-6)))
        * oiNWeight;
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
        address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
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
        surf, rd_i, macroNormal);
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

    fb.specularAlbedo = evaluateSpecularAlbedo(surf, rd_i, macroNormal);
    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    vec3 nonSpecularAlbedo = evaluateNonSpecularAlbedo(surf, rd_i, macroNormal);
    fb.diffuseAlbedo = nonSpecularAlbedo * (1.0 - transmissionSelector);
    fb.transmissionAlbedo = nonSpecularAlbedo * transmissionSelector;
}

void writeDiffuseOutput(uvec2 xy, FirstBounceData fb, vec3 L_indirect,
    vec3 L_direct_0, vec3 L_direct_0_dir, vec3 ro) {
    vec3 pos_rel = fb.p - ro;
    writeGeo0(GEO_N_GEO, xy, pos_rel, fb.t);
    // Stable material continuity is required by RELAX. The previous first-lobe
    // value in this slot was never consumed by composition.
    writeGeo1(GEO_N_NORMALS, xy, fb.geometry_n, fb.roughness, fb.materialID, fb.roughness);
    writeMicroNormal(GEO_N_MICRONORMAL, xy, fb.macro_n);
    writeAlbedosPath(GEO_N_ALBEDOS, xy, fb.specularAlbedo, fb.diffuseAlbedo);
    writeMisc(GEO_N_MISC, xy, fb.transmissionAlbedo, fb.emission_val, fb.rd_i);
    writeLightAbs(GEO_N_LIGHTABS, xy, fb.light_surf, fb.absorption);

    AliceEncoding combinedAlice = init_alice();
    float mask = 0.0;
    if (fb.t > -0.5) {
        L_indirect = clamp(L_indirect, 0.0, 32000.0);
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
        refl_color = clamp(totalIllumination / max(fb.specularAlbedo, vec3(1e-6)), 0.0, 200.0 * div_avgExposure);
    }
    writeReflGeo(xy, pos_rel, refl_R);
    writeReflLight(xy, refl_color, refl_vprojdist, 0.0);
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

    vec4 fogColor = (isEyeInWater == 2u) ? vec4(0, 0.05, 0.075, 0.1) * 5.0 : vec4(0, 0.325, 0.295, 0.3);
    vec3 globalEmission = (isEyeInWater == 2u) ? vec3(1, 0.25, 0.05) * 10.0 : vec3(0);

    float fireflyCap = FIREFLY_SUPPRESSION_MULTIPLIER * div_avgExposure;

    FirstBounceData fb = initFirstBounceData(ro, rd);

    // ===== FIRST BOUNCE =====
    vec3 ro_o, rd_o;
    float t = raycast(ro_i, rd_i, ro_o, rd_o, !inside, false);

    if (t < -0.5) {
        // Primary ray hit sky
        vec3 sky = sampleSkyNoSun(ro_i.y, rd_i, lightDir).xyz;
        L_indirect += throughput * sky;
        throughput = vec3(0.0);
        fb.t = -1.0;
        fb.absorption = originalInside ? vec3(0.0) : vec3(1.0);
    } else {
        // --- Material evaluation ---
        Material surfaceMat = evaluateMaterial(tmp_Payload, rd_i, 0u);
        vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
        vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
#if defined(FIRST_LOBE_DIFFUSE)
        markRadianceCacheGeometryHit(xy, ro_o, geometryNormal);
#endif
        vec3 macroNormal = normalize(faceforward(surfaceMat.macroNormal, surfaceMat.macroNormal, -geometryNormal));
        int blockID;
        payload_unpackShadow(tmp_Payload.data, blockID);
        material surf = materialFromEvaluated(surfaceMat, blockID);
        vec3 microNormal = GGXNormal(macroNormal, surf.R.x, ro_o);
        float n_i = inside ? REFRACTIVE_INDEX : 1.0;
        float n_o = inside ? 1.0 : REFRACTIVE_INDEX;
        float rs = n_i / n_o;

        // --- Medium absorption ---
        MediumResult medium = evalMedium(t, rd_i, ro_i.y, inside, fogColor, globalEmission);

        // --- Lobe probabilities ---
        LobeProbs lobes = computeLobeProbs(surf, rd_i, microNormal, rs);

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
            handleFirstBounce_Reflection(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, rs, bsdf_weight, next_rd);
        }
        #elif defined(FIRST_LOBE_REFRACTION)
        {
            current_type = REFRACTION;
            bool wasInside = inside;
            handleFirstBounce_Refraction(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, rs, bsdf_weight, next_rd,
                inside, psr, wasInside, 0);
            fb.refr_dir = psr.refrDir;
            fb.t2_ior_adjusted = psr.virtualDist;
            fb.pathRoughness = psr.pathRoughness;
        }
        #else
        {
            current_type = DIFFUSION;
            handleFirstBounce_Diffuse(rd_i, ro_o, macroNormal, geometryNormal,
                surf, lobes, bsdf_weight, next_rd);
        }
        #endif

        // --- NEE at depth 0 ---
        #if defined(FIRST_LOBE_REFLECTION)
        float P_first = lobes.P_spec + lobes.P_refr * lobes.F;
        #elif defined(FIRST_LOBE_REFRACTION)
        float P_first = lobes.P_refr;
        #else
        float P_first = lobes.P_diff;
        #endif
        if (dot(geometryNormal, lightDir) < 0.0 && !isDarkened && P_first > 0.0) {
            vec3 sunL = vec3(0.0);
            if (current_type == DIFFUSION) {
                sunL = evalDirectDiffuseAliceSample(
                    ro_o, geometryNormal, lightDir, inside,
                    L_direct_0_dir);
            } else if (current_type == REFLECTION) {
                sunL = evalDirectSpecular(ro_o, macroNormal, surf.Cs,
                    surf.S.x, surf.S.y, surf.R.x,
                    rd_i, lightDir, inside);
            }
            L_direct_0 = sunL;
        }

        // --- Record G-Buffer ---
        recordFirstBounceGBuffer(ro_o, ro, macroNormal, geometryNormal, microNormal,
            surf, blockID, rd_i, next_rd, t, n_i,
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

    // ===== SECONDARY LOOP =====
    if (max(throughput.r, max(throughput.g, throughput.b)) > 0.0 && !any(isnan(throughput))) {
        for (int depth = 1; depth < MaxRay; depth++) {
            // --- Ray cast ---
            float t2 = raycast(ro_i, rd_i, ro_o, rd_o, !inside, false);

            // Distance of the actual noisy specular sample. This replaces the
            // unrelated extra ray previously traced along a fitted direction.
            if (depth == 1)
                fb.reflectionHitDistance = (t2 > -0.5) ? t2 : VPROJDIST_SKY;

            // --- Miss -> sky ---
            if (t2 < -0.5) {
                vec3 sky = sampleSkyNoSun(ro_i.y, rd_i, lightDir).xyz;
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
            vec3 macroNormal = normalize(faceforward(surfaceMat.macroNormal, surfaceMat.macroNormal, rd_i));
            int blockID;
            payload_unpackShadow(tmp_Payload.data, blockID);
            material surf = materialFromEvaluated(surfaceMat, blockID);
            vec3 microNormal = GGXNormal(macroNormal, surf.R.x, ro_o);
            float n_i2 = inside ? REFRACTIVE_INDEX : 1.0;
            float n_o2 = inside ? 1.0 : REFRACTIVE_INDEX;
            float rs2 = n_i2 / n_o2;

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
            LobeProbs lobes = computeLobeProbs(surf, rd_i, microNormal, rs2);

            // --- Secondary bounce: stochastic mixture ---
            vec3 bsdf_weight;
            vec3 next_rd;
            int current_type;
            bool sampledSpecularLobe;
            handleSecondaryBounce(rd_i, ro_o, macroNormal, geometryNormal, microNormal,
                surf, lobes, bsdf_weight, next_rd,
                current_type, sampledSpecularLobe, inside);

            float nextCascadedRoughness2 = current_type == DIFFUSION
                ? 1.0
                : cascadedRoughness2 + surf.R.x * surf.R.x;

            // --- NEE ---
            // The cache contains no direct-light estimate for this surface.
            // Keep one shadow ray and divide by the sampled lobe probability.
            bool sampledDiffuseLobe = current_type == DIFFUSION;
            float neeLobeProbability = sampledDiffuseLobe
                ? lobes.P_diff
                : (sampledSpecularLobe ? lobes.P_spec : 0.0);
            if (dot(geometryNormal, lightDir) < 0.0
                    && !isDarkened && neeLobeProbability > 1e-8) {
                vec3 sunL = vec3(0.0);
                if (sampledDiffuseLobe) {
                    vec3 selectedDiffuseAlbedo =
                        lobes.diffWeight * lobes.P_diff;
                    sunL = evalDirectDiffuse(
                        ro_o, geometryNormal, macroNormal,
                        selectedDiffuseAlbedo,
                        rd_i, lightDir, inside);
                } else {
                    sunL = evalDirectSpecular(ro_o, macroNormal, surf.Cs,
                        surf.S.x, surf.S.y, surf.R.x,
                        rd_i, lightDir, inside);
                }
                vec3 neeContrib =
                    throughput * sunL / neeLobeProbability;
                if (depth >= 2) {
                    float lum = dot(neeContrib, vec3(0.2126, 0.7152, 0.0722));
                    neeContrib *= fireflyCap / max(lum, fireflyCap);
                }
                L_indirect += neeContrib;
            }

            // --- Radiance-cache path termination ---
            // Query only after NEE so direct sunlight at this surface is never
            // replaced by the cache. The cache currently represents air-side
            // incident radiance, so paths inside a medium continue normally.
            float roughThreshold2 =
                RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD
                * RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD;
            bool roughSpecular = sampledSpecularLobe
                && nextCascadedRoughness2 >= roughThreshold2;
            int bounceNumber = depth + 1;
            bool diffuseCacheEligible = sampledDiffuseLobe
                && bounceNumber >= RADIANCE_CACHE_DIFFUSE_MIN_BOUNCE;
            bool cacheEligible = !inside
                && (diffuseCacheEligible || roughSpecular);
            if (cacheEligible) {
                RadianceCache cache;
                if (loadSecondaryRadianceCache(
                        ro_o, geometryNormal, cache)) {
                    vec3 cachedLighting = sampledDiffuseLobe
                        ? evaluateCachedDiffuseLighting(
                            cache, macroNormal, lobes)
                        : evaluateCachedRoughSpecularLighting(
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
            if (any(isnan(throughput))) {
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
        }
    }

    // ===== OUTPUT =====
    vec3 totalIllumination = clamp(L_indirect + L_direct_0, 0.0, 65504.0);

    #if defined(FIRST_LOBE_DIFFUSE)
    writeDiffuseOutput(
        xy, fb, L_indirect, L_direct_0, L_direct_0_dir, ro);
    #elif defined(FIRST_LOBE_REFLECTION)
    writeReflectionOutput(xy, fb, totalIllumination, ro);
    #else
    writeRefractionOutput(xy, fb, totalIllumination, ro);
    #endif
}
