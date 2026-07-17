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
#include "/lib/buffers/denoise.glsl"
#include "/lib/buffers/frame_data.glsl"
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
const float PSR_ROUGHNESS_THRESHOLD  = 0.15; // first-surface roughness above this → disable PSR, fall back to first-surface temporal accumulation
const float PATH_ROUGHNESS_TERMINATE = 0.8;  // accumulated sqrt(Σ r_i²) above this → terminate refractive chain early (too diffuse)
const int   MAX_REFRACTIVE_BOUNCES   = 4;    // max refractive surfaces to trace through before stopping

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
layout(binding = 6) writeonly uniform image2D RayTraceData;
layout(location = 6) rayPayloadEXT Payload payload;

void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir);

bool isDarkened = false;

void main() {
    vec2 px = vec2(gl_LaunchIDEXT.xy);
    vec2 p = px / vec2(gl_LaunchSizeEXT.xy);

    vec3 origin = cam.viewInverse[3].xyz;
    vec3 target = mix(mix(cam.corners[0], cam.corners[2], p.y), mix(cam.corners[1], cam.corners[3], p.y), p.x);
    vec3 direction = normalize((cam.viewInverse * vec4(target.xyz, 0.0)).xyz);

    setFrame(cam.frameId);
    isDarkened = world_type_global != WORLD_OVERWORLD && world_type_global != WORLD_THE_END && world_type_global != WORLD_THE_NETHER;

    wseed = floatBitsToUint(rand(direction * cam.frameId));

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

Payload tmp_Payload;

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0, int ignore_block_id, uint bounce_depth) {
    bool inside = !inverse_0;
    uint ignoreEnc = payload_encodeIgnoreID(ignore_block_id);
    payload_packFlags(payload.data, 0.0, inside, bounce_depth, false, ignoreEnc);
    payload_packShadow(payload.data, vec3(1.0), 0);
    float tMin = 0;
    float tMax = 2048.0;
    uint rayFlags = inverse_0 ? gl_RayFlagsCullBackFacingTrianglesEXT : 0u;
    traceRayEXT(acc, rayFlags, 0xFF, 0, 0, 0, ro, tMin, rd, tMax, 6);
    Payload hitPayload = payload;
    float t;
    ro_o = payload_unpackHitPos(hitPayload.data, t);
    rd_o = rd;
    tmp_Payload = hitPayload;
    return t;
}

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0) {
    return raycast(ro, rd, ro_o, rd_o, inverse_0, 0, 0u);
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
    bool _inside;
    uint _bounce;
    bool handedness;
    uint _ignore;
    payload_unpackFlags(pld.data, _inside, _bounce, handedness, _ignore);
    float bitangentSign = handedness ? 1.0 : -1.0;

    // --- TBN ---
    vec3 bitangent = cross(tangent, geomN) * bitangentSign;
    mat3 tbn = mat3(tangent, bitangent, geomN);

    vec2 localCoord = getRelativeUV(uv, atlas);

    // --- POM — first hit only ---
    vec2 sampleUV;
    vec2 derivatives;
    vec2 res = vec2(textureSize(blockTexNormal, 0));

    if (bounce == 0u) {
        sampleUV = computeParallaxUV(blockTexNormal, localCoord, atlas, rd_i, tbn, derivatives);
    } else {
        vec2 rcpRes = 1.0 / res;
        derivatives = pom_computeDerivatives(blockTexNormal, localCoord, atlas, res, rcpRes);
        sampleUV = uv;
    }

    // --- Sample textures (bicubic albedo+specular first hit, bilinear otherwise) ---
    vec4 albedoTex, specularTex, normalTex;
    albedoTex = texture(blockTex, sampleUV);
    specularTex = texture(blockTexSpecular, sampleUV);

    if (bounce == 0u) {
        normalTex = textureBicubic(blockTexNormal, sampleUV, atlas, res);
    } else {
        normalTex = texture(blockTexNormal, sampleUV);
    }

    albedoTex.rgb = pow(albedoTex.rgb * tint, vec3(2.2));

    return getMaterial(albedoTex, normalTex, specularTex, tbn,
        wetStrength_global, wetness_global, skylight, geomN);
}

// Check if a block is a transmissive/refractive surface (water, glass).
// LabPBR standard: translucent blocks are identified by the rendering layer,
// not by any material channel. In our ray-tracing pipeline we conservatively
// treat only water (1000) and glass (1001) as transmissive for the PSR chain.
bool isTransmissiveBlock(int blockID) {
    return blockID == 1000 || blockID == 1001;
}

// Convert Material (from getMaterial) to BSDF material struct
material materialFromEvaluated(Material mat, int blockID) {
    float metallic = mat.metallic;
    float trans = float(blockID != 1000 && 0.9 < mat.translucent && blockID != 1001);
    trans = blockID == 1002 ? 0.25 : trans;
    float roughness = blockID == 1000 ? 0.0 : mat.roughness;
    roughness = blockID == 1002 ? 0.0 : roughness;
    vec3 albedo = blockID == 1000 ? vec3(1.0) : mat.albedo;
    vec3 emission = blockID == 1002 ? albedo * (1.0 - trans) : mat.emission;
    float specSelector = blockID == 1000 ? 1.0 : mix(trans, 1.0, metallic);

    return newMaterial(clamp(mat.F0, 0.0, 1.0), albedo,
        vec2(specSelector, 1.0 - trans),
        vec4(roughness > 0.01 ? max(roughness, 0.0125) : 0.0, trans,
            blockID == 1000, mat.subsurface_scattering),
        emission);
}

vec3 reproject(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - prevRaytracingCamPos;
    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

// -----------------------------------------------------------------------------------
// Subsurface-scatter direct-lighting sample (NEE)
// -----------------------------------------------------------------------------------
vec3 sampleSunlight(vec3 ro, vec3 normal, vec3 Cs, vec3 Cd, vec3 rd_i, vec2 S, vec4 R, vec3 lightDir, bool night, int type, vec3 macroNormal, bool inside) {
    ro += (dot(lightDir, macroNormal) > 0.15 ? lightDir : macroNormal) * 0.001;

    vec3 X, Y, Z;
    XYZ(lightDir, X, Y, Z);
    float r1 = getRandom();
    float alpha = getRandom() * 2 * PI;
    float cosbeta = 1.0 - r1 * (1.0 - cosD_S);
    vec3 sampleDir = cosbeta * Y + sqrt(1.0 - cosbeta * cosbeta) * (cos(alpha) * X + sin(alpha) * Z);

    vec3 ro_o, rd_o;
    float t = raycast(ro, -sampleDir, ro_o, rd_o, !inside, 1001, 1u); // POM skipped for shadow rays
    if (t > -0.5) return vec3(0.0);

    vec3 wi = -sampleDir;
    vec3 Li = sampleSky(ro.y, wi, lightDir).xyz * payload_unpackShadow(tmp_Payload.data);

    float IoN = abs(dot(rd_i, normal));
    float OiN = dot(wi, normal);
    if (OiN <= 0.0) return vec3(0.0);

    if (type == DIFFUSION) {
        return max(vec3(0.0), Cd * Li * (2.0 * OiN * (1.0 - cosD_S)));
    }

    if (type == REFLECTION) {
        float a = max(R.x, 1e-6);
        vec3 H = normalize(wi - rd_i);
        float cosThetaH = clamp(dot(H, normal), 0.0, 1.0);
        float D = GGXpdf(cosThetaH, 0.0, a) / max(cosThetaH, 1e-6);
        float G2 = GGX_G2(IoN, OiN, a);
        vec3 F = reflectanceColor(Cs, abs(dot(rd_i, H))).xyz;
        return max(vec3(0.0), D * G2 * F * Li * (PI * 0.5 * (1.0 - cosD_S) / max(IoN, 1e-6)) * S.x);
    }

    return vec3(0.0);
}

vec3 GetSpecularDominantDirection(vec3 N, vec3 V, float R) {
    float f = (1.0 - R) * (sqrt(1.0 - R) + R);
    vec3 R0 = reflect(V, N);
    return normalize(mix(N, R0, f));
}

// -----------------------------------------------------------------------------------
// Core: Forward Path Tracing
// -----------------------------------------------------------------------------------
void Trace(uvec2 coord, vec3 ro, vec3 rd, vec3 lightDir) {
    uint isEyeInWater = cam.flags & 3u;
    uint idx = getIndex(coord);

    bool original_inverse_0 = isEyeInWater != 0;
    bool inverse_0 = original_inverse_0;

    vec3 ro_i = ro;
    vec3 rd_i = rd;

    vec3 throughput = vec3(1.0);
    vec3 L_indirect = vec3(0.0);
    vec3 L_direct_0 = vec3(0.0);
    vec3 current_absorption = vec3(1.0);

    vec4 fogColor = (isEyeInWater == 2u) ? vec4(0, 0.05, 0.075, 0.1) * 5.0 : vec4(0, 0.325, 0.295, 0.3);
    vec3 global_emission = (isEyeInWater == 2u) ? vec3(1, 0.25, 0.05) * 10.0 : vec3(0);

    float fireflyCap = FIREFLY_SUPPRESSION_MULTIPLIER * div_avgExposure;

    bool hit_sky_first = false;
    vec3 first_p = ro;
    vec3 first_n = -rd;
    vec3 first_macro_n = -rd;
    vec3 first_specularAlbedo = vec3(0.0);
    vec3 first_diffuseAlbedo = vec3(0.0);
    vec3 first_transmissionAlbedo = vec3(0.0);
    vec3 first_rd_o = rd;
    float first_t = -1.0;
    float first_roughness = 1.0;
    int first_type = -1;
    vec3 first_rd_i = rd;
    float first_n_i = 1.0;
    float first_n_o = 1.0;
    vec3 first_micro_n = -rd;
    vec2 mixWeight = vec2(0.0);
    vec3 first_absorption = vec3(1.0);
    vec3 first_emission_val = vec3(0.0);
    vec3 first_light_surf = vec3(0.0);
    float first_t2_ior_adjusted = 0.0; // IOR-adjusted virtual distance through refractive chain (PSR)
    float first_pathRoughness  = 0.0;  // accumulated sqrt(Σ r_i²) through refractive chain
    vec3  first_refr_dir       = rd;   // refracted direction at first surface (for edge-stopping)

    int depth = 0;
    for (; depth < MaxRay; depth++) {
        vec3 ro_o, rd_o;
        vec4 fogA = float(inverse_0) * fogColor;
        vec3 emissionA = float(inverse_0) * global_emission;

        float t = raycast(ro_i, rd_i, ro_o, rd_o, !inverse_0, 0, uint(depth));

        // --- 1. Miss: sky / background ---
        if (t < -0.5) {
            if (depth == 0) {
                hit_sky_first = true;
                first_t = -1.0;
                first_n = -rd_i;
                first_macro_n = -rd_i;
                first_rd_o = rd_i;
                first_absorption = original_inverse_0 ? vec3(0.0) : vec3(1.0);
            }

            bool skipFirstBounceSky = false;
            if (!skipFirstBounceSky) {
                vec3 sky = sampleSkyNoSun(ro_i.y, rd_i, lightDir).xyz;
                vec3 sky_contrib = throughput * sky * current_absorption;

                if (depth >= 2) {
                    float lum_sky = dot(sky_contrib, vec3(0.2126, 0.7152, 0.0722));
                    sky_contrib *= fireflyCap / max(lum_sky, fireflyCap);
                }

                L_indirect += sky_contrib;
            }

            break;
        }

        // --- 2. Geometry & material extraction ---
        bool insideFlag;
        uint bounceDepth;
        bool handedness;
        uint ignoreEnc;
        payload_unpackFlags(tmp_Payload.data, insideFlag, bounceDepth, handedness, ignoreEnc);
        Material surfaceMat = evaluateMaterial(tmp_Payload, rd_i, bounceDepth);

        // Geometry normal from payload (interpolated by rchit)
        vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
        vec3 macroNormal = faceforward(geomN, geomN, rd_i);

        vec3 normal = normalize(faceforward(surfaceMat.normal, surfaceMat.normal, rd_i));
        int blockID;
        vec3 _shadow = payload_unpackShadow(tmp_Payload.data, blockID);
        material surface = materialFromEvaluated(surfaceMat, blockID);

        vec3 microNormal = GGXNormal(normal, surface.R.x, ro_o);
        float n_i = inverse_0 ? REFRACTIVE_INDEX : 1.0;
        float n_o = inverse_0 ? 1.0 : REFRACTIVE_INDEX;
        float rs = n_i / n_o;

        // --- 3. Medium absorption & emission ---
        vec3 segment_absorption = exp2(-(inverse_0 ? t * fogA.yzw : max(b_Q * (b_P.x - ro_i.y) * t - 0.5 * b_Q * t * t * rd_i.y, 0.0)) * LOG2_E);
        current_absorption *= segment_absorption;

        vec3 segment_emission = (1.0 - segment_absorption) / (fogA.yzw + 1e-5) * emissionA;

        bool suppressSurfaceLight = (depth == 0);
        vec3 local_emission = (suppressSurfaceLight ? vec3(0.0) : surface.light) + segment_emission;

        vec3 bounce_emission = throughput * local_emission;
        if (depth >= 2) {
            float lum_emit = dot(bounce_emission, vec3(0.2126, 0.7152, 0.0722));
            bounce_emission *= fireflyCap / max(lum_emit, fireflyCap);
        }
        L_indirect += bounce_emission;

        // --- 4. BSDF lobe probabilities & weights ---
        float F = clamp(fresnel(-rd_i, microNormal, rs), 0.0, 1.0);
        vec4 rC = reflectanceColor(surface.Cs, dot(rd_i, microNormal));

        float P_spec = clamp(rC.w * surface.S.x, 0.0, 1.0);
        float transmissionSelector = clamp(surface.S.y, 0.0, 1.0);
        float diffuseSelector = 1.0 - transmissionSelector;
        float P_refr = (1.0 - P_spec) * transmissionSelector;
        float P_diff = (1.0 - P_spec) * diffuseSelector;

        vec3 nonSpecColor = surface.Cd * max(vec3(0.0), vec3(1.0) - rC.rgb * surface.S.x);
        vec3 specLobeWeight = rC.rgb * surface.S.x / max(P_spec, 1e-5);
        vec3 refrLobeWeight = nonSpecColor * transmissionSelector / max(P_refr, 1e-5);
        vec3 diffLobeWeight = nonSpecColor * diffuseSelector / max(P_diff, 1e-5);

        int current_type = -1;
        vec3 next_rd = rd_i;
        vec3 bsdf_weight = vec3(0.0);

        if (depth == 0) {
            // --- First bounce: compile-time forced lobe (#if defined) ---
            // Weight = P_lobe * <lobeWeight>: removes single-pass mixture /P_lobe compensation,
            // making each buffer an unbiased per-lobe estimate (E = L_lobe) every frame.
            // P_lobe=0 → weight=0 → path ends cleanly.
            #if defined(FIRST_LOBE_REFLECTION)
            current_type = REFLECTION;

            // ---- ALICE 引导概率 (clamp!) ----
            float reflGuideProb = 0.0;
            float reflKappa = 0.0;
            vec3 reflAxis = macroNormal;

            {
                vec2 prev_coord = reproject(ro_o).xy;
                bool validPrev = all(greaterThanEqual(prev_coord, vec2(0.0))) &&
                        all(lessThanEqual(prev_coord, vec2(1.0)));
                if (validPrev) {
                    DiffuseIlluminationWriteData data0 =
                        samplePrevDiffuse(prev_coord * resolution_global);
                    vec3 x = data0.data_swap.aliceY.xyz;
                    float omega = data0.data_swap.aliceY.w;
                    float length_x = max(length(x), 1e-20);
                    omega = max(omega, length_x);
                    reflAxis = x / length_x;
                    float rho = clamp(length_x / omega, 0.0, 1.0);
                    reflKappa = alice_kappa(length_x, omega);
                    reflGuideProb = length_x > 1e-8
                        ? PATH_GUIDING_SPECULAR_STRENGTH * surface.R.x * rho : 0.0;
                }
            }

            // ---- 采样：NDF 或 ALICE ----
            vec3 reflGGX_wi = reflect(rd_i, microNormal); // microNormal 由 GGXNormal() 生成
            bool useReflGuide = surface.R.x > 0.01 && reflGuideProb > 0.0 && getRandom() < reflGuideProb;

            if (useReflGuide) {
                next_rd = sample_alice_guiding(reflAxis, reflKappa, vec2(getRandom(), getRandom()));
            } else {
                next_rd = reflGGX_wi;
            }

            // ---- 统一 evaluate BRDF 并除以 mixture PDF ----
            vec3 wo = -rd_i;
            vec3 wi = next_rd;

            float NoV = dot(normal, wo);
            float NoL = dot(normal, wi);

            if (NoV > 1e-6 && NoL > 1e-6) {
                vec3 Hsum = wo + wi;
                float Hlen2 = dot(Hsum, Hsum);

                if (Hlen2 > 1e-12) {
                    vec3 H = Hsum * inversesqrt(Hlen2);

                    float NoH = dot(normal, H);
                    float VoH = dot(wo, H);

                    if (NoH > 1e-6 && VoH > 1e-6) {
                        float rough = max(surface.R.x, 1e-4);

                        float D_NoH = GGXpdf(NoH, 0.0, rough);
                        float D = D_NoH / NoH;
                        float G2 = GGX_G2_standard(NoV, NoL, rough);
                        vec3 Fh = reflectanceColor(surface.Cs, VoH).rgb;

                        // 标准 microfacet BRDF
                        vec3 fSpec = Fh * surface.S.x * D * G2 / max(4.0 * NoV * NoL, 1e-8);

                        float pdfNDF = GGX_ndf_pdf(wo, wi, normal, rough);
                        float pdfAlice = alice_guiding_pdf(wi, reflAxis, reflKappa);
                        float pdfMix = (1.0 - reflGuideProb) * pdfNDF + reflGuideProb * pdfAlice;

                        bsdf_weight = (pdfMix > 1e-8) ? (fSpec * NoL / pdfMix) : vec3(0.0);
                    }
                }
            }

            #elif defined(FIRST_LOBE_REFRACTION)
            // [Refraction: transmission only, no Fresnel sub-event]
            // TIR (total internal reflection) → bsdf_weight=0 — that energy is covered by the
            // reflection pass's F term.  Transmission direction driven by GGX microNormal.
            //
            // PSR (Primary Surface Replacement):
            //   Trace through the refractive chain to find the first non-refractive surface.
            //   Accumulate IOR-adjusted virtual distance: t_virtual = Σ(t_i / n_i).
            //   Virtual image position: P_virtual = C + V × (t1 + Σ[t_i / n_i]).
            //   If the first surface is too rough, disable PSR and fall back to first-surface
            //   temporal accumulation (virtualProjDist = 0).
            current_type = REFRACTION;
            // ---- 用于实际路径追踪的 BSDF 方向 (POM 细节) ----
            vec3 refract_dir = refract(rd_i, microNormal, rs);
            // ---- 用于 PSR 链追踪的稳定折射方向 (几何宏法线, 不受 POM/jitter 影响) ----
            vec3 psr_refract_dir = refract(rd_i, macroNormal, rs);
            if (dot(refract_dir, refract_dir) > 0.0) {
                next_rd = refract_dir;
                // oct_dir 也存几何折射方向 → 时域边缘停止更稳定
                first_refr_dir = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
                bool was_inverse_0 = inverse_0;
                inverse_0 = !inverse_0;

                // ---- PSR roughness gate ----
                float r_accum2 = surface.R.x * surface.R.x;
                bool psrEnabled = surface.R.x < PSR_ROUGHNESS_THRESHOLD;

                if (psrEnabled) {
                    vec3 ro_chain = ro_o;
                    vec3 rd_chain = dot(psr_refract_dir, psr_refract_dir) > 0.0 ? psr_refract_dir : refract_dir;
                    bool inside_chain = !was_inverse_0; // medium state after first refract
                        vec3 departN = macroNormal;          // normal of surface we're departing from
                        // n_camera: original camera medium IOR (constant; derivation → t_virtual = t × n_camera / n_segment)
                        float n_camera = was_inverse_0 ? REFRACTIVE_INDEX : 1.0;

                        for (int refr_depth = 0; refr_depth < MAX_REFRACTIVE_BOUNCES; refr_depth++) {
                        vec3 ro_next, rd_next;
                        float t_next = raycast(ro_chain + departN * (inside_chain ? -0.00025 : 0.00025),
                                                rd_chain, ro_next, rd_next, !inside_chain, 0, uint(depth + 1 + refr_depth));

                        if (t_next < -0.5) {
                            // Sky — virtual image at infinity
                            first_t2_ior_adjusted = VPROJDIST_SKY;
                            break;
                        }

                        // Evaluate hit material
                        Material hitMat = evaluateMaterial(tmp_Payload, rd_chain, uint(depth + 1 + refr_depth));
                        int hitBlockID;
                        payload_unpackShadow(tmp_Payload.data, hitBlockID);
                        material hitSurf = materialFromEvaluated(hitMat, hitBlockID);

                        // IOR-adjusted virtual distance for this segment.
                        // Derivation: from Snell's law, t_virtual = t_segment × n_camera / n_segment
                        // where n_segment is the medium the ray traveled through for this segment.
                        float n_segment = inside_chain ? REFRACTIVE_INDEX : 1.0;
                        first_t2_ior_adjusted += t_next * n_camera / n_segment;

                        // Accumulate roughness (variance-additive)
                        r_accum2 += hitSurf.R.x * hitSurf.R.x;

                        // PSR anchor: only continue through water / glass surfaces.
                        if (!isTransmissiveBlock(hitBlockID)) {
                            // Non-transmissive surface (diffuse/reflective) — PSR anchor found
                            break;
                        }

                        // Transmissive surface — continue through with GEOMETRIC normal
                        // (PSR chain uses geometric normals exclusively — no GGX, no POM —
                        //  so virtualProjDist is deterministic frame to frame)
                        vec3 hitGeomN = payload_unpackGeomNormal(tmp_Payload.data);
                        hitGeomN = faceforward(hitGeomN, hitGeomN, rd_chain);

                        // IOR ratio for Snell's refraction: n_from / n_to
                        float n_from_refr = inside_chain ? REFRACTIVE_INDEX : 1.0; // medium ray is coming FROM
                        float n_to_refr   = inside_chain ? 1.0 : REFRACTIVE_INDEX; // medium ray is going TO
                        float rs_chain = n_from_refr / n_to_refr;
                        vec3 next_refract = refract(rd_chain, hitGeomN, rs_chain);

                        if (dot(next_refract, next_refract) <= 0.0) {
                            // TIR — chain terminates
                            break;
                        }

                        rd_chain = next_refract;
                        ro_chain = ro_next;
                        inside_chain = !inside_chain;
                        departN = hitGeomN; // surface normal we'll offset from next iteration

                        // Early termination: path too diffuse
                        if (sqrt(r_accum2) > PATH_ROUGHNESS_TERMINATE) break;
                    }
                }
                // else: psrEnabled=false → first_t2_ior_adjusted stays 0.0 → fall back to
                //       first-surface temporal accumulation (virtualProjDist=0).

                first_pathRoughness = sqrt(r_accum2);
                bsdf_weight = P_refr * refrLobeWeight * (1.0 - F);
            } else {
                // TIR — transmission impossible, zero contribution; pick valid next_rd
                next_rd = reflect(rd_i, microNormal);
                first_refr_dir = next_rd; // TIR → oct_dir uses reflected direction
            }
            #else
            // [Diffuse + ALICE screen-space path guiding]
            current_type = DIFFUSION;
            float guideWeight = 1.0;
            float guideProb = 0.0;
            float kappa = 0.0;
            vec3 axis = macroNormal;

            vec2 prev_coord = reproject(ro_o).xy;
            bool validPrev = all(greaterThanEqual(prev_coord, vec2(0.0))) &&
                    all(lessThanEqual(prev_coord, vec2(1.0)));
            if (validPrev) {
                DiffuseIlluminationWriteData data0 =
                    samplePrevDiffuse(prev_coord * resolution_global);
                vec3 x = data0.data_swap.aliceY.xyz;
                float omega = data0.data_swap.aliceY.w;
                float length_x = max(length(x), 1e-20);
                omega = max(omega, length_x);
                axis = x / length_x;
                float rho = clamp(length_x / omega, 0.0, 1.0);
                kappa = alice_kappa(length_x, omega);
                // If length_x is too small, skip guiding — tiny values destabilize axis → NaN.
                guideProb = length_x > 1e-8 ? PATH_GUIDING_STRENGTH * rho : 0.0;
            }
            bool useGuide = getRandom() < guideProb;
            if (useGuide) {
                next_rd = sample_alice_guiding(axis, kappa, vec2(getRandom(), getRandom()));
            } else {
                next_rd = DiffuseNormal(macroNormal, ro_o);
            }
            float NoL = max(0.0, dot(macroNormal, next_rd));
            float pdfCos = NoL / PI;
            float pdfAlice = guideProb > 0.0 ? alice_guiding_pdf(next_rd, axis, kappa) : 0.0;
            float pdfMix = (1.0 - guideProb) * pdfCos + guideProb * pdfAlice;
            guideWeight = (NoL > 0.0 && pdfMix > 1e-8) ? (pdfCos / pdfMix) : 0.0;
            // ALICE probe: first-bounce throughput = MIS weight only (no BRDF).
            // The diffuse BSDF is deferred to the composite pass so the ALICE
            // encoding captures the material-independent incident light field.
            // guideWeight = pdfCos / pdfMix already provides unbiased MIS weighting.
            bsdf_weight = vec3(guideWeight);
            #endif
        } else {
            // --- Secondary bounces: stochastic mixture with correct MC estimator ---
            float rnd_lobe = getRandom();
            if (rnd_lobe < P_spec) {
                current_type = REFLECTION;
                // NDF-sampled reflection direction (same as first-bounce GGX sampling)
                next_rd = reflect(rd_i, microNormal);
                if (dot(next_rd, normal) > 0.0) {
                    vec3 wo = -rd_i;
                    vec3 wi = next_rd;
                    float NoV = abs(dot(normal, wo));
                    float NoL = dot(normal, wi);

                    if (NoV > 1e-6 && NoL > 1e-6) {
                        vec3 Hsum = wo + wi;
                        float Hlen2 = dot(Hsum, Hsum);
                        if (Hlen2 > 1e-12) {
                            vec3 H = Hsum * inversesqrt(Hlen2);
                            float NoH = abs(dot(normal, H));
                            float VoH = abs(dot(wo, H));

                            if (NoH > 1e-6 && VoH > 1e-6) {
                                float rough = max(surface.R.x, 1e-4);
                                float D_NoH = GGXpdf(NoH, 0.0, rough);
                                float D = D_NoH / NoH;
                                float G2 = GGX_G2_standard(NoV, NoL, rough);
                                vec3 Fh = reflectanceColor(surface.Cs, VoH).rgb;

                                // Full microfacet BRDF → f × NoL / pdf_ndf
                                // = Fh × S.x × G2 × VoH / (NoV × NoH)  (D cancels with pdf)
                                vec3 fSpec = Fh * surface.S.x * D * G2 / max(4.0 * NoV * NoL, 1e-8);
                                float pdfNDF = GGX_ndf_pdf(wo, wi, normal, rough);

                                // Stochastic-lobe MIS: divide by P_spec (lobe selection probability)
                                bsdf_weight = (pdfNDF > 1e-8)
                                    ? (fSpec * NoL / (pdfNDF * P_spec)) : vec3(0.0);
                            }
                        }
                    }
                }
            } else if (rnd_lobe < P_spec + P_refr) {
                current_type = REFRACTION;
                bool chooseTransmission = getRandom() < (1.0 - F);
                if (chooseTransmission) {
                    vec3 refract_dir = refract(rd_i, microNormal, rs);
                    if (dot(refract_dir, refract_dir) > 0.0) {
                        next_rd = refract_dir;
                        inverse_0 = !inverse_0;
                    } else {
                        next_rd = reflect(rd_i, microNormal);
                        current_type = REFLECTION;
                    }
                } else {
                    next_rd = reflect(rd_i, microNormal);
                    current_type = REFLECTION;
                }
                bsdf_weight = refrLobeWeight;
            } else {
                current_type = DIFFUSION;
                next_rd = DiffuseNormal(macroNormal, ro_o);
                bsdf_weight = diffLobeWeight;
            }
        }

        // --- 5. NEE: direct sunlight ---
        // At depth 0, guard with P_first>0: inactive forced lobes must not inject spurious
        // direct sun (never chosen in single-pass, weight=0 → path ends before NEE).
        // Reflection pass: P_spec + P_refr*F (both reflection mechanisms).
        // Refraction pass: P_refr (transmission only; TIR→weight=0 in pass itself).
        #if defined(FIRST_LOBE_REFLECTION)
        float P_first = P_spec + P_refr * F;
        #elif defined(FIRST_LOBE_REFRACTION)
        float P_first = P_refr;
        #else
        float P_first = P_diff;
        #endif
        if (current_type >= 0 && dot(macroNormal, lightDir) < 0.0 && !isDarkened
                && (depth > 0 || P_first > 0.0)) {
            vec3 sunL = sampleSunlight(
                    ro_o,
                    current_type == DIFFUSION ? macroNormal : normal,
                    surface.Cs,
                    surface.Cd,
                    rd_i,
                    surface.S,
                    surface.R,
                    lightDir,
                    lightDir.y > 0.0,
                    current_type,
                    macroNormal,
                    inverse_0
                );

            if (depth == 0) {
                #if defined(FIRST_LOBE_DIFFUSE)
                // sampleSunlight returns Cd * Li * dω_sun; strip Cd so the ALICE
                // probe encodes pure incident irradiance.  The full diffuse BRDF
                // (nonSpecColor × diffuseSelector) is applied in fog.glsl.
                L_direct_0 = sunL / max(surface.Cd, vec3(1e-3));
                #else
                L_direct_0 = sunL;
                #endif
            } else {
                vec3 nee_contrib = throughput * sunL;
                if (depth >= 2) {
                    float lum_nee = dot(nee_contrib, vec3(0.2126, 0.7152, 0.0722));
                    nee_contrib *= fireflyCap / max(lum_nee, fireflyCap);
                }
                L_indirect += nee_contrib;
            }
        }

        // --- 6. Record G-Buffer ---
        if (depth == 0) {
            first_p = ro_o;
            first_n = normal;
            first_macro_n = macroNormal;
            first_t = t;
            #if defined(FIRST_LOBE_REFLECTION)
            first_type = REFLECTION;
            #elif defined(FIRST_LOBE_REFRACTION)
            first_type = REFRACTION;
            #else
            first_type = DIFFUSION;
            #endif
            // G-Buffer material multipliers: use normal (texture detail normal) for stability.
            // microNormal (GGX-perturbed) is noisy per-frame; macroNormal (block face) loses detail.
            // normal captures the normal-map spatial variation while staying deterministic each frame.
            vec4 rC_stable = reflectanceColor(surface.Cs, abs(dot(rd_i, normal)));
            vec3 nonSpecColor_stable = surface.Cd * max(vec3(0.0), vec3(1.0) - rC_stable.rgb * surface.S.x * 0.5);
            {
                vec3 F0 = surface.Cs * surface.S.x;
                float NoV = clamp(abs(dot(rd_i, normal)), 0.0, 1.0);
                float rough = surface.R.x;
                vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
                vec4 c1 = vec4(1.0, 0.0425, 1.040, -0.040);
                vec4 r = rough * c0 + c1;
                float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
                vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
                first_specularAlbedo = max(F0 * AB.x + vec3(AB.y * surface.S.x), vec3(1e-5));
            }
            // Deferred diffuse BRDF for composite (fog.glsl):
            //   nonSpecColor × diffuseSelector = Cd × (1−rC.rgb×S.x) × (1−S.y)
            // The first bounce no longer carries BRDF in throughput; the full
            // diffuse reflectance factor is applied at composite time instead.
            // This keeps the ALICE encoding material-independent.
            first_diffuseAlbedo = nonSpecColor_stable * (1.0 - clamp(surface.S.y, 0.0, 1.0));
            first_transmissionAlbedo = nonSpecColor_stable * transmissionSelector;
            first_roughness = surface.R.x;
            first_rd_o = next_rd;
            first_n_i = n_i;
            first_n_o = current_type == REFRACTION ? n_o : n_i;
            first_micro_n = microNormal;
            first_emission_val = segment_emission;
            first_light_surf = surface.light;
            first_absorption = current_absorption;

            float reflectWeight = P_spec + P_refr * F;
            float refractWeight = P_refr * (1.0 - F);
            mixWeight = vec2(reflectWeight, refractWeight);
        }

        // --- 7. Update throughput ---
        throughput *= bsdf_weight;

        if (any(isnan(throughput))) {
            throughput = vec3(0.0);
            break;
        }

        if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.0) {
            break;
        }

        // --- 8. Russian Roulette ---
        if (depth >= 2) {
            float p_survive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (getRandom() > p_survive) break;
            throughput /= p_survive;
        }

        ro_i = ro_o + macroNormal * (current_type == REFRACTION ? -0.001 : 0.001);
        rd_i = next_rd;
    }

    // -----------------------------------------------------------------------------------
    // Final G-Buffer & denoising buffer write
    // -----------------------------------------------------------------------------------

    vec3 total_illumination = L_indirect + L_direct_0;
    if (any(isnan(total_illumination))) total_illumination = vec3(0.0);
    total_illumination = clamp(total_illumination, 0.0, 65504.0);

    // Shared G-Buffer (denoiseBuffer): only written by diffuse pass (ray0).
    // All three passes hit the same first surface → fields are fully deterministic.
    #if defined(FIRST_LOBE_DIFFUSE)
    denoiseBuffer.data[idx].reflectWeight = mixWeight.x;
    denoiseBuffer.data[idx].refractWeight = mixWeight.y;
    denoiseBuffer.data[idx].specularAlbedo = first_specularAlbedo;
    denoiseBuffer.data[idx].diffuseAlbedo = first_diffuseAlbedo;
    denoiseBuffer.data[idx].transmissionAlbedo = first_transmissionAlbedo;
    denoiseBuffer.data[idx].distance = first_t;
    denoiseBuffer.data[idx].light = first_light_surf;
    denoiseBuffer.data[idx].macroNormal = first_macro_n;
    denoiseBuffer.data[idx].illuminationType = first_type;
    denoiseBuffer.data[idx].roughness = first_roughness;
    denoiseBuffer.data[idx].pathRoughness = first_roughness; // default: surface roughness; overwritten by refraction pass with accumulated path roughness
    denoiseBuffer.data[idx].absorption = first_absorption;
    denoiseBuffer.data[idx].rd = first_rd_i;
    #endif

    // Each pass writes only its own illumination buffer.
    // Sky pixels (hit_sky_first / first_t < -0.5): color=0 + default geometry,
    // matching single-pass behaviour (fog sky branch, 100/101/102 reset on distance<-0.5).
    vec3 pos_rel = first_p - ro;

    #if defined(FIRST_LOBE_DIFFUSE)
    {
        // --- Diffuse pass: write diffuseIlluminationBuffer (Alice encoding + geometry) ---
        diffuseIlluminationBuffer.data[idx].rt_aliceY_xy = 0.0;
        diffuseIlluminationBuffer.data[idx].rt_aliceY_zw = 0.0;
        diffuseIlluminationBuffer.data[idx].rt_CoCg = 0.0;
        if (!hit_sky_first && first_t > -0.5) {
            // L_indirect: first-bounce throughput = guideWeight only (no BRDF).
            //   Pure incident radiance — secondary bounces carry their surface
            //   BRDFs, which is physically correct for interreflection colouring.
            //
            // L_direct_0: already stripped of Cd at the NEE site (depth==0,
            //   FIRST_LOBE_DIFFUSE).  Pure incident irradiance from the sun.
            //
            // Both are encoded directly — no albedo demodulation needed.
            // The full diffuse BRDF (nonSpecColor × diffuseSelector) is baked
            // into denoiseBuffer.diffuseAlbedo and applied in fog.glsl.
            AliceEncoding indAlice = irradiance_to_alice(L_indirect, first_rd_o);
            AliceEncoding dirAlice = irradiance_to_alice(L_direct_0, -lightDir);
            indAlice.CoCg += dirAlice.CoCg;
            indAlice.aliceY += dirAlice.aliceY;
            diffuseIlluminationBuffer.data[idx].rt_aliceY_xy = uintBitsToFloat(packHalf2x16(indAlice.aliceY.xy));
            diffuseIlluminationBuffer.data[idx].rt_aliceY_zw = uintBitsToFloat(packHalf2x16(indAlice.aliceY.zw));
            diffuseIlluminationBuffer.data[idx].rt_CoCg = uintBitsToFloat(packHalf2x16(indAlice.CoCg));
        }
        diffuseIlluminationBuffer.data[idx].px = pos_rel.x;
        diffuseIlluminationBuffer.data[idx].py = pos_rel.y;
        diffuseIlluminationBuffer.data[idx].pz = pos_rel.z;
        diffuseIlluminationBuffer.data[idx].oct_n = encodeNormal(first_macro_n);
        diffuseIlluminationBuffer.data[idx].oct_n2 = encodeNormal(faceforward(first_n, first_n, -first_macro_n));
    }
    #elif defined(FIRST_LOBE_REFLECTION)
    {
        // --- Reflection pass: write reflectIlluminationBuffer ---
        // virtualProjDist: one extra raycast along the GGX dominant direction
        vec3 refl_R = first_rd_i;
        float refl_vprojdist = 0.0;
        vec3 refl_color = vec3(0.0);
        if (!hit_sky_first && first_t > -0.5) {
            vec3 r_rd, r_ro;
            vec3 r_rd_i = GetSpecularDominantDirection(first_n, first_rd_i, first_roughness);
            float t_refl = raycast(first_p + first_macro_n * 0.00025, r_rd_i, r_ro, r_rd, false, 0, 1u);
            refl_R = r_rd_i;
            refl_vprojdist = (t_refl > -0.5) ? t_refl : VPROJDIST_SKY;
            refl_color = clamp(total_illumination / max(first_specularAlbedo, vec3(1e-6)), 0.0, 200.0 * div_avgExposure);
        }
        reflectIlluminationBuffer.data[idx].px = pos_rel.x;
        reflectIlluminationBuffer.data[idx].py = pos_rel.y;
        reflectIlluminationBuffer.data[idx].pz = pos_rel.z;
        reflectIlluminationBuffer.data[idx].oct_dir = encodeNormal(refl_R);
        reflectIlluminationBuffer.data[idx].virtualProjDist = refl_vprojdist;
        reflectIlluminationBuffer.data[idx].color_rg = pack2HalfClamped(refl_color.r, refl_color.g);
        reflectIlluminationBuffer.data[idx].color_b = pack2HalfClamped(refl_color.b, 0.0);
    }
    #else
    {
        // --- Refraction pass: write refractIlluminationBuffer (transmission only; TIR → reflection pass) ---
        // PSR (Primary Surface Replacement):
        //   oct_dir encodes the refracted direction T at the first surface (for edge-stopping lobe similarity,
        //   analogous to the reflection pass storing the reflected direction R).
        //   The view direction V is reconstructible from pos (camera-relative: pos = P_surf - C → V = normalize(pos)).
        //   Virtual position: P_virtual = P_surf + V × virtualProjDist, where virtualProjDist = Σ(t_i × n_camera / n_i).
        //
        //   If the first surface is too rough (psrEnabled=false), first_t2_ior_adjusted=0
        //   → virtualProjDist=0 → P_virtual = P_surf → falls back to first-surface temporal accumulation.
        vec3 refr_R = first_refr_dir;           // refracted direction T (TIR→reflect; for edge-stopping)
        float refr_vprojdist = 0.0;
        vec3 refr_color = vec3(0.0);
        if (!hit_sky_first && first_t > -0.5) {
            if (mixWeight.y > 0.001) {
                refr_vprojdist = first_t2_ior_adjusted; // IOR-adjusted Σ(t_i / n_i), 0 if PSR disabled
            }
            refr_color = clamp(total_illumination / max(first_transmissionAlbedo, vec3(1e-6)), 0.0, 200.0 * div_avgExposure);
        }
        refractIlluminationBuffer.data[idx].px = pos_rel.x;
        refractIlluminationBuffer.data[idx].py = pos_rel.y;
        refractIlluminationBuffer.data[idx].pz = pos_rel.z;
        refractIlluminationBuffer.data[idx].oct_dir = encodeNormal(refr_R);
        refractIlluminationBuffer.data[idx].virtualProjDist = refr_vprojdist;
        refractIlluminationBuffer.data[idx].color_rg = pack2HalfClamped(refr_color.r, refr_color.g);
        refractIlluminationBuffer.data[idx].color_b = pack2HalfClamped(refr_color.b, 0.0);
        // Write accumulated path roughness for the denoiser (overwrites the default set by diffuse pass)
        denoiseBuffer.data[idx].pathRoughness = first_pathRoughness;
    }
    #endif

    #if defined(FIRST_LOBE_DIFFUSE)
    denoiseBuffer.data[idx].emission = first_emission_val;
    #endif
}
