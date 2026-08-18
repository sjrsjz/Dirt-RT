// #version 460 core is declared by each enclosing rayN.rgen entry. ray0 owns
// primary visibility, ray1..ray3 own the first lobes, and ray4/ray5 resolve
// and commit the low-history biased ReSTIR GI path-guiding prewarm. This orchestration
// file is never compiled alone.
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
#include "/lib/rt/volume_extinction.glsl"
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
#if EON_ENABLED
#include "/lib/lighting/eon.glsl"
#endif

// ray0 builds the primary-surface cache. Continuation entries define exactly
// one of FIRST_LOBE_DIFFUSE / FIRST_LOBE_REFLECTION / FIRST_LOBE_REFRACTION.
#if !defined(PRIMARY_GBUFFER_PASS) && !defined(FIRST_LOBE_DIFFUSE) && !defined(FIRST_LOBE_REFLECTION) && !defined(FIRST_LOBE_REFRACTION)
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#endif

// ray1 creates one fresh proposal per pixel. ray4 resamples those proposals and
// ray5 only commits the prewarm result, so only the original diffuse pass is
// allowed to write proposal metadata.
#if defined(FIRST_LOBE_DIFFUSE) && RESTIR_GI_ENABLED && EON_ENABLED \
        && !defined(RESTIR_GI_RESOLVE_PASS) \
        && !defined(RESTIR_GI_FINAL_PASS)
#define RESTIR_GI_INITIAL_PASS
#endif

// ---------------------------------------------------------------------------
// PSR (Primary Surface Replacement) — refraction virtual-image reprojection
// ---------------------------------------------------------------------------
// Rough paths still trace an endpoint, but use the radiance cache instead of
// screen-space diffuse reuse.
const float PSR_ROUGHNESS_THRESHOLD = 0.15;
const float PSR_PATH_ROUGHNESS_THRESHOLD = 0.8;
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
#if defined(RESTIR_GI_RESOLVE_PASS)
void ResolveFirstBounceRestirGI(uvec2 coord, vec3 ro);
#elif defined(RESTIR_GI_FINAL_PASS)
void FinalizeFirstBounceRestirGI(uvec2 coord);
#endif
#endif

bool isDarkened = false;
float rtCurrentConeWidth = 0.0;
float rtCurrentConeSpread = 0.0;

// Shared payload and feature modules. Keep this order: later modules use
// types and helpers declared by the modules before them.
Payload tmp_Payload;

#if !defined(RESTIR_GI_RESOLVE_PASS) && !defined(RESTIR_GI_FINAL_PASS)
#include "/lib/rt/raytrace/scene.glsl"
#include "/lib/rt/raytrace/transport.glsl"
#include "/lib/rt/raytrace/bounces.glsl"
#include "/lib/rt/raytrace/lighting.glsl"
#include "/lib/rt/raytrace/gbuffer_io.glsl"
#include "/lib/rt/raytrace/primary_pass.glsl"
#include "/lib/rt/raytrace/path_trace.glsl"
#endif
#include "/lib/rt/raytrace/restir_gi.glsl"

#if !defined(RADIANCE_CACHE_TRACE)

#if !defined(PRIMARY_GBUFFER_PASS)
// Continuation passes only need the primary-distance word to reject sky.
// Keep this before camera-ray reconstruction, RNG setup, setSkyVars(), and
// material decoding: all of those are dead work when ray0 reported a miss.
bool clearSkyContinuation(uvec2 pixel) {
    float primaryDistance = readPrimaryDistance(pixel);
    if (primaryDistance >= -0.5) return false;

    #if defined(FIRST_LOBE_DIFFUSE)
    diffuseBuffer.data[addr(DIF_N_LIGHT, pixel)] = uvec4(0u);
    diffuseBuffer.data[addr(DIF_N_SURFACE, pixel)] = uvec4(0u);
    clearRestirGIScratch(pixel);
    #elif defined(FIRST_LOBE_REFLECTION)
    reflectBuffer.data[addr(SPEC_N_LIGHT, pixel)] = uvec4(0u);
    #else
    refractBuffer.data[addr(REFR_N_ENDPOINT, pixel)] = uvec4(0u);
    refractBuffer.data[addr(REFR_N_SURFACE, pixel)] = uvec4(0u);
    refractBuffer.data[addr(REFR_N_META, pixel)] = uvec4(0u);
    refractBuffer.data[addr(REFR_N_TRANSPORT, pixel)] = uvec4(0u);
    #endif
    return true;
}
#endif

void main() {
    uvec2 pixel = uvec2(gl_LaunchIDEXT.xy);
    #if !defined(PRIMARY_GBUFFER_PASS)
    if (clearSkyContinuation(pixel)) return;
    #endif

    #if defined(RESTIR_GI_FINAL_PASS)
    // Commit is buffer-only; avoid camera-ray reconstruction, RNG and sky
    // setup in this full-screen RT dispatch.
    FinalizeFirstBounceRestirGI(pixel);
    #elif defined(RESTIR_GI_RESOLVE_PASS)
    // Resolve needs the camera origin for world-space endpoint shifting, but
    // no path-tracing RNG, material modules or sky state.
    ResolveFirstBounceRestirGI(pixel, cam.viewInverse[3].xyz);
    #else
    vec2 px = vec2(gl_LaunchIDEXT.xy);
    vec2 taaJitter = vec2(0.0);
    #if defined(PRIMARY_GBUFFER_PASS)
    taaJitter = rtTaaJitter(cam.frameId);
    #endif
    vec2 p = (px + taaJitter) / vec2(gl_LaunchSizeEXT.xy);

    vec3 origin = cam.viewInverse[3].xyz;
    vec3 target = mix(mix(cam.corners[0], cam.corners[2], p.y), mix(cam.corners[1], cam.corners[3], p.y), p.x);
    vec3 direction = normalize((cam.viewInverse * vec4(target.xyz, 0.0)).xyz);
    vec2 coneResolution = max(vec2(resolution_global),
        vec2(gl_LaunchSizeEXT.xy));
    rtCurrentConeWidth = 0.0;
    rtCurrentConeSpread = rtPixelConeSpread(cam.corners[0],
        cam.corners[1], cam.corners[2], coneResolution);

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
    #elif defined(FIRST_LOBE_REFRACTION)
    TraceRefractionPSR(pixel, origin);
    #else
    Trace(pixel, origin, direction, -lightDir_global);
    #endif

    // Per-frame-once global state: only primary pass pixel (0,0). Updating the
    // previous matrices from any continuation pass would break reprojection.
    #if defined(PRIMARY_GBUFFER_PASS)
    if (gl_LaunchIDEXT.xy == vec2(0)) {
        world_type_global = int(cam.world_type);
        frame_id = int(cam.frameId);
        eye_medium_global = cam.flags & 3u;
        camPos = origin;
        camY_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[2]), 0)).xyz;
        camX_global = (cam.viewInverse * vec4(normalize(cam.corners[0] - cam.corners[1]), 0)).xyz;

        // Save the already-composed transform for next-frame reprojection.
        rtPrevViewProjection = rtViewProjection;

        // ModelView: pure rotation (transpose of viewInverse), no translation
        rtModelView = mat4(transpose(mat3(cam.viewInverse)));

        // The projection carries the same sub-pixel phase as the traced ray,
        // so distance-only G-buffer reconstruction remains exact.
        float zNear = -cam.corners[0].z;
        float w = cam.corners[1].x - cam.corners[0].x;
        float h = cam.corners[2].y - cam.corners[0].y;
        vec2 jitterNdc = 2.0 * taaJitter / vec2(gl_LaunchSizeEXT.xy);
        float farD = 2048.0;
        mat4 projection = mat4(0.0);
        projection[0][0] = (2.0 * zNear) / w;
        projection[1][1] = (2.0 * zNear) / h;
        projection[2][0] = (cam.corners[1].x + cam.corners[0].x) / w + jitterNdc.x;
        projection[2][1] = (cam.corners[2].y + cam.corners[0].y) / h + jitterNdc.y;
        projection[2][2] = -(farD + zNear) / (farD - zNear);
        projection[2][3] = -1.0;
        projection[3][2] = -(2.0 * farD * zNear) / (farD - zNear);
        rtViewProjection = projection * rtModelView;
        rtInverseViewProjection = inverse(rtViewProjection);
        rtProjectionParams = vec4(
            projection[0][0], projection[1][1],
            projection[2][0], projection[2][1]);
    }
    #endif
    #endif
}
#endif
