// #version 460 core is declared by each enclosing rayN.rgen entry. ray0 owns
// primary visibility and ray1..ray3 own the first lobes.
// This orchestration file is never compiled alone.
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
#include "/lib/rt/camera_state.glsl"
#include "/lib/lighting/denoiser/diffuse_reprojection.glsl"
#include "/lib/buffers/radiance_cache.glsl"
#include "/lib/pbr/material.glsl"
#include "/lib/common.glsl"
#if EON_ENABLED
#define EON_RT_CUSTOM_TEXTURES
#include "/lib/lighting/eon.glsl"
#undef EON_RT_CUSTOM_TEXTURES
#endif

// ray0 builds the primary-surface cache. Continuation entries define exactly
// one of FIRST_LOBE_DIFFUSE / FIRST_LOBE_REFLECTION / FIRST_LOBE_REFRACTION.
#if !defined(PRIMARY_GBUFFER_PASS) && !defined(FIRST_LOBE_DIFFUSE) && !defined(FIRST_LOBE_REFLECTION) && !defined(FIRST_LOBE_REFRACTION)
#define FIRST_LOBE_DIFFUSE
#define FIRST_LOBE_VAL 2
#endif


// ---------------------------------------------------------------------------
// PSR (Primary Surface Replacement) — refraction virtual-image reprojection
// ---------------------------------------------------------------------------
// Rough paths still trace an endpoint, but use the radiance cache instead of
// screen-space diffuse reuse.
const float PSR_ROUGHNESS_THRESHOLD = 0.15;
const float PSR_PATH_ROUGHNESS_THRESHOLD = 0.8;
const int MAX_REFRACTIVE_BOUNCES = 4; // max refractive surfaces to trace through before stopping

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
void TracePrimaryGBuffer(uvec2 coord, vec3 ro, vec3 rd,
    mat4 currentViewProjection);
#endif

bool isDarkened = false;
float rtCurrentConeWidth = 0.0;
float rtCurrentConeSpread = 0.0;
mat4 rtCurrentModelViewLocal;
vec4 rtCurrentProjectionParamsLocal;
mat4 rtCurrentViewProjectionLocal;

// Shared payload and feature modules. Keep this order: later modules use
// types and helpers declared by the modules before them.
Payload tmp_Payload;

#include "/lib/rt/raytrace/scene.glsl"
#include "/lib/rt/raytrace/transport.glsl"
#include "/lib/rt/raytrace/bounces.glsl"
#include "/lib/rt/raytrace/lighting.glsl"
#include "/lib/rt/raytrace/gbuffer_io.glsl"
#include "/lib/rt/raytrace/primary_pass.glsl"
#include "/lib/rt/raytrace/path_trace.glsl"

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
    writeDiffuseSurfaceInvalid(pixel);
    invalidatePreparedDiffuseHistory(pixel, cam.frameId);
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

    vec2 px = vec2(gl_LaunchIDEXT.xy);
    vec2 taaJitter = rtTaaJitter(cam.frameId);
    #if defined(PRIMARY_GBUFFER_PASS) || defined(FIRST_LOBE_DIFFUSE)
    RtCameraMatrices currentCamera = buildRtCameraMatrices(
        vec2(gl_LaunchSizeEXT.xy));
    rtCurrentModelViewLocal = currentCamera.modelView;
    rtCurrentProjectionParamsLocal = currentCamera.projectionParams;
    rtCurrentViewProjectionLocal = currentCamera.viewProjection;
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
    int currentWorldType = int(cam.world_type);
    #if END_SKYBOX == 1
    isDarkened = currentWorldType != WORLD_OVERWORLD
        && currentWorldType != WORLD_THE_NETHER;
    #else
    isDarkened = currentWorldType != WORLD_OVERWORLD
        && currentWorldType != WORLD_THE_END
        && currentWorldType != WORLD_THE_NETHER;
    #endif

    wseed = floatBitsToUint(hash12(px + fract(cam.frameId * vec2(0.6180339887498949, 0.4142135623730950))));

    vec3 seed0 = vec3(randcore4(), randcore4(), randcore4());
    wseed3.x = floatBitsToUint(seed0.x);
    wseed3.y = floatBitsToUint(seed0.y);
    wseed3.z = floatBitsToUint(seed0.z);

    setSkyVars();
    #if defined(PRIMARY_GBUFFER_PASS)
    TracePrimaryGBuffer(pixel, origin, direction,
        rtCurrentViewProjectionLocal);
    #elif defined(FIRST_LOBE_REFRACTION)
    TraceRefractionPSR(pixel, origin, direction);
    #else
    Trace(pixel, origin, direction, -lightDir_global);
    #endif

}
#endif
