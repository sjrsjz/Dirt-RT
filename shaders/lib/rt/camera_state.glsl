#ifndef RT_CAMERA_STATE_GLSL
#define RT_CAMERA_STATE_GLSL

#include "/lib/buffers/frame_data.glsl"
#include "/lib/rt/mipmap.glsl"

layout(std430, binding = 0) uniform CameraInfo {
    vec3 corners[4];
    mat4 viewInverse;
    uint frameId;
    uint flags;
    uint world_type;
} cam;

struct RtCameraMatrices {
    mat4 modelView;
    mat4 projection;
    mat4 viewProjection;
    vec4 projectionParams;
    vec2 taaJitter;
};

RtCameraMatrices buildRtCameraMatrices(vec2 launchResolution) {
    RtCameraMatrices result;
    result.taaJitter = rtTaaJitter(cam.frameId);
    result.modelView = mat4(transpose(mat3(cam.viewInverse)));
    float zNear = -cam.corners[0].z;
    float width = cam.corners[1].x - cam.corners[0].x;
    float height = cam.corners[2].y - cam.corners[0].y;
    vec2 jitterNdc = 2.0 * result.taaJitter
        / max(launchResolution, vec2(1.0));
    float zFar = 2048.0;
    result.projection = mat4(0.0);
    result.projection[0][0] = (2.0 * zNear) / width;
    result.projection[1][1] = (2.0 * zNear) / height;
    result.projection[2][0] =
        (cam.corners[1].x + cam.corners[0].x) / width + jitterNdc.x;
    result.projection[2][1] =
        (cam.corners[2].y + cam.corners[0].y) / height + jitterNdc.y;
    result.projection[2][2] = -(zFar + zNear) / (zFar - zNear);
    result.projection[2][3] = -1.0;
    result.projection[3][2] =
        -(2.0 * zFar * zNear) / (zFar - zNear);
    result.viewProjection = result.projection * result.modelView;
    result.projectionParams = vec4(
        result.projection[0][0], result.projection[1][1],
        result.projection[2][0], result.projection[2][1]);
    return result;
}

// Called only by ray4's sole active invocation, after every full-screen path
// pass has stopped reading the previous state and before post processing reads
// the current state.
void publishRtCameraState(RtCameraMatrices current) {
    rtPrevViewProjection = rtViewProjection;
    rtModelView = current.modelView;
    rtViewProjection = current.viewProjection;
    rtInverseViewProjection = inverse(current.viewProjection);
    rtProjectionParams = current.projectionParams;
    world_type_global = int(cam.world_type);
    frame_id = int(cam.frameId);
    eye_medium_global = cam.flags & 3u;
    camPos = cam.viewInverse[3].xyz;
    camY_global = (cam.viewInverse * vec4(normalize(
        cam.corners[0] - cam.corners[2]), 0.0)).xyz;
    camX_global = (cam.viewInverse * vec4(normalize(
        cam.corners[0] - cam.corners[1]), 0.0)).xyz;
}

#endif
