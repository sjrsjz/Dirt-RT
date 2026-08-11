#version 430 core

// Clear both overlay color and its linear-depth marker. Keeping either plane
// from a previous frame can resurrect stale raster geometry during tonemap.

/* RENDERTARGETS: 7,8 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 depthMarker;

void main() {
    fragColor = vec4(0.0);
    depthMarker = vec4(0.0);
}
