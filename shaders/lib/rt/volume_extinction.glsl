#ifndef VOLUME_EXTINCTION_GLSL
#define VOLUME_EXTINCTION_GLSL

// ===========================================================================
// Shared volume extinction — used by both rahit and rchit.
// Applies Beer-Lambert absorption through a homogeneous segment of a block.
//
// Water: physically-based absorption coefficients (real measured values, m⁻¹)
//         R: 0.14426950  G: 0.04328085  B: 0.05770780
// Other: LabPBR dielectric extinction
//         translucency = albedo.a  (0=opaque, 1=fully transmitting)
//         T(d) = mix(0, albedo^d, translucency)
// ===========================================================================

// Pure-water absorption coefficients (m⁻¹) at RGB wavelengths
const vec3 WATER_ABSORPTION = vec3(0.14426950, 0.04328085, 0.05770780);

vec3 applyVolumeExtinction(vec3 shadowTrans, float segDist, vec4 texColor, int blockID) {
    if (blockID == BLOCK_WATER) {
        return shadowTrans * exp2(-segDist * WATER_ABSORPTION);
    }
    float translucency = texColor.a;
    vec3 beersLambert = pow(max(texColor.rgb, 0.005), vec3(segDist));
    return shadowTrans * mix(vec3(1.0), beersLambert, translucency);
}

#endif // VOLUME_EXTINCTION_GLSL
