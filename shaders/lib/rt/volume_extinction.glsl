#ifndef VOLUME_EXTINCTION_GLSL
#define VOLUME_EXTINCTION_GLSL

// ===========================================================================
// Shared homogeneous-medium extinction.
// Applies Beer-Lambert absorption through a homogeneous segment of a block.
//
// Water: physically-based absorption coefficients (real measured values, m⁻¹)
//         R: 0.14426950  G: 0.04328085  B: 0.05770780
// Glass/ice use the authored transmission tint and alpha as a per-block
// transmittance. Their interfaces remain color-neutral and only contribute
// Fresnel, so tint is applied exactly once and scales with travelled distance.
// ===========================================================================

// Pure-water absorption coefficients (m⁻¹) at RGB wavelengths
const vec3 WATER_ABSORPTION = vec3(0.14426950, 0.04328085, 0.05770780);

vec3 waterVolumeTransmittance(float segmentDistance) {
    return exp2(-max(segmentDistance, 0.0) * WATER_ABSORPTION
        * 1.4426950408889634);
}

vec3 tintedVolumeTransmittance(float segmentDistance, vec3 linearTint,
        float extinctionWeight) {
    vec3 transmittancePerBlock = mix(vec3(1.0),
        clamp(linearTint, vec3(0.005), vec3(1.0)),
        clamp(extinctionWeight, 0.0, 1.0));
    return pow(transmittancePerBlock,
        vec3(max(segmentDistance, 0.0)));
}

vec3 mediumVolumeTransmittance(float segmentDistance, int blockID,
        vec3 linearTint, float extinctionWeight) {
    if (blockID == BLOCK_WATER) {
        return waterVolumeTransmittance(segmentDistance);
    }
    if (blockID == BLOCK_GLASS || blockID == BLOCK_ICE) {
        return tintedVolumeTransmittance(segmentDistance, linearTint,
            extinctionWeight);
    }
    return vec3(1.0);
}

vec3 applyVolumeExtinction(vec3 shadowTrans, float segDist, vec4 texColor,
        int blockID) {
    vec3 linearTint = pow(max(texColor.rgb, vec3(0.0)), vec3(2.2));
    return shadowTrans * mediumVolumeTransmittance(segDist, blockID,
        linearTint, texColor.a);
}

#endif // VOLUME_EXTINCTION_GLSL
