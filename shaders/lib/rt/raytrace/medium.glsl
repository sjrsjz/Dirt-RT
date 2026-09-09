#ifndef DIRT_RT_LIB_RT_RAYTRACE_MEDIUM_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_MEDIUM_GLSL

// Segment attenuation and emission in the currently occupied medium.

// ===========================================================================
// Volumetric Medium
// ===========================================================================

MediumResult evalMedium(float t, vec3 rd_i, float ro_i_y, bool inside,
    int mediumBlockID, vec3 mediumTint, float mediumExtinctionWeight,
    vec4 fogColor, vec3 globalEmission) {
    MediumResult m;
    if (inside) {
        if (mediumBlockID == BLOCK_WATER
                || mediumBlockID == BLOCK_GLASS
                || mediumBlockID == BLOCK_ICE) {
            m.absorption = mediumVolumeTransmittance(t, mediumBlockID,
                mediumTint, mediumExtinctionWeight);
            m.emission = vec3(0.0);
        } else {
            // Non-block camera media (currently lava) retain their emissive
            // fog model.
            m.absorption = exp2(-t * fogColor.yzw * LOG2_E);
            m.emission = (1.0 - m.absorption)
                / (fogColor.yzw + 1e-5) * globalEmission;
        }
    } else {
        m.absorption = exp2(-max(b_Q * (b_P.x - ro_i_y) * t
                        - 0.5 * b_Q * t * t * rd_i.y, 0.0) * LOG2_E);
        m.emission = vec3(0.0);
    }
    return m;
}

#endif // DIRT_RT_LIB_RT_RAYTRACE_MEDIUM_GLSL
