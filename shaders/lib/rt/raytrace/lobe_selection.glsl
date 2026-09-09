#ifndef DIRT_RT_LIB_RT_RAYTRACE_LOBE_SELECTION_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_LOBE_SELECTION_GLSL

// Continuation proposal probabilities, independent of the sampled microfacet.

// ===========================================================================
// BSDF Lobe Probabilities
// ===========================================================================

LobeProbs computeLobeProbs(material surf, vec3 rd_i, vec3 macroNormal, float rs) {
    LobeProbs p;
    // Lobe selection must be independent of the subsequently sampled GGX
    // half-vector. This gives the continuation strategy a well-defined PDF
    // that can be evaluated for an arbitrary NEE direction.
    float macroF = clamp(fresnel(-rd_i, macroNormal, rs), 0.0, 1.0);
    vec4 rC = reflectanceColor(surf.Cs, dot(rd_i, macroNormal));

    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float diffuseSelector = 1.0 - transmissionSelector;
    float specularSelector = clamp(surf.S.x, 0.0, 1.0);

    vec3 interfaceF = mix(
            rC.rgb * specularSelector,
            vec3(macroF),
            transmissionSelector);
    // Macro-normal Fresnel is not the support of a rough microfacet lobe.  In
    // particular, F0 == 0 and NoV ~= 1 make it exactly zero even though tilted
    // GGX facets have VoH < 1 and therefore non-zero Schlick Fresnel.
    //
    // Build a half-vector-independent proposal importance from a fixed
    // cosine-weighted angular kernel.  Its Schlick moment is analytic:
    //   E[(1 - mu)^5] = integral_0^1 2 mu (1 - mu)^5 dmu = 1 / 21.
    // GGX alpha blends from the macro-normal value to that broad-kernel mean.
    // This value only selects a lobe; the sampled BRDF and its PDF below still
    // determine the Monte Carlo weight.
    float etaDenominator = max(abs(1.0 + rs), 1e-8);
    float dielectricF0 = (1.0 - rs) / etaDenominator;
    dielectricF0 *= dielectricF0;
    float dielectricF90 = (rs == 1.0) ? 0.0 : 1.0;
    vec3 interfaceF0 = mix(
            clamp(surf.Cs, vec3(0.0), vec3(1.0)) * specularSelector,
            vec3(dielectricF0), transmissionSelector);
    vec3 interfaceF90 = mix(
            vec3(specularSelector), vec3(dielectricF90),
            transmissionSelector);
    const float SCHLICK_COSINE_KERNEL_MOMENT = 1.0 / 21.0;
    vec3 cosineKernelMeanF = interfaceF0
            + (interfaceF90 - interfaceF0) * SCHLICK_COSINE_KERNEL_MOMENT;
    float proposalWidth = isDeltaSpecular(surf.R.x)
            ? 0.0 : clamp(surf.R.x, 0.0, 1.0);
    // A rough interface can transmit through tilted facets even when the
    // macro normal is in total internal reflection. Blend both ways: max(F,
    // meanF) would pin reflection to one and erase that transmission support.
    vec3 roughInterfaceImportance = mix(interfaceF,
            cosineKernelMeanF, proposalWidth);
    // Probabilities choose a non-zero transport lobe; they are not Fresnel
    // absorption probabilities. In particular, a metal has Cd == 0 and must
    // never spend samples on a zero-valued diffuse branch.
    float interfaceEnergy = clamp(luma(roughInterfaceImportance), 0.0, 1.0);
    float diffuseBaseEnergy = max(luma(max(surf.Cd, vec3(0.0))), 0.0);
    vec3 transmissionColor = evaluateTransmissionAlbedo(surf);
    float transmissionEnergy = max(luma(max(transmissionColor,
        vec3(0.0))), 0.0);
    float remainingEnergy = max(1.0 - interfaceEnergy, 0.0);
    float specImportance = interfaceEnergy;
    float refrImportance = remainingEnergy * transmissionSelector
        * transmissionEnergy;
    float diffImportance = remainingEnergy * diffuseSelector
        * diffuseBaseEnergy;
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
    p.diffWeight = nonSpecColor * diffuseSelector / max(p.P_diff, 1e-5);

    return p;
}

#endif // DIRT_RT_LIB_RT_RAYTRACE_LOBE_SELECTION_GLSL
