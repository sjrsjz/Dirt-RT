#ifndef MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL
#define MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL

// Squared Bures--Wasserstein distance on the 2x2 PSD embedding of the linear
// light-field moment m=(v,w)=(E[R u],E[R]):
//
//     M(m) = (w I + v.sigma) / 2.
//
// The realizability cone |v|<=w is exactly the PSD cone of M. For a 2x2 state,
// q=sqrt(w^2-|v|^2) reduces the matrix-square-root definition to elementary
// operations. The rationalized form below is algebraically identical to
// wA+wB-2*rootAffinity and avoids subtracting nearly equal HDR values.
float maxentLightSampleDistanceSq(vec4 sampleA, vec4 sampleB)
{
    float wA = max(sampleA.w, 0.0);
    float wB = max(sampleB.w, 0.0);

    vec3 vA = sampleA.xyz;
    vec3 vB = sampleB.xyz;

    // Enforce realizability |v| <= w against FP16 / roundoff violations.
    float vA2 = dot(vA, vA);
    float vB2 = dot(vB, vB);

    if (vA2 > wA * wA && vA2 > 0.0)
        vA *= wA * inversesqrt(vA2);

    if (vB2 > wB * wB && vB2 > 0.0)
        vB *= wB * inversesqrt(vB2);

    float qA = sqrt(max(wA * wA - dot(vA, vA), 0.0));
    float qB = sqrt(max(wB * wB - dot(vB, vB), 0.0));
    float rootAffinity = sqrt(max(0.5 * (wA * wB
        + dot(vA, vB) + qA * qB), 0.0));

    vec3 deltaV = vA - vB;
    float deltaQ = qA - qB;
    float numerator = dot(deltaV, deltaV) + deltaQ * deltaQ;
    float denominator = wA + wB + 2.0 * rootAffinity;
    return denominator > 0.0 ? max(numerator / denominator, 0.0) : 0.0;
}
#endif // MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL
