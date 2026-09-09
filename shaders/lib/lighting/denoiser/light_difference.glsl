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
struct MaxEntLightMetric {
    vec3 v;
    float w;
    float q;
};

// Prepare invariant endpoint terms once when comparing a center to many taps.
MaxEntLightMetric maxentPrepareLightMetric(vec4 moment) {
    float w = max(moment.w, 0.0);
    vec3 v = moment.xyz;
    float v2 = dot(v, v);
    // FP16 rounding may place a stored moment just outside the PSD cone.
    if (v2 > w * w && v2 > 0.0) v *= w * inversesqrt(v2);
    return MaxEntLightMetric(v, w, sqrt(max(w * w - dot(v, v), 0.0)));
}

float maxentLightSampleDistanceSq(MaxEntLightMetric a, MaxEntLightMetric b) {
    float wA = a.w, wB = b.w;
    vec3 vA = a.v, vB = b.v;
    float qA = a.q, qB = b.q;
    float rootAffinity = sqrt(max(0.5 * (wA * wB
        + dot(vA, vB) + qA * qB), 0.0));

    vec3 deltaV = vA - vB;
    float deltaQ = qA - qB;
    float numerator = dot(deltaV, deltaV) + deltaQ * deltaQ;
    float denominator = wA + wB + 2.0 * rootAffinity;
    return denominator > 0.0 ? max(numerator / denominator, 0.0) : 0.0;
}

float maxentLightSampleDistanceSq(vec4 sampleA, vec4 sampleB) {
    return maxentLightSampleDistanceSq(maxentPrepareLightMetric(sampleA),
        maxentPrepareLightMetric(sampleB));
}
#endif // MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL
