#ifndef MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL
#define MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL

// Metric of the encoded linear light field m=(E[R u], E[R]). This policy is
// intentionally separate from variance propagation and temporal statistics.
// Changing the perceptual light difference must not change either of them.
float maxentLightSampleDistanceSq(vec4 sampleA, vec4 sampleB) {
    // vec4 deltaAB = sampleA - sampleB;
    // return dot(deltaAB, deltaAB);
    vec3 rhoA = sampleA.xyz / max(sampleA.w, 1e-20);
    vec3 rhoB = sampleB.xyz / max(sampleB.w, 1e-20);
    vec3 deltaRho = rhoA - rhoB;
    float deltaRhoSq = dot(deltaRho, deltaRho);
    float deltaR = sampleA.w - sampleB.w;
    return sampleA.w * sampleB.w * deltaRhoSq + deltaR * deltaR;
}

#endif // MAXENT_DENOISER_LIGHT_DIFFERENCE_GLSL
