// MIT License
//
// Copyright (c) 2026 sjrsjz (https://github.com/sjrsjz)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MAXENT_GLSL
#define MAXENT_GLSL

#include "/lib/constants.glsl"
#include "/lib/math/sampling.glsl"

// Linear directional-light state: xyz is the energy-weighted first moment,
// w is total energy, and length(xyz) <= w. Blend these moments before decoding.
// Directions and query normals must be unit vectors. Diffuse stores Li/p;
// specular uses its own q*Li/p measure and must use its matching decoder.
float maxent_kappa(float len_v, float omega) {
    if (omega < 1e-8) return 0.0;
    return clamp(len_v / omega, 0.0, 1.0 - 1e-6);
}

// Unit-energy clamped-cosine response. Rationalize the back-facing branch
// to preserve small positive tails instead of subtracting nearly equal terms.
float maxent_cosine_response(float kappa, float cosine) {
    float oneMinusK2 = (1.0 - kappa) * (1.0 + kappa);
    float kCosine = kappa * cosine;
    float d = sqrt(max(oneMinusK2 + kCosine * kCosine, 1e-20));
    float sumTerm = kCosine < 0.0
        ? oneMinusK2 / (d - kCosine) : d + kCosine;
    return sumTerm * sumTerm / (4.0 * d);
}

// Returns irradiance, without a material albedo or the Lambertian 1/pi.
float maxent_irradiance(vec4 encoded, vec3 n) {
    float omega = encoded.w;
    if (omega <= 1e-6) return 0.0;
    float lenV = length(encoded.xyz);
    if (lenV <= 1e-6) return 0.25 * omega;
    float kappa = maxent_kappa(lenV, omega);
    float mu = clamp(dot(encoded.xyz / lenV, n), -1.0, 1.0);
    return omega * maxent_cosine_response(kappa, mu);
}

// Spherical proposal density. Keep the same kappa and normalized axis in
// this evaluator and sampler; hemisphere rejection is handled by the caller.
float maxent_guiding_pdf(vec3 wi, vec3 axis, float kappa) {
    float mu = clamp(dot(axis, wi), -1.0, 1.0);
    float d = (1.0 - kappa) + kappa * (1.0 - mu);
    float oneMinusK2 = (1.0 - kappa) * (1.0 + kappa);
    float norm = oneMinusK2 * oneMinusK2 / (4.0 * PI);
    return norm / max(d * d * d, 1e-30);
}

vec3 sample_maxent_guiding(vec3 axis, float kappa, vec2 xi) {
    vec3 T, B;
    orthonormalBasis(axis, T, B);

    float oneMinusK = 1.0 - kappa;
    float oneMinusK2 = oneMinusK * (1.0 + kappa);
    float root = sqrt(oneMinusK * oneMinusK + 4.0 * kappa * xi.x);
    // The rationalized form includes kappa=0 and avoids cancellation near
    // isotropy. The other branch preserves the narrow-lobe endpoints.
    float mu = kappa < 0.1
        ? (4.0 * xi.x - 2.0 + kappa * (3.0 - kappa * kappa))
            / (root * (root + oneMinusK2))
        : (1.0 - oneMinusK2 / root) / kappa;
    mu = clamp(mu, -1.0, 1.0);
    float phi = 2.0 * PI * xi.y;
    float sine = sqrt(max((1.0 - mu) * (1.0 + mu), 0.0));
    return mu * axis + sine * (cos(phi) * T + sin(phi) * B);
}

#endif // MAXENT_GLSL
