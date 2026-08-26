#ifndef MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL
#define MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL

// Fixed denoiser model and scheduling constants. These are not shader-pack options: changing them alters the
// estimator, provisional pipeline, or fixed-kernel covariance model and therefore requires code-level validation.
const float MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE = 1e-10;
const float MAXENT_TEMPORAL_FIXED_ALPHA = 0.01;
// 1-dot(A,B) approximates half the squared angular separation without acos.
const float MAXENT_SPATIAL_PDF_DIRECTION_EXPONENT_SCALE = 64.0;

// Diffuse reprojection expands its bilinear reconstruction footprint under screen-space minification. Bounding
// each Jacobian singular value prevents an unbounded history gather at grazing angles or near clip singularities.
const float MAXENT_DIFFUSE_TEMPORAL_MAX_JACOBIAN_STRETCH = 2.0;

// Trace-error correlations induced by the fixed A-Trous kernels. They depend on the pass footprint, not on phi,
// scene geometry, MaxEnt decoding, the light distribution, or N_eff.
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_1 = 0.0;
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_2 = 0.1633;
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_4 = 0.2171;
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_8 = 0.2508;
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_16 = 0.2585;
const float MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_32 = 0.2665;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_1 = 0.0;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_2 = 0.1060;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_4 = 0.1412;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_8 = 0.1394;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_16 = 0.1471;
const float MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_32 = 0.1525;

// The fixed final kernel gives about 0.9446 axial and 0.9443 diagonal adjacent correlation. Uniform bilinear
// phases weight these 0.8 and 0.2. Surface and virtual histories use that adjacent value as a local-overlap model.
// const float MAXENT_TEMPORAL_REPROJECTION_CORRELATION = 0.9446;
// const float MAXENT_SPECULAR_BRANCH_CORRELATION = 0.9446;
const float MAXENT_TEMPORAL_REPROJECTION_CORRELATION = 0.0;
const float MAXENT_SPECULAR_BRANCH_CORRELATION = 1.0;

// Keep the propagation correlation fixed during the 3x3-versus-5x5 support test so the experiment changes only
// the robust reconstruction neighborhood rather than simultaneously changing its variance model.
// const float MAXENT_TEMPORAL_ROBUST_PROPAGATION_CORRELATION = 0.9913;
const float MAXENT_TEMPORAL_ROBUST_PROPAGATION_CORRELATION = 1.0;

#endif // MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL
