#ifndef DENOISER_ATROUS_POLICY_GLSL
#define DENOISER_ATROUS_POLICY_GLSL

// Lifetime policy shared by both signal domains. Intermediate steps need both
// estimators. Resolve consumes only the final independent-current estimator;
// proposal inputs still guide every tap, including the last step.
#if defined(MAXENT_ATROUS_STEP) && MAXENT_ATROUS_STEP == 32
#define DENOISER_SPATIAL_WRITE_PROPOSAL 0
#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FILTERED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FILTERED_MONTE_CARLO_VARIANCE
#define DENOISER_SPATIAL_ACCUMULATE_PROPOSAL 1
#else
#define DENOISER_SPATIAL_ACCUMULATE_PROPOSAL 0
#endif
#else
#define DENOISER_SPATIAL_WRITE_PROPOSAL 1
#define DENOISER_SPATIAL_ACCUMULATE_PROPOSAL 1
#endif

#endif // DENOISER_ATROUS_POLICY_GLSL
