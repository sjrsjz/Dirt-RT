#ifndef MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL
#define MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL

// Fixed denoiser model and scheduling constants. These are not shader-pack options: changing them alters the
// estimator or provisional pipeline and therefore requires code-level validation.
const float MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE = 1e-10;
const float MAXENT_TEMPORAL_FIXED_ALPHA = 0.01;
// 1-dot(A,B) approximates half the squared angular separation without acos.
const float MAXENT_SPATIAL_PDF_DIRECTION_EXPONENT_SCALE = 0.0; //64.0;

// Cumulative spatial confidence entering each A-Trous pass. These fixed-kernel factors approximate the previous
// flat, fully accepted estimator-variance contraction without changing the linearly propagated MC variance field.
// Difference covariance is intentionally not folded into them.
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_1 = 1.00000000;
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_2 = 2.88235658;
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_4 = 6.23648076;
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_8 = 12.61065131;
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_16 = 25.90473642;
const float MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_32 = 52.46326890;

// Recursive constant-correlation closure for the independent-current Kish
// N_eff. Calibrated on a homogeneous front-facing plane with the runtime
// Bures rejection, phi=0.5, FP16 fields, history=1/4/16/64 and kappa=0/.7/.98.
// Independent probes isolate overlap under frozen signal weights; a held-out
// MC ensemble audits the full adaptive filter separately. See
// tools/calibrate_bures_pass_correlations.py and doc/calibration/README.md.
// Recalibrate when rejection strength, sample pattern or variance policy changes.
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_1 = 0.0;
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_2 = 0.09184833;
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_4 = 0.12613998;
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_8 = 0.13781854;
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_16 = 0.14262104;
const float MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_32 = 0.14655028;

// Surface and virtual specular histories can overlap. This closure affects only their reconstructed Kish N_eff;
// MC variance remains a linearly reconstructed field.
const float MAXENT_SPECULAR_BRANCH_CORRELATION = 1.0;

#endif // MAXENT_DENOISER_INTERNAL_CONSTANTS_GLSL
