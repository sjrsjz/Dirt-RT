#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Number of times the ray bounces before stopping. Larger values lead to better image quality but lower performance. [2 3 4 5 6 7]
#define Refractive_Index 1.331 //select the refractive index of water. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 8 // Number of frames to accumulate before displaying the result, only applies when accumulation is set to "Reprojection". Larger values result in smoother images, but more ghosting with lights. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define maxWetness 0.4 // [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define ExposureS 5 // [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.25 1.5 1.75 2 2.5 3 3.5 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100]
#define Sharp_Volumetric_Light 1 // Real or Sharp [0 1]
#define Volumetric_Light_Samples 8 // [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 28 32]
#define useReSTIR 1 // Whether to use ReSTIR GI [0 1]

// Denoiser — Temporal accumulation (100.glsl)
#define TEMPORAL_NORMAL_PARAM 8.0 // Normal similarity power for history rejection. Higher = stricter on curved surfaces, less ghosting. [0.5 1 2 4 8 16 32 64]
#define TEMPORAL_POSITION_PARAM 64.0 // Position/depth sensitivity for history rejection. Higher = less ghosting but more noise on disocclusion. [1 2 4 8 16 32 64 128 256]

// Denoiser — Variance pre-filter (swap2.glsl)
#define VAR_FILTER_RADIUS 1 // Variance pre-filter radius: 1=3×3, 2=5×5, 3=7×7. Larger = more stable variance but slower. [1 2 3]
#define VARIANCE_SCALE 10.0 // Global variance scale applied before SVGF. Higher = more aggressive denoising. [0.5 1 2 4 6 8 10 15 20 30]

// Denoiser — SVGF spatial filter (300.glsl)
#define SVGF_NORMAL_POWER 32.0 // Normal edge-stopping power in à-trous filter. Higher = sharper normal edges. [1 2 4 8 16 32 64 128]
#define SVGF_PHI_L 16.0 // Luminance edge-stopping sensitivity (phi_l). Higher = preserves more texture detail. [0.5 1 2 4 8 16 32 64]
#define SVGF_POSITION_PARAM 1.0 // Depth edge-stopping sensitivity. Higher = sharper depth edges. [0.25 0.5 1 2 4 8]

// ReSTIR — Reservoir spatiotemporal resampling (ray0.rgen)
#define RESTIR_SPATIAL_SAMPLES 3 // Number of neighbor pixels for spatial reuse. Higher = better convergence but slower. [1 2 3 4 5 6]
#define RESTIR_SPATIAL_RADIUS 4 // Spatial reuse radius in pixels. [1 2 3 4 6 8 10]
#define RESTIR_M_CAP 10.0 // Maximum effective sample count M before clamping. Higher = brighter but more variance. [2 3 5 8 10 15 20 30]

const float sunPathRotation = 0.0;

#endif // SETTINGS_GLSL