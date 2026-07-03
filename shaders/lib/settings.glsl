#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Number of times the ray bounces before stopping. Larger values lead to better image quality but lower performance. [2 3 4 5 6 7]
#define REFRACTIVE_INDEX 1.331 //select the refractive index of water. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 8 // Number of frames to accumulate before displaying the result, only applies when accumulation is set to "Reprojection". Larger values result in smoother images, but more ghosting with lights. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define MAX_WETNESS 0.4 // [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define EXPOSURE_S 5 // [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.25 1.5 1.75 2 2.5 3 3.5 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100]
#define SHARP_VOLUMETRIC_LIGHT 1 // Real or Sharp [0 1]
#define VOLUMETRIC_LIGHT_SAMPLES 8 // [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 28 32]

// Denoiser — Temporal accumulation (100.glsl)
#define TEMPORAL_NORMAL_PARAM 4.0 // Normal similarity power for history rejection. Higher = stricter on curved surfaces, less ghosting. [0.5 1 2 4 8 16 32 64]
#define TEMPORAL_POSITION_PARAM 64.0 // Position/depth sensitivity for history rejection. Higher = less ghosting but more noise on disocclusion. [1 2 4 8 16 32 64 128 256]

// Denoiser — Variance pre-filter (swap2.glsl)
#define VAR_FILTER_RADIUS 1 // Variance pre-filter radius: 1=3×3, 2=5×5, 3=7×7. Larger = more stable variance but slower. [1 2 3]
#define VARIANCE_SCALE 1.0 // Global variance scale applied before SVGF. Higher = more aggressive denoising. [0.125 0.25 0.5 1 2 4 6 8 10 15 20 30]

// Denoiser — SVGF spatial filter (300.glsl)
#define SVGF_NORMAL_POWER 32.0 // Normal edge-stopping power in à-trous filter. Higher = sharper normal edges. [1 2 4 8 16 32 64 128]
#define SVGF_PHI_L 0.0325 // Luminance edge-stopping sensitivity (phi_l). Higher = preserves more texture detail. [0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4]
#define SVGF_POSITION_PARAM 0.0025 // Depth edge-stopping sensitivity. Higher = sharper depth edges. [0.00075 0.00125 0.0025 0.005 0.01 0.02 0.04 0.08]

// Bloom — 多级降采样-上采样金字塔
#define BLOOM_MIX 0.15 // Bloom 与场景的混合比例 (0=关, 1=全 bloom)。 [0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0]
//#define BLOOM_DEBUG_ATLAS // [DEBUG] 取消注释以直接输出 bloomAtlas 图集到屏幕

const float sunPathRotation = 0.0;

/*
const int depthtex0Format = RGBA32F;
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA16F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;
const int colortex9Format = RGBA32F;

const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
const bool colortex9Clear = true;
*/


#endif // SETTINGS_GLSL