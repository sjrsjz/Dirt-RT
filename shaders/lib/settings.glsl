#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Max ray bounces before termination. Higher = better image quality, lower FPS. [2 3 4 5 6 7]
#define REFRACTIVE_INDEX 1.331 // Water Index of Refraction (IOR). Affects caustics, underwater distortion and specular. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 8 // Frames to accumulate via reprojection. Higher = smoother image, more ghosting on moving lights/camera. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define MAX_WETNESS 0.4 // Maximum surface wetness from rain or water. Controls specular reflection on wet blocks. [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define SHARP_VOLUMETRIC_LIGHT 1 // Volumetric light quality. ON = sharp analytic falloff. OFF = realistic soft scattering. [0 1]
#define VOLUMETRIC_LIGHT_SAMPLES 8 // Samples per volumetric light ray. Higher = less noise, lower FPS. [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 28 32]

// -- Temporal history rejection (100.glsl) --
#define TEMPORAL_NORMAL_PARAM 4.0 // How strictly normal differences reject history. Higher = less ghosting on curved surfaces, more noise. [0.5 1 2 4 8 16 32 64]
#define TEMPORAL_POSITION_PARAM 64.0 // How strictly position/depth differences reject history. Higher = less ghosting, more disocclusion noise. [1 2 4 8 16 32 64 128 256]

// -- Variance pre-filter (swap2.glsl) --
#define VAR_FILTER_RADIUS 1 // Variance pre-filter radius. 1=3×3, 2=5×5, 3=7×7. Larger = more stable variance, slower. [1 2 3]
#define VARIANCE_SCALE 1.0 // Global variance scale before SVGF. Higher = more aggressive denoising. [0.125 0.25 0.5 1 2 4 6 8 10 15 20 30]

// -- SVGF spatial filter (300.glsl) --
#define SVGF_NORMAL_POWER 32.0 // Normal edge-stopping sensitivity in à-trous wavelet filter. Higher = sharper normal edges preserved. [1 2 4 8 16 32 64 128]
#define SVGF_PHI_L 0.0325 // Luminance edge-stopping sensitivity. Higher = preserves more fine texture detail. [0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4]
#define SVGF_POSITION_PARAM 0.0025 // Depth edge-stopping sensitivity. Higher = sharper depth boundaries preserved. [0.00075 0.00125 0.0025 0.005 0.01 0.02 0.04 0.08]

// -- Geometry-guided filter (swap2 → 300) --
#define ENABLE_GAUSSIAN_FILTER 0 // Curvature-guided geometry weight mask. OFF = fast path (skip curvature computation). [0 1]

// -- Bloom --
#define BLOOM_MIX 0.15 // Bloom blend strength. 0 = off (scene only), 1 = full bloom. [0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0]

#define DISPLAY_MAX_LUMINANCE 400.0 // Peak brightness of your display in nits (cd/m²). Used for HDR exposure calculation. [50 75 100 150 200 300 400 500 600 700 800 900 1000]

// -- Exposure curve --
#define EXPOSURE_CURVE_K 1.0 // Highlight compression before tonemap: f(x)=ln(k+e^x)-ln(1+k). 0=off, higher=more compression. [0.0 0.1 0.25 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

// -- Debug view --
#define DEBUG_VIEW 0 // Debug output mode. 0=Normal 1=Diffuse 2=Refract 3=Reflect 4=WhiteModel 5=LightField 6=Normals 7=Absorption [0 1 2 3 4 5 6 7]

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