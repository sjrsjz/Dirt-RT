#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Max ray bounces before termination. Higher = better image quality, lower FPS. [2 3 4 5 6 7]
#define REFRACTIVE_INDEX 1.331 // Water Index of Refraction (IOR). Affects caustics, underwater distortion and specular. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 20 // Frames to accumulate via reprojection. Higher = smoother image, more ghosting on moving lights/camera. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define MAX_WETNESS 0.4 // Maximum surface wetness from rain or water. Controls specular reflection on wet blocks. [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

// -- Temporal history rejection (temporal_diffuse.glsl) --
#define TEMPORAL_NORMAL_PARAM 4.0 // How strictly normal differences reject history. Higher = less ghosting on curved surfaces, more noise. [0.5 1 2 4 8 16 32 64]
#define TEMPORAL_POSITION_PARAM 64.0 // How strictly position/depth differences reject history. Higher = less ghosting, more disocclusion noise. [1 2 4 8 16 32 64 128 256]

// -- NRD-inspired low-weight blend (buffer_swap_diffuse.glsl) --
#define NRD_BLEND_STRENGTH 1.0 // Spatial filter blend strength at low temporal confidence. Blends the large-radius spatial result back into history when frame accumulation is insufficient. 0 = off. [0.0 0.1 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 4.0 5.0]

// -- Specular reflection temporal confidence (temporal_reflect.glsl) --
#define REFLECT_GGX_CONFIDENCE 25.0 // GGX lobe similarity strictness for specular temporal accumulation. Lower = more tolerant of normal/roughness mismatch → smoother but more ghosting. Higher = stricter lobe match → less ghosting but noisier on curved surfaces. [1.0 2.0 5.0 10.0 20.0 25.0 50.0 75.0 100.0]

// -- Diffuse temporal accumulation (temporal_diffuse.glsl) --
#define TEMPORAL_MAX_HISTORY 32 // Maximum effective sample count clamped per pixel. Higher = smoother but more ghosting. [1 2 4 8 16 32 64 128 256 512 1024 8192 16384]
#define TEMPORAL_CONFIDENCE_POWER 1.0 // Reprojection confidence exponent. Higher = more aggressive rejection of mismatched history. [0.1 0.25 0.5 1.0 2.0 4.0]
#define TEMPORAL_HISTORY_MIN_WEIGHT 0.0001 // Weight threshold below which history is discarded and reset. [0.000001 0.00001 0.0001 0.001 0.01]
#define TEMPORAL_AABB_ENABLE 1 // AABB clamp in ALICE augmented space to prevent ghosting. 0 = fall back to raw EMA blend. [0 1]
#define TEMPORAL_AABB_NEIGHBOR_RADIUS 2 // AABB neighborhood radius. 1 = 3×3, 2 = 5×5. [1 2 3]
#define TEMPORAL_AABB_EXPAND 2.0 // AABB extent expand factor. Compensates for min/max underestimation from sparse neighbor samples. [0.5 1.0 1.5 2.0 3.0 4.0]
#define TEMPORAL_AABB_SIGMA_SCALE 3.0 // AABB sigma-guided expansion (× √Var_scalar). 3.0 ≈ 3-sigma. [0.5 1.0 1.5 2.0 3.0 4.0 5.0]
#define TEMPORAL_AABB_MIN_EXTENT 1.0 // AABB minimum absolute extent. Prevents dark regions from being clamped to zero. [0.01 0.1 0.5 1.0 2.0 5.0]
#define TEMPORAL_AABB_BOX_SCALE 1.0 // AABB global scale. Tweak this first when overall clamp feels too aggressive or too conservative. [0.25 0.5 0.75 1.0 1.5 2.0]
#define TEMPORAL_AABB_MIN_VALID_NEIGHBORS 2 // Minimum valid neighbor count below which AABB clamp is skipped. [1 2 3 4 5 6 7 8]

// -- À-trous spatial filter (atrous_denoise_diffuse.glsl) --
#define ATROUS_NORMAL_POWER 32.0 // Normal edge-stopping sensitivity in à-trous wavelet filter. Higher = sharper normal edges preserved. [1 2 4 8 16 32 64 128]
#define ATROUS_PHI_L 0.000005 // Luma edge-stopping sensitivity. Higher = more aggressive denoising. [0.000001 0.0000025 0.000005 0.000075 0.00001 0.000025 0.00005 0.0001 0.00025 0.0005 0.001]
#define ATROUS_POSITION_PARAM 0.0025 // Depth edge-stopping sensitivity. Higher = sharper depth boundaries preserved. [0.00075 0.00125 0.0025 0.005 0.01 0.02 0.04 0.08]
#define ATROUS_GAMMA 1.0 // Roughness-dependent filter order adaptation strength. 0 = constant filter width regardless of roughness (blurrier on rough surfaces), 1 = standard roughness adaptation, higher = more aggressive widening on rough surfaces. [0.0 0.25 0.5 0.75 1.0 1.5 2.0]

// -- Bloom --
#define BLOOM_MIX 0.15 // Bloom blend strength. 0 = off (scene only), 1 = full bloom. [0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0]

#define DISPLAY_MAX_LUMINANCE 100 // Peak brightness of your display in nits (cd/m²). Used for HDR exposure calculation. [50 75 100 150 200 300 400 500 600 700 800 900 1000]
#define DISPLAY_PAPER_WHITE_LUMINANCE 0.5 // Target paper white luminance in sRGB normalized space (0–1). Sets the mid-gray anchor for auto exposure — lower = brighter scene. [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

// -- Exposure curve --
#define EXPOSURE_CURVE_K 0.0 // Highlight compression before tonemap: f(x)=ln(k+e^x)-ln(1+k). 0=off, higher=more compression. [0.0 0.1 0.25 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

// -- Path guiding --
#define PATH_GUIDING_STRENGTH 0.75 // Mix probability weight for ALICE-guided importance sampling vs cosine-weighted sampling. Higher = more samples steered toward the prior, lower = more uniform. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.925 0.95 0.975 0.99 0.999]
#define PATH_GUIDING_SPECULAR_STRENGTH 0.85 // Base mix probability for ALICE-guided specular reflection. Final probability = STRENGTH × roughness × rho, so smooth surfaces (low roughness) or isotropic fields (low rho) naturally suppress guiding. Guiding is only active when both roughness and rho are meaningful. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 0.99]

// -- NRD curvature correction --
#define CURVATURE_CORRECTION_STRENGTH 1.0 // NRD curvature correction strength for specular virtual distance. 1.0 = standard NRD, 0.0 = off (planar assumption). Corrects virtual reprojection distance based on local surface curvature to reduce ghosting on curved surfaces. [0.0 0.25 0.5 0.75 1.0 1.25 1.5]

// -- Debug view --
#define DEBUG_VIEW 0 // Debug output mode. 0=Normal 1=Diffuse 2=Refract 3=Reflect 4=WhiteModel 5=LightField 6=Normals 7=Absorption 8=ReflDir 9=ReflDist 10=SpecAlbedo 11=Roughness 12=ReflRaw 13=DiffuseWeight 14=ReflectWeight 15=RefractWeight 16=DirectLight 17=Emission 18=DiffuseAlbedo 19=RefrVProjDist 20=PathGuide 21=TemporalRaw [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21]

#define SUN_PATH_ROTATION 45.0 // Sun path rotation angle (degrees around X axis). Adjusts the sun's apparent path in the sky. [0 15 30 45 60 75 90]

#define END_SKYBOX 1 // End skybox rendering. 1 = custom rune-ring skybox with FBM nebula background (higher GPU cost). 0 = fall back to atmospheric scattering + NEE (faster). [0 1]

/*
const int depthtex0Format = RGBA32F;
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;
const int colortex9Format = RGBA32F;

const bool depthtex0Clear = true;
const bool colortex1Clear = true;
const bool colortex2Clear = true;
const bool colortex3Clear = true;
const bool colortex4Clear = true;
const bool colortex5Clear = true;
const bool colortex6Clear = true;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
const bool colortex9Clear = true;
*/


#endif // SETTINGS_GLSL