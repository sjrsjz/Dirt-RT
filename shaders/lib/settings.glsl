#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Max ray bounces before termination. Higher = better image quality, lower FPS. [2 3 4 5 6 7]
#define REFRACTIVE_INDEX 1.331 // Water Index of Refraction (IOR). Affects caustics, underwater distortion and specular. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 20 // Refraction history length. Higher = smoother refraction, more ghosting during motion. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define MAX_WETNESS 0.4 // Maximum surface wetness from rain or water. Controls specular reflection on wet blocks. [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define POM_ENABLED 1 // Enable Parallax Occlusion Mapping (POM) for detailed surface displacement on the first ray hit. 0 = off (better FPS), 1 = on (better visuals). [0 1]

// -- Sparse radiance cache --
#define RADIANCE_CACHE_MAX_ALLOCATIONS_PER_FRAME 128 // Maximum new physical bricks admitted per frame. Lower values reduce allocation spikes but fill the cache more slowly. [32 64 128 256 512]
#define RADIANCE_CACHE_RESIDENT_HYSTERESIS_BUCKETS 2 // Distance-bucket advantage for resident bricks. Higher values reduce camera-motion churn but retain old bricks longer. [0 1 2 3 4 6 8]
#define RADIANCE_CACHE_MARK_TILE_SIZE 8 // One screen sample per NxN tile submits geometry requests. Lower values improve coverage but increase request atomics. [4 8 16]
#define RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD 0.35 // PSR-style cascaded roughness required before a secondary specular lobe may terminate into the cache. Lower values are faster; higher values preserve sharper reflections. [0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.8]
#define RADIANCE_CACHE_DIFFUSE_MIN_BOUNCE 3 // First path bounce at which a diffuse surface may terminate into the cache. Higher values reduce cache artifacts but trace more rays. [2 3 4 5 6]

// -- NRD-inspired low-weight blend (buffer_swap_diffuse.glsl) --
#define NRD_BLEND_STRENGTH 1.0 // Spatial filter blend strength at low temporal confidence. Blends the large-radius spatial result back into history when frame accumulation is insufficient. 0 = off. [0.0 0.1 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 4.0 5.0]

// -- Diffuse temporal accumulation (temporal_diffuse.glsl) --
// -- NVIDIA RELAX specular denoiser --
// RELAX consumes perceptual roughness. The G-buffer stores GGX alpha and the
// front end converts it exactly once.
#define RELAX_SPEC_MAX_HISTORY 30 // Slow specular history length. [5 10 15 20 30 40 60 90]
#define RELAX_SPEC_MAX_FAST_HISTORY 6 // Responsive history length. [1 2 3 4 6 8 12 16]
#define RELAX_PREPASS_RADIUS 1.0 // Specular pre-pass radius in pixels. [0.0 1.0 2.0 3.0 4.0]
#define RELAX_DISOCCLUSION_THRESHOLD 0.01 // Relative temporal plane threshold. [0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.05]
#define RELAX_ROUGHNESS_FRACTION 0.15 // Roughness edge tolerance. [0.05 0.1 0.15 0.2 0.25 0.35 0.5]
#define RELAX_LOBE_ANGLE_FRACTION 0.5 // Accepted GGX lobe fraction. [0.25 0.35 0.5 0.65 0.75 0.9]
#define RELAX_LOBE_ANGLE_SLACK 0.02 // Additional lobe tolerance in radians. [0.0 0.005 0.01 0.02 0.04 0.08]
#define RELAX_DEPTH_THRESHOLD 0.003 // World-space plane-distance scale. [0.001 0.002 0.003 0.005 0.01 0.02]
#define RELAX_MIN_HIT_DISTANCE_WEIGHT 0.1 // Minimum pre-pass hit-distance weight. [0.0 0.05 0.1 0.2 0.35 0.5]
#define RELAX_CURVATURE_STRENGTH 1.0 // Thin-lens curvature correction. [0.0 0.25 0.5 0.75 1.0 1.25 1.5]
#define RELAX_MAX_VIRTUAL_MOTION_ACCELERATION 2.0 // Virtual-motion acceleration guard. [0.5 1.0 1.5 2.0 3.0 4.0]
#define RELAX_ANTIFIREFLY_ENABLE 1 // Rank-conditioned anti-firefly pass. [0 1]
#define RELAX_HISTORY_FIX_FRAMES 3.0 // Frames repaired after disocclusion. [1.0 2.0 3.0 4.0 5.0 8.0]
#define RELAX_HISTORY_FIX_BASE_STRIDE 14.0 // Maximum history-fix stride. [4.0 8.0 12.0 14.0 18.0 24.0]
#define RELAX_COLOR_BOX_SIGMA 2.0 // Responsive YCoCg color-box width. [0.5 1.0 1.5 2.0 2.5 3.0 4.0]
#define RELAX_HISTORY_ACCELERATION 1.0 // Slow-to-responsive anti-lag acceleration. [0.0 0.25 0.5 0.75 1.0 1.5 2.0]
#define RELAX_HISTORY_RESET_TEMPORAL_SIGMA 3.0 // Temporal reset tolerance. [1.0 2.0 3.0 4.0 5.0]
#define RELAX_HISTORY_RESET_SPATIAL_SIGMA 3.0 // Spatial reset tolerance. [1.0 2.0 3.0 4.0 5.0]
#define RELAX_HISTORY_RESET_AMOUNT 0.5 // Maximum reset fraction. [0.0 0.25 0.5 0.75 1.0]
#define RELAX_HISTORY_THRESHOLD 5.0 // Frames before temporal variance is trusted. [1.0 2.0 3.0 5.0 8.0 12.0]
#define RELAX_SPEC_VARIANCE_BOOST 8.0 // Low-confidence zero-sample variance. [0.0 1.0 2.0 4.0 8.0 16.0]
#define RELAX_ROUGHNESS_EDGE_RELAXATION 0.3 // View-vector edge relaxation. [0.0 0.1 0.2 0.3 0.5 0.75 1.0]
#define RELAX_NORMAL_RELAXATION 0.5 // Low-confidence normal relaxation. [0.0 0.25 0.5 0.75 1.0]
#define RELAX_LUMINANCE_RELAXATION 0.5 // Low-confidence luminance relaxation. [0.0 0.25 0.5 0.75 1.0]
#define RELAX_SPEC_PHI_LUMINANCE 1.0 // Variance-normalized luminance sensitivity. [0.25 0.5 0.75 1.0 1.5 2.0 3.0]
#define RELAX_MAX_LUMINANCE_DIFFERENCE 2.0 // Relative luminance rejection clamp. [0.5 1.0 1.5 2.0 3.0 4.0 8.0]

#define TEMPORAL_MAX_HISTORY 32 // Maximum effective sample count clamped per pixel. Higher = smoother but more ghosting. [1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384]
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
#define ATROUS_PHI_L 0.25 // Luma edge-stopping sensitivity. Higher = more aggressive denoising. [0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5]
#define ATROUS_POSITION_PARAM 0.0025 // Depth edge-stopping sensitivity. Higher = sharper depth boundaries preserved. [0.00075 0.00125 0.0025 0.005 0.01 0.02 0.04 0.08]
#define ATROUS_GAMMA 1.0 // Roughness-dependent filter order adaptation strength. 0 = constant filter width regardless of roughness (blurrier on rough surfaces), 1 = standard roughness adaptation, higher = more aggressive widening on rough surfaces. [0.0 0.25 0.5 0.75 1.0 1.5 2.0]

// -- Bloom --
#define BLOOM_MIX 0.15 // Bloom blend strength. 0 = off (scene only), 1 = full bloom. [0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0]

#define DISPLAY_MAX_LUMINANCE 100 // Peak brightness of your display in nits (cd/m²). Used for HDR exposure calculation. [50 75 100 150 200 300 400 500 600 700 800 900 1000]
#define DISPLAY_PAPER_WHITE_LUMINANCE 0.5 // Target paper white luminance in sRGB normalized space (0–1). Sets the mid-gray anchor for auto exposure — lower = brighter scene. [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

// -- Exposure curve --
#define EXPOSURE_CURVE_K 0.0 // Highlight compression before tonemap: f(x)=ln(k+e^x)-ln(1+k). 0=off, higher=more compression. [0.0 0.1 0.25 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

// -- Path guiding --
#define PATH_GUIDING_STRENGTH 0.75 // ALICE-guided mixture strength for diffuse paths and radiance-cache probes. Higher values follow history more aggressively; lower values stay closer to the unbiased baseline sampler. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.925 0.95 0.975 0.99 0.999]
#define PATH_GUIDING_SPECULAR_STRENGTH 0.85 // Base mix probability for ALICE-guided specular reflection. Final probability = STRENGTH × roughness × rho, so smooth surfaces (low roughness) or isotropic fields (low rho) naturally suppress guiding. Guiding is only active when both roughness and rho are meaningful. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 0.99]

// -- Debug view --
#define DEBUG_VIEW 0 // Debug output mode. 0=Normal 1=Diffuse 2=Refract 3=Reflect 4=WhiteModel 5=LightField 6=Normals 7=Absorption 8=ReflDir 9=ReflDist 10=SpecAlbedo 11=Roughness 12=ReflRaw 13=DiffuseWeight 14=ReflectWeight 15=RefractWeight 16=SurfaceEmission 17=MediumEmission 18=DiffuseAlbedo 19=RefrVProjDist 20=PathGuide 21=TemporalRaw 22=RadianceCache [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22]

#define SUN_PATH_ROTATION 45.0 // Sun path rotation angle (degrees around X axis). Adjusts the sun's apparent path in the sky. [0 15 30 45 60 75 90]

#define END_SKYBOX 1 // End skybox rendering. 1 = custom rune-ring skybox with FBM nebula background (higher GPU cost). 0 = fall back to atmospheric scattering + NEE (faster). [0 1]

/*
const int depthtex0Format = RGBA32F;
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32UI;
const int colortex5Format = RGBA32UI;
const int colortex6Format = RGBA32UI;
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
