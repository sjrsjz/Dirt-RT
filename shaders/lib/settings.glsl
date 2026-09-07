#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL


#if defined(MC_GL_NV_gpu_shader5)
    #extension GL_NV_gpu_shader5 : require
    #define HAS_NATIVE_FP16 1
#elif defined(MC_GL_AMD_gpu_shader_half_float)
    #extension GL_AMD_gpu_shader_half_float : require
    #define HAS_NATIVE_FP16 1
#else
    #error "Dirt RT requires either GL_NV_gpu_shader5 or GL_AMD_gpu_shader_half_float."
#endif


// ===========================================================================
// Dirt RT — Shader Settings
// ===========================================================================
// All adjustable parameters are defined here with Iris-compatible #[...] hints.
// Iris reads these defines & comments to auto-generate the settings UI.
// ===========================================================================

// -- Ray Tracing --
#define RAY_BOUNCES 5 // Max ray bounces before termination. Higher = better image quality, lower FPS. [2 3 4 5 6 7]
#define ACCUMULATION_LENGTH 32 // Refraction history length. Higher = smoother refraction, more ghosting during motion. [1 2 4 8 16 32 64]
#define PATH_GUIDING_STRENGTH 0.75 // Total guided mixture strength. Cache probes split this probability between stable MaxEnt and validated RIS proposals while retaining at least 10% uniform sampling. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.925 0.95 0.975 0.99 0.999]
#define SPECULAR_PATH_GUIDING_STRENGTH 0.5 // Previous denoised q*Li guiding strength for primary non-delta reflection paths. The actual probability is reduced by directional concentration and valid reprojection coverage, then capped at 0.75. [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75]
#define SUN_PATH_ROTATION 30.0 // Latitude-like sun path tilt. 30 degrees is the clear-sky spring reference used by the atmosphere calibration. [0 15 30 45 60 75 90]
#define END_SKYBOX 1 // End skybox rendering. 1 = custom rune-ring skybox with FBM nebula background (higher GPU cost). 0 = fall back to atmospheric scattering + NEE (faster). [0 1]

// -- Material --
#define EON_ENABLED 1 // Enable energy-preserving Oren--Nayar rough diffuse. 0 keeps the legacy Disney/Lambert reconstruction. [0 1]
#define REFRACTIVE_INDEX 1.331 // Water Index of Refraction (IOR). Affects caustics, underwater distortion and specular. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define GLASS_REFRACTIVE_INDEX 1.52 // Glass Index of Refraction (IOR). Kept separate from water throughout reflection, BTDF and PSR. [1.40 1.42 1.45 1.47 1.50 1.52 1.55 1.60 1.65 1.70]
#define MAX_WETNESS 0.4 // Maximum surface wetness from rain or water. Controls porous darkening and roughness without altering substrate F0. [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define POM_ENABLED 1 // Enable Parallax Occlusion Mapping (POM) for detailed surface displacement on the first ray hit. 0 = off (better FPS), 1 = on (better visuals). [0 1]

// -- POM Quality --
#define POM_STEPS 32 // POM linear ray-marching steps. Higher = sharper displacement, lower = faster. [8 16 24 32 48 64]
#define POM_DEPTH 0.25 // POM displacement depth scale. Higher values produce deeper apparent surface relief. [0.05 0.1 0.15 0.2 0.25 0.35 0.5 0.75 1.0]
#define BINARY_SEARCH_STEPS 6 // POM binary refinement iterations after linear search. Higher = more precise surface intersection. [0 2 4 6 8 12]

// -- Sparse Radiance Cache --
#define RADIANCE_CACHE_MAX_ALLOCATIONS_PER_FRAME 128 // Maximum new physical bricks admitted per frame. Lower values reduce allocation spikes but fill the cache more slowly. [32 64 128 256 512]
#define RADIANCE_CACHE_RESIDENT_HYSTERESIS_BUCKETS 2 // Distance-bucket advantage for resident bricks. Higher values reduce camera-motion churn but retain old bricks longer. [0 1 2 3 4 6 8]
#define RADIANCE_CACHE_MARK_TILE_SIZE 8 // One screen sample per NxN tile submits geometry requests. Lower values improve coverage but increase request atomics. [4 8 16]
#define RADIANCE_CACHE_ROUGH_SPECULAR_THRESHOLD 0.35 // PSR-style cascaded roughness required before a secondary specular lobe may terminate into the cache. Lower values are faster; higher values preserve sharper reflections. [0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.8]
#define RADIANCE_CACHE_DIFFUSE_MIN_BOUNCE 3 // First path bounce at which a diffuse surface may terminate into the cache. Higher values reduce cache artifacts but trace more rays. [2 3 4 5 6]
#define VOXEL_SIZE 1.0 // Radiance-cache voxel size in world units. Lower values increase spatial precision at the cost of memory and fill rate. [0.5 0.75 1.0 1.5 2.0]
#define RADIANCE_CACHE_MAX_HIST 32.0 // Maximum effective sample count M retained by each cache voxel's temporal RIS reservoir. Higher values reuse bright samples longer; lower values respond faster to lighting changes. [4.0 8.0 16.0 24.0 32.0 48.0 64.0 96.0 128.0]
#define RADIANCE_CACHE_FILTER_MAX_HIST 16.0 // Temporal sample count used to accumulate the cache's directional MaxEnt moments. Higher values reduce variance; lower values react faster and ghost less. [1.0 2.0 3.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0]
#define RADIANCE_CACHE_UPDATE_PERIOD 4 // Existing cache voxels update once per N frames; newly allocated bricks are fully initialized immediately. [1 2 4 8 16]
#define RADIANCE_CACHE_RIS_GUIDING_STRENGTH 0.35 // Fraction of the cache probe's guided probability assigned to a directionally consistent temporal RIS proposal. [0.0 0.1 0.2 0.25 0.35 0.5 0.65 0.75 1.0]
#define RADIANCE_CACHE_RIS_GUIDING_KAPPA 0.75 // Concentration of the finite-width RIS proposal lobe. Higher values focus more tightly around the selected direction. [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95]

// -- Shared MaxEnt temporal response (diffuse + specular) --
#define MAXENT_TEMPORAL_RESPONSE_DISTANCE_SCALE 0.005 // Legacy Bures response strength; 0 disables light-change response. Higher values reject history more aggressively. [0.0 0.00001 0.000025 0.00005 0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1]

// -- MaxEnt diffuse temporal accumulation --
#define MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE 1.0 // Reprojection footprint depth tolerance. [0.25 0.5 0.75 1.0 1.5 2.0 3.0 4.0]
#define MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS 1.0 // Diffuse reprojection footprint radius in pixels. [0.5 1.0 1.5 2.0 3.0]

// -- MaxEnt specular temporal accumulation --
// The G-buffer stores GGX alpha; the front end converts it to perceptual
// roughness exactly once before these controls are evaluated.
#define MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD 0.01 // Relative temporal plane threshold. [0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.05]
#define MAXENT_SPECULAR_TEMPORAL_LOBE_FRACTION 0.5 // Accepted GGX lobe fraction. [0.25 0.35 0.5 0.65 0.75 0.9]
#define MAXENT_SPECULAR_TEMPORAL_REPROJECTION_RADIUS 1.0 // Specular reprojection footprint radius in pixels. [0.5 1.0 1.5 2.0 3.0]

// -- Shared MaxEnt variance preparation (diffuse + specular) --
#define MAXENT_VARIANCE_KERNEL_SIGMA 1.0 // Gaussian kernel sigma (pixels) for short-history spatial variance pooling. Higher = wider support. [0.5 0.75 1.0 1.25 1.5 2.0 2.5]
#define MAXENT_VARIANCE_SPATIAL_ONLY_SAMPLES 1.0 // Effective frames forced to use spatially reconstructed temporal-moment MC variance. [1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0]
#define MAXENT_VARIANCE_TRANSITION_SAMPLES 1.0 // Additional effective frames used to transition from spatial to temporal MC variance. [1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0]

// -- Unified MaxEnt spatial filter --
#define MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE 0.1 // Relative projected plane-depth tolerance, including MaxEnt virtual-image planes. [0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5]
#define MAXENT_SPATIAL_DIFFUSE_LIGHT_FIELD_SENSITIVITY 0.2 // Diffuse variance-normalized Bures rejection strength. Higher preserves more contrast. [0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define MAXENT_SPATIAL_SPECULAR_LIGHT_FIELD_SENSITIVITY 0.2 // Specular variance-normalized Bures rejection strength. Higher preserves more contrast. [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

// -- Bloom --
#define CAMERA_VIGNETTE_STRENGTH 0.5 // Mix weight of the RT-projection/FOV-aware cos^4 lens falloff. 0 = off, 1 = ideal cos^4 falloff. [0.0 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.75 1.0]
#define BLOOM_MIX 0.15 // Bloom blend strength. 0 = off (scene only), 1 = full bloom. [0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0]
#define BLOOM_DIFFUSION_SCALE 1.0 // Multiplies the physical RGB scattering sigma at every Gaussian diffusion stage. [0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.5]
#define BLOOM_CHROMATIC_SCATTER 0.5 // Rayleigh fraction of lens-scattering power. 0 = wavelength-independent large-particle limit; 1 = pure small-particle Rayleigh limit. [0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0]

// -- Exposure & Display --
#define DISPLAY_MAX_LUMINANCE 100 // Peak brightness of your display in nits (cd/m²). Used for HDR exposure calculation. [50 75 100 150 200 300 400 500 600 700 800 900 1000]
#define DISPLAY_PAPER_WHITE_LUMINANCE 0.5 // Target paper white luminance in sRGB normalized space (0–1). Sets the mid-gray anchor for auto exposure — lower = brighter scene. [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define AUTO_EXPOSURE_OUTLIER_TOLERANCE 2.0 // Log-luminance outlier weighting width. Lower values reject extreme samples more aggressively; higher values retain more of them. [0.25 0.5 0.75 1.0 1.5 2.0 3.0 4.0 6.0 8.0]
#define EXPOSURE_CURVE_K 0.0 // Highlight compression before tonemap: f(x)=ln(k+e^x)-ln(1+k). 0=off, higher=more compression. [0.0 0.1 0.25 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
#define PUPIL_MIN_DIAMETER_MM 2.0 // Minimum physical pupil diameter in millimetres. Limits fast optical adaptation in bright scenes; neural adaptation still supplies the remaining exposure range. [1.0 1.5 2.0 2.5 3.0 3.5 4.0]
#define PUPIL_MAX_DIAMETER_MM 8.0 // Maximum physical pupil diameter in millimetres. Limits fast optical adaptation in dark scenes; neural adaptation still supplies the remaining exposure range. [4.0 5.0 6.0 7.0 8.0 9.0 10.0]

// -- Misc --
#define FIREFLY_SUPPRESSION_MULTIPLIER 50.0 // Per-sample brightness cap multiplier (× average exposure). Lower values clamp fireflies more aggressively. [1.0 5.0 10.0 25.0 50.0 100.0 250.0 500.0]
#define GI_CLAMP_MAX 32000.0 // Hard ceiling on indirect lighting before MaxEnt encoding. Prevents individual path outliers from destabilising the light field. Higher = more headroom, lower = stronger clamping. [50.0 75.0 100.0 150.0 250.0 500.0 1000.0 2500.0 5000.0 10000.0 20000.0 32000.0]
#define VPROJDIST_SKY 60000.0 // Virtual projected distance assigned to sky hits (m). Used by specular/refraction denoiser to tag infinity. [5000.0 10000.0 25000.0 50000.0 60000.0 100000.0 250000.0]

// -- Debug --
#ifndef DEBUG_RT_FORCE_GEOMETRY_NORMAL
#define DEBUG_RT_FORCE_GEOMETRY_NORMAL 0 // Use face-forward geometry normals for all RT material evaluations and bypass RT POM. Raster G-buffer materials are unchanged. [0 1]
#endif
#ifndef DEBUG_VIEW
#define DEBUG_VIEW 0 // Categorized debug output mode. IDs are declared in /lib/debug/view_ids.glsl. [0 1 2 3 4 5 6 7 10 11 12 13 20 21 22 23 24 25 30 31 32 33 34 35 36 37 38 40 41 50 51 52 53 54 55 60 61 62]
#endif

/*
const int depthtex0Format = RGBA32F;
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32UI;
const int colortex4Format = RGBA32UI;
const int colortex5Format = RGBA32UI;
const int colortex6Format = RGBA32UI;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool depthtex0Clear = true;
const bool colortex1Clear = true;
const bool colortex2Clear = true;
const bool colortex3Clear = true;
const bool colortex4Clear = true;
const bool colortex5Clear = true;
const bool colortex6Clear = true;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

#endif // SETTINGS_GLSL
