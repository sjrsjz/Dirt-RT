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
#define SUN_PATH_ROTATION 30.0 // Latitude-like sun path tilt. 30 degrees is the clear-sky spring reference used by the atmosphere calibration. [0 15 30 45 60 75 90]
#define END_SKYBOX 1 // End skybox rendering. 1 = custom rune-ring skybox with FBM nebula background (higher GPU cost). 0 = fall back to atmospheric scattering + NEE (faster). [0 1]
#define PATHGUIDE_SPATIAL_RADIUS 16.0 // Path-guide ReSTIR Poisson disk spatial sampling radius (pixels). Higher values explore farther neighbours at the cost of more incoherent memory access. [4.0 8.0 12.0 16.0 24.0 32.0]
#define PATHGUIDE_MAX_TEMPORAL_M 64.0 // ReSTIR reservoir temporal memory cap. Limits the effective sample count carried forward from history. Higher values retain more history but react slower. [8.0 16.0 32.0 48.0 64.0 96.0 128.0 256.0]
#define RESTIR_GI_ENABLED 1 // Biased low-history ReSTIR GI path-guiding prewarm. Luminance-resamples a bounded directional prior, then retires to canonical paths as history matures; requires EON. [0 1]
#define RESTIR_GI_SPATIAL_SAMPLES 8 // Current-frame neighbouring proposals considered at the target. Only the selected shifted donor receives one geometry visibility ray. [0 1 2 4 8]
#define RESTIR_GI_SPATIAL_RADIUS 8.0 // Radius in current-frame pixels for first-bounce GI proposals. [2.0 4.0 6.0 8.0 12.0 16.0 24.0]
#define RESTIR_GI_VISIBILITY_MAX_DISTANCE 128.0 // Maximum distance in blocks for ReSTIR spatial geometry-only support rays. [16.0 32.0 64.0 96.0 128.0 192.0 256.0 512.0 1024.0 2048.0]
#define RESTIR_GI_HISTORY_FADE_START 1.0 // Keep full ReSTIR below this geometrically validated previous-frame diffuse history length. [0.0 1.0 2.0 3.0 4.0 6.0 8.0 12.0 16.0]
#define RESTIR_GI_HISTORY_FADE_END 4.0 // Clear the biased prior and switch fully to the canonical MaxEnt path sample at this diffuse history length. [2.0 3.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0]
#define RESTIR_GI_NORMAL_COS 0.98 // Minimum primary geometry-normal cosine for joining a neighbour proposal set. [0.90 0.95 0.98 0.99 0.995]
#define RESTIR_GI_PLANE_DISTANCE 0.05 // Maximum separation from the target primary tangent plane, in blocks. [0.01 0.02 0.03 0.05 0.08 0.12]

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

// -- Shared MaxEnt temporal minimum-MSE model --
// Per-current-estimator trace variance of the denoiser's intrinsic
// moment-space error. This is a statistical model parameter in squared-moment
// units, not a numerical epsilon. Temporal current/history covariance is zero.
#define MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE 1e-10
// Diagnostic fixed current-frame weight. The temporal proposal, Raw history
// commit, denoised history resolve, and Kish update must all use this value.
#define MAXENT_TEMPORAL_FIXED_ALPHA 0.05

// -- MaxEnt diffuse temporal accumulation --
#define MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY 256 // Maximum finite-window Kish effective sample count. [1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384]
#define MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE 1.0 // Reprojection footprint depth tolerance. [0.25 0.5 0.75 1.0 1.5 2.0 3.0 4.0]
#define MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS 1.0 // Diffuse reprojection footprint radius in pixels. [0.5 1.0 1.5 2.0 3.0]

// -- MaxEnt specular temporal accumulation --
// The G-buffer stores GGX alpha; the front end converts it to perceptual
// roughness exactly once before these controls are evaluated.
#define MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY 64 // Maximum finite-window Kish effective sample count. [1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384]
#define MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD 0.01 // Relative temporal plane threshold. [0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.05]
#define MAXENT_SPECULAR_TEMPORAL_LOBE_FRACTION 0.5 // Accepted GGX lobe fraction. [0.25 0.35 0.5 0.65 0.75 0.9]
#define MAXENT_SPECULAR_TEMPORAL_REPROJECTION_RADIUS 1.0 // Specular reprojection footprint radius in pixels. [0.5 1.0 1.5 2.0 3.0]

// -- Shared MaxEnt variance preparation (diffuse + specular) --
#define MAXENT_VARIANCE_KERNEL_SIGMA 1.0 // Gaussian kernel sigma (pixels) for short-history spatial variance pooling. Higher = wider support. [0.5 0.75 1.0 1.25 1.5 2.0 2.5]
#define MAXENT_VARIANCE_HISTORY_BEGIN 2.0 // N_eff at which spatial→temporal variance transition begins. [1.0 2.0 4.0 6.0 8.0]
#define MAXENT_VARIANCE_HISTORY_END 4.0 // N_eff at which temporal estimator variance becomes fully trusted. [4.0 6.0 8.0 12.0 16.0 24.0 32.0]

// -- Unified MaxEnt spatial filter --
#define MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE 0.1 // Relative projected plane-depth tolerance, including MaxEnt virtual-image planes. [0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5]
// Trace-error correlations induced by the fixed sampling kernels alone.
// These constants depend on the pass and accumulated kernel footprint, but
// not on phi, scene geometry, MaxEnt, the light distribution, or N_eff.
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_1 0.0
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_2 0.1633
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_4 0.2171
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_8 0.2508
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_16 0.2585
#define MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_32 0.2665
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_1 0.0
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_2 0.1060
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_4 0.1412
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_8 0.1394
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_16 0.1471
#define MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_32 0.1525
// The fixed final kernel gives about 0.9446 axial and 0.9443 diagonal
// adjacent correlation. Uniform bilinear phases weight these 0.8 and 0.2.
#define MAXENT_TEMPORAL_REPROJECTION_CORRELATION 0.9446
// Surface- and virtual-motion histories have scene-dependent separation. The
// fixed-kernel adjacent value is only an explicit local-overlap approximation.
#define MAXENT_SPECULAR_BRANCH_CORRELATION 0.9446
// The post-A-Trous 5x5 robust estimator consumes heavily overlapping final
// kernels. Expanding the six fixed sampling passes while omitting phi and
// geometry gives about 0.9913 pair correlation within that neighborhood.
#define MAXENT_TEMPORAL_ROBUST_PROPAGATION_CORRELATION 0.9913
#define MAXENT_SPATIAL_DIFFUSE_LIGHT_FIELD_SENSITIVITY 0.25 // Diffuse variance-normalized moment-space rejection strength. Higher preserves more contrast. [0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define MAXENT_SPATIAL_SPECULAR_LIGHT_FIELD_SENSITIVITY 0.25 // Specular variance-normalized moment-space rejection strength. Higher preserves more contrast. [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

// -- Refraction/path-guide spatial compatibility (not MaxEnt filtering) --
#define MAXENT_SPATIAL_NORMAL_SENSITIVITY 32.0 // Refraction variance-filter texture-normal rejection strength. [1 2 4 8 16 32 64 128]

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
#define GI_CLAMP_MAX 100.0 // Hard ceiling on indirect lighting before MaxEnt encoding. Prevents individual path outliers from destabilising the light field. Higher = more headroom, lower = stronger clamping. [50.0 75.0 100.0 150.0 250.0 500.0 1000.0 2500.0 5000.0 10000.0 20000.0 32000.0]
#define VPROJDIST_SKY 60000.0 // Virtual projected distance assigned to sky hits (m). Used by specular/refraction denoiser to tag infinity. [5000.0 10000.0 25000.0 50000.0 60000.0 100000.0 250000.0]

// -- Debug view --
#define DEBUG_VIEW 0 // Debug output mode. [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 31 32 33 34 35 36 37]

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
