#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL
#include "/lib/settings.glsl"
const float PI = 3.141592654;
const float LOG2_E = 1.44269504089; // 1/ln(2), for exp(x)=exp2(x*LOG2_E)
const float SunDivMoon = 1000;
const int MaxRay = RAY_BOUNCES;
const float FIREFLY_SUPPRESSION_MULTIPLIER = 50.0;
const int REFLECTION = 1;
const int DIFFUSION = 2;
const int REFRACTION = 3;

// NRD specular denoising (301)
const float SPEC_BLUR_BOOST       = 0.025; // edge-stop softening (<1 = more blur)
const float SPEC_SURF_PARAM       = 1.0;   // surface continuity
const float SPEC_LOBE_DIVISOR     = 4.0;   // GGX lobe width = 1/(α²·DIV)
const float SPEC_HIT_DIST_SENS    = 2.0;   // virtual hit distance sensitivity
const float SPEC_MIN_ALPHA        = 0.001; // min GGX α to avoid div-by-zero
const float SPEC_LUMA_ROUGH_SOFT  = 3.0;   // roughness softening for luma weight
const float SPEC_ROUGH_NORM_A     = 0.99;  // roughness weight norm: r²·A + B
const float SPEC_ROUGH_NORM_B     = 0.01;

// Curvature-guided geometry skip (swap2 → 300)
const float CURVATURE_THRESHOLD   = 0.01;  // |K| > this → omega negated → skip geometry edge-stop

//Vulkanite
const int WORLD_OVERWORLD = 0;
const int WORLD_THE_NETHER = 1;
const int WORLD_THE_END = 2;
const int WORLD_OVERWORLD_CAVE = 3;


#endif // CONSTANTS_GLSL
