#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL
#include "/lib/settings.glsl"
const float PI = 3.141592654;
const float LOG2_E = 1.44269504089; // 1/ln(2), for exp(x)=exp2(x*LOG2_E)
const float SunDivMoon = 1000;
const int MaxRay = RAY_BOUNCES;
// FIREFLY_SUPPRESSION_MULTIPLIER — now a #define in settings.glsl
const int REFLECTION = 1;
const int DIFFUSION = 2;
const int REFRACTION = 3;

// VPROJDIST_SKY — now a #define in settings.glsl

//Vulkanite
const int WORLD_OVERWORLD = 0;
const int WORLD_THE_NETHER = 1;
const int WORLD_THE_END = 2;
const int WORLD_OVERWORLD_CAVE = 3;

// Block IDs — synchronized with shaders/block.properties
#define BLOCK_WATER  1000 // water
#define BLOCK_GLASS  1001 // ice, stained glass (all colors + panes), blue_ice, packed_ice
#define BLOCK_PORTAL 1002 // nether_portal (frosted / translucent emissive)

#endif // CONSTANTS_GLSL
