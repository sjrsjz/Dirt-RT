#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL
#include "/lib/settings.glsl"
const float PI = 3.141592654;
const float LOG2_E = 1.44269504089; // 1/ln(2), for exp(x)=exp2(x*LOG2_E)
const float SunDivMoon = 1000;
const int MaxRay = RAY_BOUNCES;
const float FIREFLY_SUPPRESSION_MULTIPLIER = 50.0;
const int Reflection = 1;
const int Diffussion = 2;
const int Refraction = 3;

//Vulkanite
const int World_OVERWORLD = 0;
const int World_THE_NETHER = 1;
const int World_THE_END = 2;
const int World_OVERWORLD_CAVE = 3;


#endif // CONSTANTS_GLSL
