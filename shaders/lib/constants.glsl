#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL
#include "/lib/settings.glsl"
const float PI = 3.141592654;
//const float EPSILON = 0.0001;
#define Correction
const float SunDivMoon = 100000;
const float EPSILONS = 0.;
const float EPSILON_MIN = 0.001;
const float Far = 20000;
const int MaxRay = RAY_BOUNCES;
const float DOF_R = 0;
const vec2 DOF_Pos = vec2(0);
const float march_s = 1;
const float FogS = 25;
const int Reflection = 1;
const int Diffussion = 2;
const int Refraction = 3;

#define Method2

//Vulkanite
const int World_OVERWORLD = 0;
const int World_THE_NETHER = 1;
const int World_THE_END = 2;
const int World_OVERWORLD_CAVE = 3;

//#define SRR //super-resolution reconstruction


#endif // CONSTANTS_GLSL
