#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL
#include "/lib/settings.glsl"
const float PI = 3.141592654;
//const float EPSILON = 0.0001;
#define Correction
const float SunDivMoon=3000;
const float EPSILONS=0.;
const float EPSILON_MIN=0.0005;
const float Far=20000;
const int MaxRay=RAY_BOUNCES;
const float DOF_R=0;
const vec2 DOF_Pos=vec2(0);
const float march_s=1;
const float FogS=25;
const int Reflection=1;
const int Diffussion=2;
const int Refraction=3;

//#define SRR //super-resolution reconstruction

uint getIdx(uvec2 xy){
    return xy.y*2560u+clamp(xy.x,0,2559u);
}
#endif // CONSTANTS_GLSL