#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define RAY_BOUNCES 5 // Number of times the ray bounces before stopping. Larger values lead to better image quality but lower performance. [2 3 4 5 6 7]
#define Refractive_Index 1.331 //select the refractive index of water. [1.30 1.31 1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.40 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.50]
#define ACCUMULATION_LENGTH 8 // Number of frames to accumulate before displaying the result, only applies when accumulation is set to "Reprojection". Larger values result in smoother images, but more ghosting with lights. [1 2 3 4 5 6 7 8 9 10 20 50 100]
#define ACCUMULATION_TYPE 1 // Type of the accumulation [0 1]
#define SRR 0 // Type of SRR [0 1]
#define maxWetness 0.4 // [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define ExposureS 5 // [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.25 1.5 1.75 2 2.5 3 3.5 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100]
#define Sharp_Volumetric_Light 1 // Real or Sharp [0 1]
#define Volumetric_Light_Samples 8 // [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 28 32]

const float sunPathRotation = 0.0;
//const int colortex3Format = RGBA32F;

#endif // SETTINGS_GLSL