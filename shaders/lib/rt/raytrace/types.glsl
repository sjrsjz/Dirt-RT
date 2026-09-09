#ifndef DIRT_RT_LIB_RT_RAYTRACE_TYPES_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_TYPES_GLSL

// Transient transport records; no buffer layout or resource bindings.

struct HalfVector {
    vec3 H;
    bool valid;
};

struct LobeProbs {
    float P_spec, P_refr, P_diff;
    vec3 diffWeight; // Diffuse cache response divided by P_diff.
};

struct MediumResult {
    vec3 absorption;
    vec3 emission;
};

struct GuideInfo {
    vec3 axis;
    float kappa;
    float prob;
    bool valid;
};

GuideInfo emptyGuideInfo() {
    GuideInfo g;
    g.axis = vec3(0.0, 1.0, 0.0);
    g.kappa = 0.0;
    g.prob = 0.0;
    g.valid = false;
    return g;
}

struct PSRResult {
    float virtualDist;
    float pathRoughness;
    vec3 refrDir;
    vec3 endpoint;
    vec3 endpointGeometryNormal;
    vec3 endpointMacroNormal;
    vec3 endpointDiffuseAlbedo;
    float endpointRoughness;
    vec3 endpointLight;
    vec3 transmittance;
    bool endpointValid;
    bool environment;
};

struct FirstBounceData {
    vec3 p, macro_n, geometry_n, micro_n, rd_o, rd_i, refr_dir;
    vec3 specularAlbedo, diffuseAlbedo, transmissionAlbedo;
    vec3 emission_val, light_surf, absorption;
    float t, roughness, n_i, n_o, t2_ior_adjusted, pathRoughness;
    float reflectionHitDistance;
    vec3 surfaceMotion;
    float motionValid;
    int type, materialID;
};

#endif // DIRT_RT_LIB_RT_RAYTRACE_TYPES_GLSL
