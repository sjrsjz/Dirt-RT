#ifndef DIRT_RT_RAYTRACE_PRIMARY_PASS_GLSL
#define DIRT_RT_RAYTRACE_PRIMARY_PASS_GLSL

// Primary visibility pass and shared surface publication.

#if defined(PRIMARY_GBUFFER_PASS)
void TracePrimaryGBuffer(uvec2 xy, vec3 ro, vec3 rd) {
    uint eyeMedium = cam.flags & 3u;
    bool inside = eyeMedium != 0u;
    vec4 fogColor = eyeMedium == 2u
        ? vec4(0.0, 0.05, 0.075, 0.1) * 5.0 : vec4(0.0, 0.325, 0.295, 0.3);
    vec3 globalEmission = eyeMedium == 2u
        ? vec3(1.0, 0.25, 0.05) * 10.0 : vec3(0.0);

    FirstBounceData fb = initFirstBounceData(ro, rd);
    material surf = newMaterial(vec3(0.0), vec3(0.0), vec2(0.0),
            vec4(1.0, 0.0, 0.0, 0.0), vec3(0.0));

    vec3 hitPosition, hitDirection;
    float t = raycast(ro, rd, hitPosition, hitDirection, !inside, false);
    if (t < -0.5) {
        fb.t = -1.0;
        fb.absorption = inside ? vec3(0.0) : vec3(1.0);
    } else {
        vec4 primaryMotion = getPrimarySurfaceMotion(tmp_Payload);
        fb.surfaceMotion = primaryMotion.xyz;
        fb.motionValid = primaryMotion.w;

        Material surfaceMat = evaluateMaterial(tmp_Payload, ro, rd, 0u);
        vec3 geomN = payload_unpackGeomNormal(tmp_Payload.data);
        vec3 geometryNormal = faceforward(geomN, geomN, rd);
        vec3 macroNormal = surfaceMat.macroNormal;
        int blockID;
        payload_unpackShadow(tmp_Payload.data, blockID);
        surf = materialFromEvaluated(surfaceMat, blockID);
        int materialID = getRelaxMaterialID(tmp_Payload, blockID);
        markRadianceCacheGeometryHit(xy, hitPosition, geometryNormal);

        float nI = inside ? REFRACTIVE_INDEX : 1.0;
        float nO = inside ? 1.0 : REFRACTIVE_INDEX;
        MediumResult medium = evalMedium(t, rd, ro.y, inside,
                fogColor, globalEmission);
        recordFirstBounceGBuffer(hitPosition, ro, macroNormal,
            geometryNormal, macroNormal, surf, materialID, rd, rd, t,
            nI, nO, -1, medium.emission, medium.absorption, fb);
    }

    writePrimarySurfaceGBuffer(xy, fb, surf, ro);
}
#endif

#endif // DIRT_RT_RAYTRACE_PRIMARY_PASS_GLSL
