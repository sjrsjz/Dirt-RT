#ifndef DIRT_RT_LIB_RT_RAYTRACE_REFRACTION_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_REFRACTION_GLSL

// Virtual refraction endpoint traversal; carries interface and segment transmission.

// ===========================================================================
// PSR (Primary Surface Replacement) Refractive Chain
// ===========================================================================

PSRResult tracePSRChain(vec3 ro, vec3 rd, vec3 geometryNormal,
    float firstRoughness, bool wasInside, int firstMediumBlockID,
    vec3 firstMediumTint, float firstMediumExtinctionWeight, int baseDepth) {
    PSRResult result;
    result.virtualDist = 0.0;
    result.pathRoughness = 0.0;
    result.refrDir = rd;
    result.endpoint = vec3(0.0);
    result.endpointGeometryNormal = vec3(0.0, 1.0, 0.0);
    result.endpointMacroNormal = vec3(0.0, 1.0, 0.0);
    result.endpointDiffuseAlbedo = vec3(0.0);
    result.endpointRoughness = 1.0;
    result.endpointLight = vec3(0.0);
    result.transmittance = vec3(1.0);
    result.endpointValid = false;
    result.environment = false;

    float r_accum2 = firstRoughness * firstRoughness;
    float firstMediumIor = transportIorFromBlock(firstMediumBlockID);
    float n_camera = wasInside ? firstMediumIor : 1.0;

    vec3 ro_chain = ro;
    vec3 rd_chain = rd;
    bool inside_chain = !wasInside;
    // This is the medium occupied by the segment about to be traced, not the
    // material at its terminal hit. Camera-originated dielectric rays have
    // already crossed the primary interface before entering this function.
    int mediumBlockID = inside_chain ? firstMediumBlockID : 0;
    vec3 mediumTint = firstMediumTint;
    float mediumExtinctionWeight = firstMediumExtinctionWeight;
    vec3 departN = geometryNormal;

    for (int i = 0; i < MAX_REFRACTIVE_BOUNCES; i++) {
        result.refrDir = rd_chain;
        float offsetSign = inside_chain ? -1.0 : 1.0;
        vec3 ro_next, rd_next;
        float t_next = raycast(ro_chain + departN * offsetSign * 0.00025,
                rd_chain, ro_next, rd_next, !inside_chain, false);

        if (t_next < -0.5) {
            result.virtualDist = VPROJDIST_SKY;
            result.environment = true;
            break;
        }

        Material hitMat = evaluateMaterial(tmp_Payload, ro_chain, rd_chain,
            uint(baseDepth + 1 + i));
        int hitBlockID;
        payload_unpackShadow(tmp_Payload.data, hitBlockID);

        // Extinction belongs to the medium occupied by this segment, never to
        // the material at its terminal hit.
        if (inside_chain) {
            result.transmittance *= mediumVolumeTransmittance(t_next,
                mediumBlockID, mediumTint, mediumExtinctionWeight);
        }
        material hitSurf = materialFromEvaluated(hitMat, hitBlockID);

        float n_segment = inside_chain
            ? transportIorFromBlock(mediumBlockID) : 1.0;
        result.virtualDist += t_next * n_camera / n_segment;

        vec3 hitGeomN = payload_unpackGeomNormal(tmp_Payload.data);
        hitGeomN = faceforward(hitGeomN, hitGeomN, rd_chain);

        if (!isTransmissiveBlock(hitBlockID)) {
            result.endpoint = ro_next;
            result.endpointGeometryNormal = hitGeomN;
            result.endpointMacroNormal = hitMat.macroNormal;
            result.endpointDiffuseAlbedo = evaluateNonSpecularAlbedo(
                hitSurf, rd_chain, hitMat.macroNormal)
                * (1.0 - clamp(hitSurf.S.y, 0.0, 1.0));
            result.endpointRoughness = hitSurf.R.x;
            result.endpointLight = max(hitSurf.light, vec3(0.0));
            result.endpointValid = true;
            break;
        }

        // Only interfaces blur the virtual image. The terminal opaque
        // surface's BRDF roughness belongs to diffuse projection, not PSR.
        r_accum2 += hitSurf.R.x * hitSurf.R.x;

        float n_from = inside_chain
            ? transportIorFromBlock(mediumBlockID) : 1.0;
        float n_to = inside_chain
            ? 1.0 : transportIorFromBlock(hitBlockID);
        float interfaceEta = n_from / n_to;
        vec3 next_refract = refract(rd_chain, hitGeomN, interfaceEta);

        if (dot(next_refract, next_refract) <= 0.0) break;

        float interfaceFresnel = clamp(fresnel(-rd_chain, hitGeomN,
            interfaceEta), 0.0, 1.0);
        vec3 interfaceColor = evaluateTransmissionAlbedo(hitSurf)
            * clamp(hitSurf.S.y, 0.0, 1.0);
        result.transmittance *= interfaceColor
            * ((1.0 - interfaceFresnel) * interfaceEta * interfaceEta);

        rd_chain = next_refract;
        result.refrDir = rd_chain;
        ro_chain = ro_next;
        inside_chain = !inside_chain;
        mediumBlockID = inside_chain ? hitBlockID : 0;
        if (inside_chain) {
            mediumTint = hitSurf.Cd;
            mediumExtinctionWeight =
                transportExtinctionWeightFromMaterial(hitSurf);
        } else {
            mediumTint = vec3(1.0);
            mediumExtinctionWeight = 0.0;
        }
        departN = hitGeomN;
    }

    result.pathRoughness = sqrt(r_accum2);
    return result;
}

#endif // DIRT_RT_LIB_RT_RAYTRACE_REFRACTION_GLSL
