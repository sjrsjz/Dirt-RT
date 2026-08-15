#ifndef DIRT_RT_RAYTRACE_SCENE_GLSL
#define DIRT_RT_RAYTRACE_SCENE_GLSL

// Ray traversal, material decoding and primary-space helpers.

float raycastMin(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o,
    bool inverse_0, bool isNEE, bool ignoreTransmissive, float tMin) {
    bool inside = !inverse_0;
    payload_packRayCone(payload.data, rtCurrentConeWidth,
        rtCurrentConeSpread);
    payload_packFlags(payload.data, 0.0, inside, false, isNEE,
        ignoreTransmissive);
    payload_packShadow(payload.data, vec3(1.0), 0);
    float tMax = 2048.0;
    traceRayEXT(acc, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, ro, tMin, rd, tMax, 6);
    Payload hitPayload = payload;
    float t = payload_unpackHitDistance(hitPayload.data);
    if (t >= 0.0 && !isNEE)
        rtCurrentConeWidth = fma(t, rtCurrentConeSpread,
            rtCurrentConeWidth);
    ro_o = t >= 0.0 ? ro + rd * t : vec3(0.0);
    rd_o = rd;
    tmp_Payload = hitPayload;
    return t;
}

float raycastMin(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o,
    bool inverse_0, bool isNEE, float tMin) {
    return raycastMin(ro, rd, ro_o, rd_o, inverse_0, isNEE, false,
        tMin);
}

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0, bool isNEE) {
    return raycastMin(ro, rd, ro_o, rd_o, inverse_0, isNEE, 0.0);
}

float raycast(in vec3 ro, in vec3 rd, out vec3 ro_o, out vec3 rd_o, bool inverse_0) {
    return raycast(ro, rd, ro_o, rd_o, inverse_0, false);
}

float raycastIgnoreTransmissive(in vec3 ro, in vec3 rd,
        out vec3 ro_o, out vec3 rd_o, bool inverse_0) {
    return raycastMin(ro, rd, ro_o, rd_o, inverse_0, false, true, 0.0);
}

vec4 getPrimarySurfaceMotion(Payload hitPayload) {
    uint instanceIdx, geometryId, primitiveId;
    payload_unpackQuadIDs(hitPayload.data, instanceIdx, geometryId, primitiveId);
    if ((instanceIdx & ENTITY_INSTANCE_FLAG) == 0u) {
        // w=2 distinguishes static scene geometry from a tracked entity in
        // motion debug views. Temporal passes only test w >= 0.5.
        return vec4(0.0, 0.0, 0.0, 2.0);
    }

    vec2 bary = payload_unpackBarycentrics(hitPayload.data);
    float w0 = 1.0 - bary.x - bary.y;
    uint baseVertex = (primitiveId >> 1u) * 4u;
    uint secondVertex = (primitiveId & 1u) == 0u ? 1u : 2u;
    uint thirdVertex = (primitiveId & 1u) == 0u ? 2u : 3u;
    vec4 m0 = vec4(entityMotionBuffer.vertices[baseVertex].deltaAndValid);
    vec4 m1 = vec4(entityMotionBuffer.vertices[baseVertex + secondVertex].deltaAndValid);
    vec4 m2 = vec4(entityMotionBuffer.vertices[baseVertex + thirdVertex].deltaAndValid);
    vec3 motion = m0.xyz * w0 + m1.xyz * bary.x + m2.xyz * bary.y;
    float valid = min(m0.w, min(m1.w, m2.w));
    return vec4(motion, valid);
}

struct material {
    vec3 Cs;
    vec3 Cd;
    vec2 S;
    vec4 R;
    vec3 light;
};

material newMaterial(vec3 Cs, vec3 Cd, vec2 S, vec4 R, vec3 light) {
    material a;
    a.Cs = Cs;
    a.Cd = Cd;
    a.S = S;
    a.R = R;
    a.light = light;
    return a;
}

Material evaluateMaterial(Payload pld, vec3 rayOrigin, vec3 rd_i,
        uint bounce) {
    // --- Unpack quad data from payload (packed by rchit, no geometryBuffers needed) ---
    vec2 uv = payload_unpackQuadUV(pld.data);
    vec4 atlas = payload_unpackAtlasBox(pld.data);
    vec3 geomN = payload_unpackGeomNormal(pld.data);
    vec3 tangent = payload_unpackTangent(pld.data);
    vec3 tint;
    float skylight;
    payload_unpackQuadExtras(pld.data, tint, skylight);
    int blockID;
    // vec3 _shadow = payload_unpackShadow(pld.data, blockID);
    bool _inside, handedness, _isNEE;
    payload_unpackFlags(pld.data, _inside, handedness, _isNEE);
    float bitangentSign = handedness ? 1.0 : -1.0;
    uint instanceIdx, entityTextureId, primitiveId;
    payload_unpackQuadIDs(pld.data, instanceIdx, entityTextureId, primitiveId);

    // --- TBN ---
    // Interpolated/compressed tangents are not guaranteed to remain
    // perpendicular to the geometric normal.  Gram-Schmidt here prevents the
    // tangent-space XY terms from pushing a normal map below the surface.
    tangent = normalize(tangent - geomN * dot(geomN, tangent));
    // tangent and geomN are now orthonormal, hence their cross is unit length.
    vec3 bitangent = cross(tangent, geomN) * bitangentSign;
    mat3 tbn = mat3(tangent, bitangent, geomN);

    float hitDistance = payload_unpackHitDistance(pld.data);
    vec3 hitPosition = rayOrigin + rd_i * hitDistance;
    vec3 gradientU, gradientV;
    payload_unpackTextureGradients(pld.data, gradientU, gradientV);
    vec3 texturePlaneNormal = cross(gradientU, gradientV);
    texturePlaneNormal = dot(texturePlaneNormal, texturePlaneNormal) > 1e-20
        ? normalize(texturePlaneNormal) : geomN;
    vec2 mipResolution = max(vec2(resolution_global),
            vec2(gl_LaunchSizeEXT.xy));
    float pixelConeSpread = rtPixelConeSpread(cam.corners[0],
            cam.corners[1], cam.corners[2], mipResolution);

    vec2 sampleUV = uv;
    vec4 albedoTex;
    vec4 specularTex;
    vec4 normalTex;
    RtTextureFootprint footprint;

    if (entityTextureId != 0u) {
        uint textureIndex = entityTextureId - 1u;
        ivec2 textureResolution = textureSize(
                entityTextures[nonuniformEXT(textureIndex)], 0);
        vec4 entityAtlas = vec4(0.0, 0.0, 1.0, 1.0);
        if (bounce == 0u) {
            footprint = rtPrimaryTextureFootprint(textureResolution,
                entityAtlas, hitPosition, texturePlaneNormal,
                gradientU, gradientV,
                vec2(gl_LaunchIDEXT.xy), vec2(gl_LaunchSizeEXT.xy),
                cam.corners[0], cam.corners[1], cam.corners[2],
                cam.corners[3], cam.viewInverse);
        } else {
            footprint = rtSecondaryTextureFootprint(textureResolution,
                entityAtlas, rtCurrentConeWidth, rd_i,
                texturePlaneNormal, tangent,
                gradientU, gradientV);
        }
        albedoTex = rtSampleAnisotropic(
            entityTextures[nonuniformEXT(textureIndex)], uv, entityAtlas,
            textureResolution, footprint, false);
        specularTex = vec4(0.0, 0.04, 0.0, 1.0);
        normalTex = vec4(0.5, 0.5, 1.0, 1.0);
    } else {
        vec2 localCoord = getRelativeUV(uv, atlas);

        // --- POM — first hit only ---
        ivec2 textureResolution = textureSize(blockTex, 0);
        if (bounce == 0u) {
            footprint = rtPrimaryTextureFootprint(textureResolution,
                atlas, hitPosition, texturePlaneNormal, gradientU, gradientV,
                vec2(gl_LaunchIDEXT.xy), vec2(gl_LaunchSizeEXT.xy),
                cam.corners[0], cam.corners[1], cam.corners[2],
                cam.corners[3], cam.viewInverse);
        } else {
            footprint = rtSecondaryTextureFootprint(textureResolution,
                atlas, rtCurrentConeWidth, rd_i,
                texturePlaneNormal, tangent,
                gradientU, gradientV);
        }

        #if POM_ENABLED == 1
        if (bounce == 0u) {
            sampleUV = computeParallaxUV(blockTexNormal, localCoord, atlas,
                    rd_i, tbn, footprint.lod);
        } else {
            sampleUV = uv;
        }
        #else
        sampleUV = uv;
        #endif

        albedoTex = rtSampleAnisotropic(blockTex, sampleUV, atlas,
            textureResolution, footprint, true);
        specularTex = rtSampleLabPbrSpecular(blockTexSpecular, sampleUV,
            atlas, textureResolution, footprint);
        normalTex = rtSampleAnisotropic(blockTexNormal, sampleUV, atlas,
            textureResolution, footprint, true);
    }

    albedoTex.rgb = pow(albedoTex.rgb * tint, vec3(2.2));

    Material evaluated = getMaterial(albedoTex, normalTex, specularTex, tbn,
        wetStrength_global, wetness_global, skylight, geomN);
    vec3 geometryNormal = faceforward(geomN, geomN, rd_i);
    evaluated.macroNormal = constrainMappedNormal(evaluated.macroNormal,
        geometryNormal, -rd_i);
    return evaluated;
}

// Check if a block is a transmissive/refractive surface.
// LabPBR standard: translucent blocks are identified by the rendering layer,
// not by any material channel. In our ray-tracing pipeline we conservatively
// classify water, glass and ice explicitly for the PSR chain.
bool isTransmissiveBlock(int blockID) {
    return blockID == BLOCK_WATER || blockID == BLOCK_GLASS
        || blockID == BLOCK_ICE;
}

// Convert Material (from getMaterial) to BSDF material struct
material materialFromEvaluated(Material mat, int blockID) {
    // Precompute block-type flags once (each was compared 3-5× before)
    bool isWater = blockID == BLOCK_WATER;
    bool isGlass = blockID == BLOCK_GLASS;
    bool isIce = blockID == BLOCK_ICE;
    bool isPortal = blockID == BLOCK_PORTAL;
    bool isTransmissive = isWater || isGlass || isIce;

    float metallic = mat.metallic;

    // albedo.a is coverage for alpha-tested geometry. The any-hit shader has
    // already rejected uncovered texels, so using the remaining filtered alpha
    // as physical transmission turns leaf/vine/lily-pad edges into glass.
    // Only explicitly classified blocks may enter the transmission branch.
    float opaqueFraction = isTransmissive ? 0.0 : 1.0;
    opaqueFraction = isPortal ? 0.25 : opaqueFraction;
    float roughness = (isWater || isPortal) ? 0.0 : mat.roughness;
    // For glass/ice this color is volume data, not an interface multiplier.
    // Transmission helpers below keep the dielectric boundary color-neutral
    // and Beer-Lambert applies the tint according to travelled distance.
    vec3 albedo = isWater ? vec3(1.0) : mat.albedo;
    vec3 emission = isPortal ? albedo * (1.0 - opaqueFraction) : mat.emission;
    float specSelector = isTransmissive
        ? 1.0 : mix(opaqueFraction, 1.0, metallic);
    float mediumClass = isWater ? 1.0 : (isGlass ? 2.0 : (isIce ? 3.0 : 0.0));
    // Values >= 1 carry the transmissive medium class. Every BSDF consumer
    // clamps S.y to [0,1], so this reuses the existing FP16 material word
    // without changing the physical transmission selector or adding a read.
    // Preserve the coverage/translucency weight in the otherwise-unused
    // fractional quarter of the FP16 medium code. Rounding still recovers the
    // integer class, while all BSDF selectors clamp the value to one.
    float extinctionWeight = (isGlass || isIce)
        ? clamp(mat.translucent, 0.0, 1.0) : 0.0;
    float transmissionCode = isTransmissive
        ? mediumClass + 0.25 * extinctionWeight
        : (1.0 - opaqueFraction);

    return newMaterial(clamp(mat.F0, 0.0, 1.0), albedo,
        vec2(specSelector, transmissionCode),
        vec4(roughness, opaqueFraction,
            mediumClass, mat.subsurface_scattering),
        emission);
}

int transportBlockFromMaterial(material surf) {
    if (surf.S.y < 0.999) return 0;
    int mediumClass = int(surf.S.y + 0.5);
    return mediumClass == 1 ? BLOCK_WATER
        : (mediumClass == 2 ? BLOCK_GLASS
        : (mediumClass == 3 ? BLOCK_ICE : 0));
}

float transportIorFromBlock(int blockID) {
    if (blockID == BLOCK_WATER) return REFRACTIVE_INDEX;
    if (blockID == BLOCK_ICE) return 1.31;
    if (blockID == BLOCK_GLASS) return GLASS_REFRACTIVE_INDEX;
    return 1.0;
}

float transportIorFromMaterial(material surf) {
    return transportIorFromBlock(transportBlockFromMaterial(surf));
}

float transportExtinctionWeightFromMaterial(material surf) {
    int mediumClass = int(surf.S.y + 0.5);
    return transportBlockFromMaterial(surf) == 0 ? 0.0
        : clamp((surf.S.y - float(mediumClass)) * 4.0, 0.0, 1.0);
}

vec3 evaluateTransmissionAlbedo(material surf) {
    // Explicit dielectric media acquire their color in the volume. Portal's
    // partial transmission is a surface effect and retains its authored tint.
    return transportBlockFromMaterial(surf) != 0 ? vec3(1.0) : surf.Cd;
}

// The compact shadow payload reserves its block-ID byte for four special
// transport classes, so ordinary blocks arrive as ID 0. MaxEnt still needs a
// stable discriminator. The atlas rectangle identifies the sampled sprite
// without another payload slot, SSBO, or image.
uint hashMaxEntMaterialWord(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

int getMaxEntMaterialID(Payload pld, int transportBlockID) {
    vec4 atlas = payload_unpackAtlasBox(pld.data);
    uvec4 a = floatBitsToUint(atlas);
    uint h = hashMaxEntMaterialWord(a.x ^ (a.y * 0x9e3779b9u));
    h = hashMaxEntMaterialWord(h ^ a.z ^ (a.w * 0x85ebca6bu));

    uint transportClass = transportBlockID == BLOCK_WATER ? 1u
        : (transportBlockID == BLOCK_GLASS ? 2u
        : (transportBlockID == BLOCK_PORTAL ? 3u
        : (transportBlockID == BLOCK_ICE ? 4u : 0u)));
    h = hashMaxEntMaterialWord(h ^ (transportClass * 0x27d4eb2du));
    return int(h & 0xffffu);
}

vec3 evaluateNonSpecularAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    // Disney's diffuse/base-color lobe is not pre-multiplied by a view-angle
    // Fresnel term. Its grazing response is part of the Burley diffuse factor.
    return surf.Cd;
}

vec3 evaluateDiffuseAlbedo(material surf, vec3 rd_i, vec3 macroNormal) {
    return evaluateNonSpecularAlbedo(surf, rd_i, macroNormal)
        * (1.0 - clamp(surf.S.y, 0.0, 1.0));
}

// Exact GLSL port of NRD.hlsli's material-factor front end. The specular path consumes
// demodulated specular radiance, so both the fit and its deliberately biased
// stability floors are part of the denoiser contract, not optional tuning.
const float MAXENT_SPEC_NRD_EPS = 1e-6;
const float MAXENT_SPEC_NRD_MATERIAL_FACTOR_MIN_SCALE = 0.02;
const float MAXENT_SPEC_NRD_ROUGHNESS_FACTOR_MIN_SCALE = 0.1;

vec3 maxentNrdEnvironmentTermRtg(vec3 Rf0, float NoV,
    float perceptualRoughness) {
    // Ray Tracing Gems, Chapter 32, Equation 4. The official fit consumes
    // perceptual roughness and squares it internally to obtain GGX alpha.
    float m = clamp(perceptualRoughness * perceptualRoughness, 0.0, 1.0);

    vec4 X = vec4(1.0, NoV, NoV * NoV, NoV * NoV * NoV);
    vec4 Y = vec4(1.0, m, m * m, m * m * m);

    // Written explicitly to preserve HLSL mul(matrix, columnVector) semantics
    // and avoid a row/column-major transpose while porting the NRD constants.
    vec2 m1x = vec2(
            0.99044 * X.x - 1.28514 * X.y,
            1.29678 * X.x - 0.755907 * X.y);
    vec3 m2x = vec3(
            1.0 * X.x + 2.92338 * X.y + 59.4188 * X.w,
            20.3225 * X.x - 27.0302 * X.y + 222.592 * X.w,
            121.563 * X.x + 626.13 * X.y + 316.627 * X.w);
    vec2 m3x = vec2(
            0.0365463 * X.x + 3.32707 * X.y,
            9.0632 * X.x - 9.04756 * X.y);
    vec3 m4x = vec3(
            1.0 * X.x + 3.59685 * X.z - 1.36772 * X.w,
            9.04401 * X.x - 16.3174 * X.z + 9.22949 * X.w,
            5.56589 * X.x + 19.7886 * X.z - 20.2123 * X.w);

    float bias = dot(m1x, Y.xy)
            / max(dot(m2x, Y.xyw), MAXENT_SPEC_NRD_EPS);
    float scale = dot(m3x, Y.xy)
            / max(dot(m4x, Y.xyw), MAXENT_SPEC_NRD_EPS);
    return clamp(Rf0 * scale + vec3(bias), vec3(0.0), vec3(1.0));
}

vec3 evaluateSpecularAlbedo(
    material surf, vec3 rd_i, vec3 macroNormal, float etaRatio
) {
    float NoV = clamp(abs(dot(rd_i, macroNormal)), 0.0, 1.0);
    float transmissionSelector = clamp(surf.S.y, 0.0, 1.0);
    float etaDenominator = max(abs(1.0 + etaRatio), 1e-4);
    float dielectricF0 = (1.0 - etaRatio) / etaDenominator;
    dielectricF0 *= dielectricF0;

    // Match evaluateSurfaceFresnel(): opaque/metallic lobes use LabPBR F0,
    // while transmissive interfaces use IOR Fresnel.
    vec3 Rf0 = mix(surf.Cs * surf.S.x, vec3(dielectricF0),
            transmissionSelector);

    // Dirt RT stores GGX alpha; NRD_MaterialFactors takes perceptual roughness.
    float perceptualRoughness = sqrt(clamp(surf.R.x, 0.0, 1.0));
    vec3 specFactor = maxentNrdEnvironmentTermRtg(
            clamp(Rf0, vec3(0.0), vec3(1.0)), NoV, perceptualRoughness);

    // These two lines intentionally reproduce NRD_MaterialFactors exactly.
    // They bias very dark/rough reflectors, preventing unstable demodulation.
    specFactor *= mix(MAXENT_SPEC_NRD_ROUGHNESS_FACTOR_MIN_SCALE,
            1.0, perceptualRoughness);
    specFactor = mix(vec3(MAXENT_SPEC_NRD_MATERIAL_FACTOR_MIN_SCALE),
            vec3(1.0), specFactor);

    if (any(isnan(specFactor)) || any(isinf(specFactor)))
        return vec3(MAXENT_SPEC_NRD_MATERIAL_FACTOR_MIN_SCALE);
    return clamp(specFactor,
        vec3(MAXENT_SPEC_NRD_MATERIAL_FACTOR_MIN_SCALE), vec3(1.0));
}

vec3 reproject(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - prevRaytracingCamPos;
    vec4 clipPos = rtPrevViewProjection * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

// -----------------------------------------------------------------------------------
// Direct-lighting helpers are defined next to the sun sampler below.
// ---------------------------------------------------------------------------

vec3 GetSpecularDominantDirection(vec3 N, vec3 V, float R) {
    float f = (1.0 - R) * (sqrt(1.0 - R) + R);
    vec3 R0 = reflect(V, N);
    return normalize(mix(N, R0, f));
}


#endif // DIRT_RT_RAYTRACE_SCENE_GLSL
