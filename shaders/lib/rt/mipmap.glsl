#ifndef RT_MIPMAP_GLSL
#define RT_MIPMAP_GLSL

struct RtTextureFootprint {
    vec2 dx;
    vec2 dy;
    float lod;
};

struct RtTextureEllipse {
    vec2 majorUv;
    float majorTexels;
    float minorTexels;
};

vec2 rtTaaJitter(uint frameId) {
    float phase = float((frameId & 0xffffu) + 1u);
    return fract(phase * vec2(0.754877666, 0.569840291)) - 0.5;
}

float rtPixelConeSpread(vec3 corner0, vec3 corner1, vec3 corner2,
        vec2 resolution) {
    vec2 safeResolution = max(resolution, vec2(1.0));
    float nearDistance = max(abs(corner0.z), 1e-4);
    vec2 nearPixelSize = vec2(
        length(corner1 - corner0) / safeResolution.x,
        length(corner2 - corner0) / safeResolution.y);
    return max(nearPixelSize.x, nearPixelSize.y) / nearDistance;
}

float rtMaxSpriteLod(ivec2 baseTextureSize, vec4 atlas) {
    ivec2 spriteTexelSize = max(ivec2(round(
        atlas.zw * vec2(baseTextureSize))), ivec2(1));
    return float(findMSB(min(spriteTexelSize.x, spriteTexelSize.y)));
}

RtTextureEllipse rtTextureEllipse(ivec2 baseTextureSize,
        vec2 gradientX, vec2 gradientY) {
    vec2 textureSize0 = vec2(baseTextureSize);
    vec2 texelX = gradientX * textureSize0;
    vec2 texelY = gradientY * textureSize0;

    // Eigen-decompose A*A^T in texel space, where A maps the unit screen
    // footprint to atlas texels. The singular values are the ellipse axes;
    // unlike max/min(length(dUVdx), length(dUVdy)), this remains correct when
    // the two gradients are sheared rather than aligned with the ellipse.
    float matrix00 = texelX.x * texelX.x + texelY.x * texelY.x;
    float matrix01 = texelX.x * texelX.y + texelY.x * texelY.y;
    float matrix11 = texelX.y * texelX.y + texelY.y * texelY.y;
    float trace = matrix00 + matrix11;
    float discriminant = sqrt(max((matrix00 - matrix11)
        * (matrix00 - matrix11) + 4.0 * matrix01 * matrix01, 0.0));
    float lambdaMajor = max(0.5 * (trace + discriminant), 0.0);
    float lambdaMinor = max(0.5 * (trace - discriminant), 0.0);

    vec2 majorDirectionTexels;
    if (abs(matrix01) > 1e-12) {
        majorDirectionTexels = normalize(vec2(matrix01,
            lambdaMajor - matrix00));
    } else {
        majorDirectionTexels = matrix00 >= matrix11
            ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }

    RtTextureEllipse result;
    result.majorTexels = sqrt(lambdaMajor);
    result.minorTexels = sqrt(lambdaMinor);
    result.majorUv = majorDirectionTexels * result.majorTexels
        / textureSize0;
    return result;
}

float rtFiniteAnisotropyLod(ivec2 baseTextureSize, vec2 gradientX,
        vec2 gradientY) {
    const float MAX_ANISOTROPY = 8.0;
    RtTextureEllipse ellipse = rtTextureEllipse(baseTextureSize,
        gradientX, gradientY);
    float perTapFootprint = max(ellipse.minorTexels,
        ellipse.majorTexels / MAX_ANISOTROPY);
    return log2(max(perTapFootprint, 1.0));
}

vec2 rtWrapAtlasCoordinate(vec2 uv, vec4 atlas) {
    return atlas.xy + fract((uv - atlas.xy) / atlas.zw) * atlas.zw;
}

// Manual finite-anisotropy filtering is required because Vulkanite currently
// creates the RT atlas sampler with maxAnisotropy=1. A gradient instruction by
// itself would therefore still choose the major-axis mip and reproduce the old
// grazing-angle blur. Split the long axis into up to eight isotropic samples,
// each evaluated at the short-axis mip.
vec4 rtSampleAnisotropic(sampler2D textureSampler, vec2 uv, vec4 atlas,
        ivec2 baseTextureSize, RtTextureFootprint footprint,
        bool wrapWithinAtlasBox) {
    const int MAX_TAPS = 8;
    RtTextureEllipse ellipse = rtTextureEllipse(baseTextureSize,
        footprint.dx, footprint.dy);
    float minorLength = max(ellipse.minorTexels, 1.0);
    int tapCount = int(ceil(clamp(ellipse.majorTexels / minorLength,
        1.0, float(MAX_TAPS))));
    // If the ellipse exceeds MAX_TAPS:1, raise the per-tap mip so the long
    // direction remains band-limited instead of aliasing.
    float perTapLength = max(minorLength,
        ellipse.majorTexels / float(tapCount));
    float sampleLod = clamp(log2(perTapLength), 0.0,
        rtMaxSpriteLod(baseTextureSize, atlas));

    vec4 result = vec4(0.0);
    for (int tap = 0; tap < MAX_TAPS; ++tap) {
        if (tap >= tapCount) break;
        float offset = (float(tap) + 0.5) / float(tapCount) - 0.5;
        vec2 sampleUv = uv + ellipse.majorUv * offset;
        if (wrapWithinAtlasBox)
            sampleUv = rtWrapAtlasCoordinate(sampleUv, atlas);
        result += textureLod(textureSampler, sampleUv, sampleLod);
    }
    return result / float(tapCount);
}

// LabPBR's G/B/A channels do not form one continuously filterable color:
// G contains conductor IDs, B switches between porosity and SSS at 64/65,
// and A=255 is a no-emission sentinel. Mip generation and bilinear filtering
// can turn two valid endpoint values into a third, unrelated material. Keep
// roughness (R) anisotropically filtered, but select the remaining semantic
// channels from one exact base-level texel. Camera jitter then resolves a
// sub-pixel material boundary stochastically instead of inventing a closure.
vec4 rtFetchAtlasSemanticTexel(sampler2D textureSampler, vec2 uv,
        vec4 atlas, ivec2 baseTextureSize) {
    ivec2 spriteOrigin = ivec2(round(atlas.xy * vec2(baseTextureSize)));
    ivec2 spriteSize = max(ivec2(round(atlas.zw
        * vec2(baseTextureSize))), ivec2(1));
    vec2 safeExtent = max(atlas.zw, vec2(1e-12));
    vec2 local = fract((uv - atlas.xy) / safeExtent);
    ivec2 localTexel = min(ivec2(floor(local * vec2(spriteSize))),
        spriteSize - ivec2(1));
    return texelFetch(textureSampler, spriteOrigin + localTexel, 0);
}

vec4 rtSampleLabPbrSpecular(sampler2D textureSampler, vec2 uv,
        vec4 atlas, ivec2 baseTextureSize, RtTextureFootprint footprint) {
    vec4 filtered = rtSampleAnisotropic(textureSampler, uv, atlas,
        baseTextureSize, footprint, true);
    vec4 semantic = rtFetchAtlasSemanticTexel(
        textureSampler, uv, atlas, baseTextureSize);
    return vec4(filtered.r, semantic.gba);
}

vec3 rtCameraRayDirection(vec2 pixel, vec2 launchSize,
        vec3 corner0, vec3 corner1, vec3 corner2, vec3 corner3,
        mat4 viewInverse) {
    vec2 p = pixel / max(launchSize, vec2(1.0));
    vec3 target = mix(mix(corner0, corner2, p.y),
        mix(corner1, corner3, p.y), p.x);
    return normalize((viewInverse * vec4(target, 0.0)).xyz);
}

vec3 rtIntersectDifferentialPlane(vec3 rayOrigin, vec3 rayDirection,
        vec3 planePoint, vec3 planeNormal) {
    float denominator = dot(planeNormal, rayDirection);
    float safeDenominator = abs(denominator) > 1e-7
        ? denominator : (denominator < 0.0 ? -1e-7 : 1e-7);
    float distance = dot(planeNormal, planePoint - rayOrigin)
        / safeDenominator;
    return rayOrigin + rayDirection * distance;
}

// Exact primary-ray footprint. Neighbouring camera rays are intersected with
// the tangent plane at the hit, then transformed by the triangle's dUV/dP
// Jacobian. The anisotropic sampler can therefore preserve the short axis of a
// grazing footprint instead of turning its long axis into an isotropic mip.
RtTextureFootprint rtPrimaryTextureFootprint(ivec2 baseTextureSize,
        vec4 atlas, vec3 hitPosition, vec3 geometryNormal,
        vec3 gradientU, vec3 gradientV, vec2 pixel, vec2 launchSize,
        vec3 corner0, vec3 corner1, vec3 corner2, vec3 corner3,
        mat4 viewInverse, vec2 jitter) {
    vec3 rayOrigin = viewInverse[3].xyz;
    vec3 directionX = rtCameraRayDirection(pixel + jitter + vec2(1.0, 0.0),
        launchSize, corner0, corner1, corner2, corner3, viewInverse);
    vec3 directionY = rtCameraRayDirection(pixel + jitter + vec2(0.0, 1.0),
        launchSize, corner0, corner1, corner2, corner3, viewInverse);
    vec3 deltaX = rtIntersectDifferentialPlane(rayOrigin, directionX,
        hitPosition, geometryNormal) - hitPosition;
    vec3 deltaY = rtIntersectDifferentialPlane(rayOrigin, directionY,
        hitPosition, geometryNormal) - hitPosition;

    RtTextureFootprint result;
    result.dx = vec2(dot(gradientU, deltaX), dot(gradientV, deltaX));
    result.dy = vec2(dot(gradientU, deltaY), dot(gradientV, deltaY));
    result.lod = clamp(rtFiniteAnisotropyLod(baseTextureSize,
        result.dx, result.dy), 0.0, rtMaxSpriteLod(baseTextureSize, atlas));
    return result;
}

// Secondary rays need propagated ray differentials to reconstruct an exact
// ellipse. The current payload does not carry them, so use a finite-anisotropy
// ray-cone footprint. Unlike the old 1/|N.V| isotropic LOD this does not blur
// both axes by the grazing-angle major axis, and it contains no bounce-count
// multiplier unrelated to the actual path.
RtTextureFootprint rtSecondaryTextureFootprint(ivec2 baseTextureSize,
        vec4 atlas, float coneWidth, vec3 rayDirection,
        vec3 geometryNormal, vec3 tangent, vec3 gradientU,
        vec3 gradientV) {
    float minorWorld = max(coneWidth, 0.0);
    float cosine = max(abs(dot(rayDirection, geometryNormal)), 1e-4);
    float majorWorld = minorWorld / cosine;
    vec3 minorDirection = cross(rayDirection, geometryNormal);
    if (dot(minorDirection, minorDirection) < 1e-12)
        minorDirection = tangent;
    minorDirection = normalize(minorDirection);
    vec3 majorDirection = normalize(cross(geometryNormal, minorDirection));
    vec3 deltaMinor = minorDirection * minorWorld;
    vec3 deltaMajor = majorDirection * majorWorld;

    RtTextureFootprint result;
    result.dx = vec2(dot(gradientU, deltaMinor),
        dot(gradientV, deltaMinor));
    result.dy = vec2(dot(gradientU, deltaMajor),
        dot(gradientV, deltaMajor));
    result.lod = clamp(rtFiniteAnisotropyLod(baseTextureSize,
        result.dx, result.dy), 0.0, rtMaxSpriteLod(baseTextureSize, atlas));
    return result;
}

#endif // RT_MIPMAP_GLSL
