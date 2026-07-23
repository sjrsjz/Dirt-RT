#ifndef END_SKY_GLSL
#define END_SKY_GLSL

#include "/lib/fonts/sga.glsl"

#define END_PI  3.141592653589793
#define END_TAU 6.283185307179586

float endHash11(float x) {
    return fract(sin(x * 127.1 + 311.7) * 43758.5453123);
}

float endHash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float endValueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);

    f = f * f * (3.0 - 2.0 * f);

    float n000 = endHash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = endHash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = endHash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = endHash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = endHash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = endHash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = endHash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = endHash31(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);

    return mix(nxy0, nxy1, f.z);
}

float endFbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;

    for (int i = 0; i < 5; ++i) {
        value += amplitude * endValueNoise(p);
        p = p * 2.03 + vec3(7.13, 3.71, 5.91);
        amplitude *= 0.5;
    }

    return value;
}

vec3 endRotateAxisAngle(vec3 p, vec3 axis, float angle) {
    axis = normalize(axis);
    float c = cos(angle);
    float s = sin(angle);
    return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1.0 - c);
}

void endMakeBasis(vec3 n, out vec3 bx, out vec3 by) {
    vec3 helper = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    bx = normalize(cross(helper, n));
    by = normalize(cross(n, bx));
}

void endRotatingRingBasis(
    vec3 baseAxis, vec3 rotationAxis,
    float precession, float spin,
    out vec3 axis, out vec3 bx, out vec3 by
) {
    baseAxis = normalize(baseAxis);
    rotationAxis = normalize(rotationAxis);

    vec3 baseX, baseY;
    endMakeBasis(baseAxis, baseX, baseY);

    axis = endRotateAxisAngle(baseAxis, rotationAxis, precession);
    bx = endRotateAxisAngle(baseX, rotationAxis, precession);
    by = endRotateAxisAngle(baseY, rotationAxis, precession);

    vec3 spunX = bx * cos(spin) + by * sin(spin);
    vec3 spunY = by * cos(spin) - bx * sin(spin);

    bx = spunX;
    by = spunY;
}

vec2 endCylinderCoordinates(vec3 rd, vec3 axis, vec3 bx, vec3 by) {
    float axial = dot(rd, axis);
    vec3 radial = rd - axis * axial;
    float radialLength = max(length(radial), 1e-5);

    float t = 1.0 / radialLength;
    vec3 hit = rd * t;

    float angle = atan(dot(hit, by), dot(hit, bx));
    if (angle < 0.0) angle += END_TAU;

    float height = dot(hit, axis);
    return vec2(angle, height);
}

float endSdfCoverage(float d, float aa) {
    return 1.0 - smoothstep(-aa, aa, d);
}

float endRuneRing(
    vec3 rd,
    vec3 baseAxis,
    vec3 rotationAxis,
    int divisionCount,
    float precessionSpeed,
    float spinSpeed,
    float phase,
    float seed
) {
    float t = time_global;

    vec3 axis, bx, by;
    endRotatingRingBasis(
        baseAxis, rotationAxis,
        t * precessionSpeed,
        t * spinSpeed + phase,
        axis, bx, by
    );

    vec2 q = endCylinderCoordinates(rd, axis, bx, by);

    float count = float(divisionCount);
    float cellSize = END_TAU / count;
    float halfCell = 0.5 * cellSize;

    float cellPosition = q.x / cellSize;
    int cellId = min(int(floor(cellPosition)), divisionCount - 1);

    float angleInCell = q.x - (float(cellId) + 0.5) * cellSize;

    // Local grid coordinate for glyph SDF
    vec2 cp = vec2(q.y, angleInCell) / halfCell;
    cp.y = -cp.y; // Flip Y for SDF glyphs

    // Strict grid boundary mask — clean seams between glyphs
    float cellMask = step(max(abs(cp.x), abs(cp.y)), 0.98);
    float insideStrip = 1.0 - step(halfCell, abs(q.y));

    // Anti-aliasing width — fwidth unavailable in raygen; use fixed small width.
    // SDF coverage inherently provides smooth edges via distance interpolation.
    float aaGlyph = max(0.015 / halfCell, 1e-4);

    // Per-glyph animation: local random timeline prevents sync
    float localTime = t * 0.5 + endHash11(float(cellId) * 13.37 + seed) * 100.0;
    float tick = floor(localTime);

    // Random glyph selection (0-25 → A-Z)
    float randomGlyph = endHash11(float(cellId) * 17.17 + tick * 0.5);
    int glyph = min(int(floor(randomGlyph * 26.0)), 25);

    // Random blank/flicker: ~70% occupied
    float randBlank = endHash11(float(cellId) * 31.73 + tick * 0.1);
    float occupied = step(0.30, randBlank);

    // Render glyph
    float skeletonDist = sdf_sga(glyph, cp * 1.25);
    float glyphSdf = skeletonDist - 0.1;
    float glyphCoverage = endSdfCoverage(glyphSdf, aaGlyph);

    return glyphCoverage * occupied * insideStrip * cellMask;
}

vec3 sampleEndFbmBackground(vec3 rd) {
    rd = normalize(rd);
    float t = time_global;

    vec3 greenCoord = rd * 3.2;

    vec3 greenOffset1 = vec3(t * 0.012, t * 0.015, t * -0.008);
    vec3 greenOffset2 = vec3(-t * 0.01, t * 0.018, t * 0.015);

    float greenBase = endFbm(greenCoord + greenOffset1);
    float greenDetail = endFbm(greenCoord * 2.0 + greenOffset2);
    float greenDensity = greenBase * 0.7 + greenDetail * 0.3;

    float greenMask = smoothstep(0.56, 0.78, greenDensity);

    vec3 purpleScale = vec3(10, 0.4, 10);

    vec3 purpleOffset1 = vec3(t * 0.015, t * 0.035, t * -0.012);
    vec3 purpleOffset2 = vec3(-t * 0.01, t * 0.025, t * 0.018);

    float purpleBase = endFbm(normalize(vec3(rd.x, 0.0, rd.z)) * purpleScale + purpleOffset1);
    float purpleDensity = purpleBase;

    purpleDensity = smoothstep(0.5, 0.75, purpleDensity);

    float equatorMask = exp(-30.0 * abs(atan(clamp(rd.y, -0.999, 0.999)))) * purpleDensity;

    vec3 blackColor = vec3(0.0003, 0.0001, 0.0005);
    vec3 greenColor = vec3(0.0015, 0.022, 0.008);
    vec3 purpleColor = vec3(6.5, 1.2, 10.5);

    vec3 col = blackColor;
    col += greenColor * greenMask;
    col += purpleColor * equatorMask;

    return col;
}

vec3 sampleEndSky(vec3 rd) {
    vec3 col = vec3(0);
    col += sampleEndFbmBackground(rd);

    // Three rune rings at different orientations, speeds, and sizes
    float ring0 = endRuneRing(
            rd,
            normalize(vec3(0.10, 0.96, 0.27)),
            normalize(vec3(0.20, 1.00, 0.10)),
            80, 0.070, 0.051, 0.0, 11.0
        );

    float ring1 = endRuneRing(
            rd,
            normalize(vec3(-0.70, 0.26, 0.66)),
            normalize(vec3(0.00, 1.00, 0.22)),
            48, -0.052, -0.084, 1.7, 37.0
        );

    float ring2 = endRuneRing(
            rd,
            normalize(vec3(0.69, 0.15, 0.71)),
            normalize(vec3(-0.18, 1.00, 0.05)),
            56, 0.036, 0.043, 3.1, 83.0
        );

    // Tint each ring with its own color
    col += vec3(40, 43, 48) * ring0;
    col += vec3(38, 32, 45) * ring1;
    col += vec3(28, 42, 32) * ring2;
    return col;
}

#endif // END_SKY_GLSL
