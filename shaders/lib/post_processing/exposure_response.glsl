#ifndef EXPOSURE_RESPONSE_GLSL
#define EXPOSURE_RESPONSE_GLSL

// Exposure calibration and adaptation; statistics live in the compute pass.
float applyExposureCurve(float x, float k) {
    if (k <= 0.0) return x;
    return log(k + exp(x)) - log(1.0 + k);
}

// 曝光曲线导数: f'(x) = e^x / (k + e^x) = 1 / (1 + k·e^(-x))
float exposureCurveDerivative(float x, float k) {
    if (k <= 0.0) return 1.0;
    return 1.0 / (1.0 + k * exp(-x));
}

// 牛顿迭代求逆: 找到 linear_L 使得 tonemap(curve(linear_L)) = expected
// 建模完整管线: linear → exposure_curve → tonemap → sRGB
float solveBaseline(float expected) {
    float k = EXPOSURE_CURVE_K;
    float x = expected;

    for (int i = 0; i < 8; i++) {
        float curved = applyExposureCurve(x, k);
        float f_x = TonyMcMapface_LumaApprox(curved) - expected;

        // 链式法则: d/dx tonemap(curve(x)) = tonemap'(curve) · curve'(x)
        float t_deriv = TonyMcMapface_LumaApprox_Deriv(curved);
        float c_deriv = exposureCurveDerivative(x, k);
        float f_prime_x = t_deriv * c_deriv;

        x = x - f_x / max(f_prime_x, 1e-6);
    }
    return max(x, 0.0);
}

// 将用户自定义的 sRGB 纸白亮度转换为线性空间
float convertPaperWhiteToLinear(float sRGB_paperWhite) {
    if (sRGB_paperWhite <= 0.04045) {
        return sRGB_paperWhite / 12.92;
    } else {
        return pow((sRGB_paperWhite + 0.055) / 1.055, 2.4);
    }
}

float calculateExposure(float avgLuminance) {
    float baseline = solveBaseline(convertPaperWhiteToLinear(DISPLAY_PAPER_WHITE_LUMINANCE));
    float baseExposure = baseline / max(avgLuminance, 1e-20);
    float exposureCorrection = pow(100.0 / DISPLAY_MAX_LUMINANCE, 0.7);
    return baseExposure * exposureCorrection;
}

// Estimate the physical pupil response from adapting luminance. The
// Stanley-Davies form uses a 20 degree adapting field (100*pi square degrees).
// Shader radiance is converted to photometric luminance with the standard
// daylight efficacy used by calibrated HDR radiance data.
float calculatePupilExposure(float sceneLuminance) {
    const float REFERENCE_PUPIL_DIAMETER_MM = 4.0;
    const float DAYLIGHT_LUMINOUS_EFFICACY = 179.0;
    const float ADAPTING_FIELD_AREA_DEG2 = 100.0 * PI;

    float diameterA = clamp(float(PUPIL_MIN_DIAMETER_MM), 1.0, 10.0);
    float diameterB = clamp(float(PUPIL_MAX_DIAMETER_MM), 1.0, 10.0);
    float minDiameter = min(diameterA, diameterB);
    float maxDiameter = max(diameterA, diameterB);

    float adaptingLuminance = max(sceneLuminance * DAYLIGHT_LUMINOUS_EFFICACY, 1e-8);
    float x = pow(adaptingLuminance * ADAPTING_FIELD_AREA_DEG2 / 846.0, 0.41);
    float diameter = 7.75 - 5.75 * x / (x + 2.0);
    diameter = clamp(diameter, minDiameter, maxDiameter);

    float diameterRatio = diameter / REFERENCE_PUPIL_DIAMETER_MM;
    return diameterRatio * diameterRatio;
}

float adaptExposureState(float currentValue, float targetValue, float speed) {
    return exp2(mix(
        log2(max(currentValue, 1e-20)),
        log2(max(targetValue, 1e-20)),
        1.0 - exp2(-max(dTime_global, 0.0) * speed * LOG2_E)
    ));
}

#endif
