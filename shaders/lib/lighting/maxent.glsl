// MIT License
//
// Copyright (c) 2026 sjrsjz (https://github.com/sjrsjz)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MAXENT_GLSL
#define MAXENT_GLSL
// ============================================================
// Four-parameter Maximum Entropy (MaxEnt) 光照库
// ============================================================
//
// 基于同构凸锥编码的光照表示、合成、降噪重建全套工具。
// 方向能量采用与一阶矩直接匹配的三次倒数闭包：
//
//   p(u) = (1-kappa^2)^2 / (4*pi*(1-kappa*dot(axis,u))^3)
//
// 因而 kappa = |v| / omega。线性矩接口 (v, omega) 与缓冲布局不变。
// 数据约定:
//   - 嵌入表示 vec4 encoded : xyz = 方向向量 v, w = 总能量 ω = |v| + I
//   - 原始光照 L = (v, I)，I >= 0 为各向同性底光强度
//
// 核心性质:
//   - 线性嵌入空间下 T* 算子退化为普通向量累加
//   - 方向闭包严格保持总能量和一阶方向矩，并保持非负
//   - Lambert 半球余弦查询和方向采样均有解析形式
//
// 命名空间前缀: maxent_
// 性能策略: 无超前clamp, 多使用内联与代数化简, 尽量避免分支
// ============================================================

// ------------------------------------------------------------
// MC 样本编码 (入射辐射率探针)
// ------------------------------------------------------------
//
// 约定变更 (better-denoiser-dev):
//   漫反射第一跳不再携带 BRDF 权重 (bsdf_weight = guideWeight only),
//   MaxEnt 编码接收的是纯入射辐射率 (incident radiance), 而非 BRDF 调制后的
//   出射辐射率。BSDF 评估延迟至 composite (fog.glsl) 统一施加。
//
//   这保证了 MaxEnt 状态对任意材质 (包括纯金属, S.x=1) 都编码完整的入射光场,
//   路径引导对全材质有效, 且消除了除以极小 Cd 的数值不稳定性。
//
//   对单样本 (1 spp):  v = dir * L,  ω = L  (锥边界, I=0)
//   其中 L 为入射辐射率标量 (luminance), 不包含任何 BSDF 调制。

// 编码单条射线探针的入射辐射率为 MaxEnt 嵌入表示 (锥边界态)
// direction: 射线入射方向 (归一化)
// radiance:  入射辐射率标量 (luminance of incident radiance)
vec4 maxent_encode_sample(vec3 direction, float radiance) {
    return vec4(normalize(direction) * radiance, radiance);
}

// 编码单条射线探针的入射辐射率为 MaxEnt 嵌入表示 (RGB 输入版本)
// incident: 入射辐射率 RGB
// direction: 射线入射方向 (归一化)
// 返回 (v, ω) = (dir * Y, Y) 其中 Y = luminance(incident)
vec4 maxent_encode_probe_rgb(vec3 incident, vec3 direction) {
    float Y = dot(incident, vec3(0.25, 0.5, 0.25));
    return vec4(normalize(direction) * Y, Y);
}

// ------------------------------------------------------------
// 线性合成 (累加器接口)
// ------------------------------------------------------------
// 因为 T* 在嵌入空间就是简单线性混合，直接使用乘加即可。
// 下列函数仅提供方便的语义封装。

// 两样本加权混合 (返回混合后的嵌入表示)
vec4 maxent_mix(vec4 a, float weight_a, vec4 b, float weight_b) {
    return a * weight_a + b * weight_b;
}

// 累加一个样本到现有均值 (常用于时域递归)
vec4 maxent_accumulate(vec4 accum, vec4 new_sample, float sample_weight) {
    // 线性累加
    return accum + new_sample * sample_weight;
}


// ------------------------------------------------------------
// 最大熵统计特征提取
// ------------------------------------------------------------

// 计算方向参数 kappa ∈ [0, 1)。该闭包的一阶矩恒等式为
// E[u] = kappa * axis，因此无需非线性反演。
float maxent_kappa(float len_v, float omega) {
    if (omega < 1e-8) return 0.0;
    return clamp(len_v / omega, 0.0, 1.0 - 1e-6);
}


// ------------------------------------------------------------
// 标量方差
// ------------------------------------------------------------
float maxent_variance(vec4 encoded) {
    float v2 = dot(encoded.xyz, encoded.xyz);
    float omega2 = encoded.w * encoded.w;
    // E[R^2] = (3+kappa^2)*omega^2/2 and |v|^2=kappa^2*omega^2.
    return max(0.0, 1.5 * omega2 - 0.5 * v2);
}

// ------------------------------------------------------------
// 径向能量不确定度方差
// ------------------------------------------------------------
float maxent_radial_variance(vec4 encoded) {
    float kappa = maxent_kappa(length(encoded.xyz), encoded.w);
    float kappa_sq = kappa * kappa;
    float omega2 = encoded.w * encoded.w;
    return max(0.0, 0.5 * (1.0 + kappa_sq) * omega2);
}

// 估计方差 (用于时域累积)
float maxent_estimator_variance(vec4 encoded, float N) {
    return maxent_variance(encoded) / max(N, 1e-6);
}

// 径向估计方差 (用于时域累积)
float maxent_radial_estimator_variance(vec4 encoded, float N) {
    return maxent_radial_variance(encoded) / max(N, 1e-6);
}

// ------------------------------------------------------------
// 从已存储的标量估计量方差提取径向估计量方差
// stored_var 为 swap2 预滤波后的 Var_scalar/N_eff
// Var(|X|)/N_eff = stored_var × (1+κ²)/(3-κ²)
// ------------------------------------------------------------
float maxent_radial_est_var_from_scalar(float stored_scalar_var, float kappa) {
    float k2 = kappa * kappa;
    float ratio = (1.0 + k2) / max(3.0 - k2, 1e-8);
    return stored_scalar_var * ratio;
}

// 计算最大熵分布的自然参数 (θ, β) 用于散度计算
// beta = 2/[omega(1-kappa^2)]，theta = beta*kappa*axis。
// 如果 ω 太小或 κ 趋近 1 会导致 β 发散, 调用方需注意上限
vec4 maxent_theta_beta(vec4 encoded) {
    vec3 v = encoded.xyz;
    float omega = encoded.w;
    if (omega <= 1e-20) {
        return vec4(0.0);
    }
    float len_v = length(v);

    float kappa = maxent_kappa(len_v, omega);
    float kappa_sq = kappa * kappa;
    float one_minus_kappa_sq = max(0.0, 1.0 - kappa_sq);
    float denom = omega * one_minus_kappa_sq;
    float beta = 2.0 / max(denom, 1e-20);
    float theta_scale = beta / max(omega, 1e-20);
    vec3 theta = theta_scale * v;

    return vec4(theta, beta);
}

// ------------------------------------------------------------
// 加权杰弗里斯散度 (Weighted Jeffreys Divergence)
// 用于双边滤波似然度量，值越小表示两状态越相似
// 需要传入两个样本的自然参数 (θ, β) 以避免重复计算
// ------------------------------------------------------------
float maxent_weighted_jeffreys_fast(vec4 sample1, vec4 sample2, vec4 maxent_theta_beta1, vec4 maxent_theta_beta2) {
    // 捕捉退化情况: 任一总能量为零
    if (sample1.w < 1e-20 || sample2.w < 1e-20) {
        // 若一方无能量，散度趋于无穷，返回极大值以拒绝
        return 1e10;
    }

    // 提取每个样本的自然参数 (θ, β)
    vec3 theta1 = maxent_theta_beta1.xyz;
    float beta1 = maxent_theta_beta1.w;
    vec3 theta2 = maxent_theta_beta2.xyz;
    float beta2 = maxent_theta_beta2.w;

    // 提取原始坐标 (v, ω)
    vec3 v1 = sample1.xyz;
    float omega1 = sample1.w;
    vec3 v2 = sample2.xyz;
    float omega2 = sample2.w;

    // 交叉项: β1*ω2 + β2*ω1 - θ1·v2 - θ2·v1
    float cross_term = beta1 * omega2 + beta2 * omega1
            - dot(theta1, v2) - dot(theta2, v1);

    // 径向参考测度的 Gamma 形状为 2，故对称散度常数为 2*2。
    float d = cross_term - 4.0;

    // 数值保护: 理论上 d >= 0, 但因精度可能出现微小负值
    return max(0.0, d);
}

// ------------------------------------------------------------
// 加权杰弗里斯散度 (Weighted Jeffreys Divergence)
// 用于双边滤波似然度量，值越小表示两状态越相似
// ------------------------------------------------------------
float maxent_weighted_jeffreys_divergence(vec4 sample1, vec4 sample2) {
    return maxent_weighted_jeffreys_fast(sample1, sample2, maxent_theta_beta(sample1), maxent_theta_beta(sample2));
}

// ------------------------------------------------------------
// 调和累积加权的杰弗里斯散度 (Harmonically Weighted Jeffreys Divergence)
// D_WJ(L1, L2) = W_eff · D_J(L1, L2)
//   其中 W_eff = (N1 · N2) / (N1 + N2) 为有效时域累积的调和均值
//
// N1, N2: 各样本的时域有效累积帧数 (≈ temporal weight)
// 行为:
//   - 低 SPP 阶段 (N1, N2 均小): W_eff 小 → 散度被压降 → 软包容，快速融合
//   - 高 SPP 阶段 (N1, N2 均大): W_eff 大 → 散度被放大 → 严苛拒绝，防残影
//   - 混合 SPP (一大一小):      W_eff ≈ min(N1,N2) → 信任高置信度侧
// ------------------------------------------------------------
float maxent_weighted_jeffreys_with_N(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = maxent_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);
    return W_eff * D_J;
}

// ------------------------------------------------------------
// MaxEnt 近似圆锥测地线距离 (Fast Riemannian Cone Geodesic)
// 在同向对齐极限下，该测地距离严格塌缩为 |sample1.w - sample2.w| * sqrt(2 W_eff)
// ------------------------------------------------------------
float maxent_distance_fast(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = maxent_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);

    // 调和有效样本数 (Wald Effective Sample Size)
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);

    // 经黎曼锥度规映射，将无量纲散度拉回绝对辐射度尺度
    return sqrt(W_eff * D_J * sample1.w * sample2.w);
}

float maxent_normalized_distance_fast(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = maxent_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);

    // 调和有效样本数 (Wald Effective Sample Size)
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);

    float v1 = maxent_estimator_variance(sample1, N1);
    float v2 = maxent_estimator_variance(sample2, N2);
    return sqrt(W_eff * D_J * sample1.w * sample2.w / max(v1 + v2, 1e-6));
}

// ------------------------------------------------------------
// 漫反射辐照度重建 (核心)
// 三次倒数方向能量密度的精确半球余弦投影
// ------------------------------------------------------------
float maxent_irradiance(vec4 encoded, vec3 n)
{
    float omega = encoded.w;
    vec3 v = encoded.xyz;
    if (omega <= 1e-6) return 0.0;

    float lenV = length(v);
    if (lenV <= 1e-6) return 0.25 * omega;

    float kappa = maxent_kappa(lenV, omega);
    float mu = clamp(dot(v / lenV, n), -1.0, 1.0);
    float oneMinusK2 = (1.0 - kappa) * (1.0 + kappa);
    float kMu = kappa * mu;
    float denominator = sqrt(max(oneMinusK2 + kMu * kMu, 1e-20));
    float response;
    if (kMu < 0.0) {
        float sumTerm = denominator - kMu;
        response = oneMinusK2 * oneMinusK2
            / (4.0 * denominator * sumTerm * sumTerm);
    } else {
        response = (oneMinusK2 + 2.0 * kMu * kMu)
            / (4.0 * denominator) + 0.5 * kMu;
    }
    return omega * response;
}

// ------------------------------------------------------------
// 辅助函数：仅从嵌入表示获取总能量和方向
// ------------------------------------------------------------
float maxent_total_energy(vec4 encoded) {
    return encoded.w;
}

vec3 maxent_direction(vec4 encoded) {
    return encoded.xyz;
}

// ------------------------------------------------------------
// MaxEnt 指导采样权重
// ------------------------------------------------------------
float maxent_guiding_pdf(vec3 wi, vec3 axis, float kappa)
{
    float k2 = kappa * kappa;
    float d = 1.0 - kappa * dot(axis, wi);
    float oneMinusK2 = max(0.0, 1.0 - k2);
    float norm = oneMinusK2 * oneMinusK2 / (4.0 * PI);
    float d2 = d * d;
    return norm / max(1e-30, d2 * d);
}

// ------------------------------------------------------------
// MaxEnt 指导采样
// ------------------------------------------------------------
vec3 sample_maxent_guiding(vec3 axis, float kappa, vec2 xi)
{
    vec3 T = normalize(cross(abs(axis.y) < 0.99999 ? vec3(0, 1, 0) : vec3(1, 0, 0), axis));
    vec3 B = cross(axis, T);

    float mu;
    if (kappa < 1e-6) {
        mu = 2.0 * xi.x - 1.0;
    } else {
        float a = 1.0 / ((1.0 + kappa) * (1.0 + kappa));
        float oneMinusKappa = max(1e-15, 1.0 - kappa);
        float b = 1.0 / (oneMinusKappa * oneMinusKappa);
        float inverseSquare = mix(a, b, xi.x);
        float t = inversesqrt(max(inverseSquare, 1e-30));
        mu = clamp((1.0 - t) / kappa, -1.0, 1.0);
    }

    float phi = 2.0 * PI * xi.y;
    float s = sqrt(max(0.0, 1.0 - mu * mu));
    return mu * axis + s * (cos(phi) * T + sin(phi) * B);
}
#endif // MAXENT_GLSL
