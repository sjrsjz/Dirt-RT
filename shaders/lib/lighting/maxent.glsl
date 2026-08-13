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
// 数据约定:
//   - 嵌入表示 vec4 encoded : xyz = 方向向量 v, w = 总能量 ω = |v| + I
//   - 原始光照 L = (v, I)，I >= 0 为各向同性底光强度
//
// 核心性质:
//   - 线性嵌入空间下 T* 算子退化为普通向量累加
//   - 最终辐照度重建基于最大熵分布闭型逼近，误差 < 0.4%
//   - 所有算法严格无偏，仅在数值边界做最小保护
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

// 计算组合方向参数 kappa ∈ [0, 1)
// 输入: len_v = |v|, omega = ω
// 对于 n=3 的闭式解
float maxent_kappa(float len_v, float omega) {
    // 防止退化: omega 极小或 rho >= 1
    if (omega < 1e-8) return 0.0;
    float rho = min(len_v / omega, 0.98); // 留出微小非奇异空间
    // ρ = |v|/ω
    float sqrt_term = sqrt(max(0.0, 4.0 - 3.0 * rho * rho));
    return (3.0 * rho) / (2.0 + sqrt_term);
}


// ------------------------------------------------------------
// 标量方差
// ------------------------------------------------------------
float maxent_variance(vec4 encoded) {
    float v2 = dot(encoded.xyz, encoded.xyz);
    float omega2 = encoded.w * encoded.w;
    float variance = (2.0 * omega2 + encoded.w * sqrt(max(4.0 * omega2 - 3.0 * v2, 0.0))) / 3.0 - 0.5 * v2;
    return max(0.0, variance);
}

// ------------------------------------------------------------
// 径向能量不确定度方差
// ------------------------------------------------------------
float maxent_radial_variance(vec4 encoded) {
    float kappa = maxent_kappa(length(encoded.xyz), encoded.w);
    float kappa_sq = kappa * kappa;
    float omega2 = encoded.w * encoded.w;
    float kappa2_3 = kappa_sq + 3.0;
    kappa2_3 *= kappa2_3;
    float radial_variance = omega2 / kappa2_3 * (3.0 + 6.0 * kappa_sq - kappa_sq * kappa_sq);
    return max(0.0, radial_variance);
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
// 本征方向标准差 (n=3, 总体)
// σ_⊥ = 2ω√(1-κ²)/(3+κ²),  σ_∥ = 2ω√(1+κ²)/(3+κ²)
// 返回 vec2(σ_⊥, σ_∥)
// ------------------------------------------------------------
vec2 maxent_eigen_std(float omega, float kappa) {
    float k2 = kappa * kappa;
    float denom = 3.0 + k2;
    float two_omega = 2.0 * omega;
    return vec2(sqrt(max(0.0, 1.0 - k2)), sqrt(1.0 + k2)) * two_omega / denom;
}

// ------------------------------------------------------------
// 从已存储的标量估计量方差提取径向估计量方差
// stored_var 为 swap2 预滤波后的 Var_scalar/N_eff
// Var(|X|)/N_eff = stored_var × [3+6κ²-κ⁴] / [4(3-κ²)]
// ------------------------------------------------------------
float maxent_radial_est_var_from_scalar(float stored_scalar_var, float kappa) {
    float k2 = kappa * kappa;
    float ratio = (3.0 + 6.0 * k2 - k2 * k2) / max(4.0 * (3.0 - k2), 1e-8);
    return stored_scalar_var * ratio;
}

// ------------------------------------------------------------
// 精确 Bures-Wasserstein 距离平方 (完全解析解, n=3)
//
// 抛弃同轴近似，支持任意夹角状态。
// d_B² = |Δv|² + Tr(Σ₁) + Tr(Σ₂) - 2 Tr( (Σ₁^1/2 Σ₂ Σ₁^1/2)^1/2 )
// 
// 解析降维原理：由于秩-1 摄动的协方差矩阵，开方迹操作在
// v1 和 v2 张成的 2D 子空间内封闭，降维为关于夹角余弦 c 的代数式：
// Tr(交叉) = σ_⊥,₁*σ_⊥,₂ + √((σ_⊥,₁*σ_∥,₂ + σ_⊥,₂*σ_∥,₁)² + c²(σ_∥,₁²-σ_⊥,₁²)(σ_∥,₂²-σ_⊥,₂²))
//
// 性能：无复杂的特征值分解，完全由多项式和单次 sqrt 构成。
// ------------------------------------------------------------
float maxent_bures_distance_sq(vec4 enc1, float kappa1, vec4 enc2, float kappa2) {
    // 获取垂直与平行标准差 (x = σ_⊥, y = σ_∥)
    vec2 std1 = maxent_eigen_std(enc1.w, kappa1);
    vec2 std2 = maxent_eigen_std(enc2.w, kappa2);

    vec3 v1 = enc1.xyz;
    vec3 v2 = enc2.xyz;
    vec3 delta_v = v1 - v2;

    // 分别计算两分布协方差矩阵的迹: Tr(Σ) = 2σ_⊥² + σ_∥²
    float tr1 = 2.0 * std1.x * std1.x + std1.y * std1.y;
    float tr2 = 2.0 * std2.x * std2.x + std2.y * std2.y;

    // 计算两分布均值方向的夹角余弦平方: c² = (v1·v2)² / (|v1|²|v2|²)
    float len1_sq = dot(v1, v1);
    float len2_sq = dot(v2, v2);
    
    float c_sq = 0.0;
    // 保护除零，若极小则 c_sq 退化为 0 (正交)
    if (len1_sq > 1e-16 && len2_sq > 1e-16) {
        float dot_v = dot(v1, v2);
        c_sq = min(1.0, (dot_v * dot_v) / (len1_sq * len2_sq));
    }

    // --- 开始 2x2 降维迹公式求值 ---
    // 变量映射以对齐数学公式: a = σ_⊥, b = σ_∥
    
    // 交叉混合项: a1*b2 + a2*b1
    float cross_ab = std1.x * std2.y + std2.x * std1.y;
    
    // 协方差各向异性强度: b² - a² (即 σ_∥² - σ_⊥²)
    float diff1 = std1.y * std1.y - std1.x * std1.x;
    float diff2 = std2.y * std2.y - std2.x * std2.x;

    // 2D 子空间内的开方迹: T_2D = √((a1*b2 + a2*b1)² + c²(b1²-a1²)(b2²-a2²))
    float T2D_sq = cross_ab * cross_ab + c_sq * diff1 * diff2;
    float T2D = sqrt(max(0.0, T2D_sq)); // max(0.0) 防止浮点精度导致的负数

    // 总体交叉迹: 三维空间的第三轴贡献了纯标量 a1*a2
    float cross_trace = std1.x * std2.x + T2D;

    // 最终 Bures 距离公式: d_B² = |μ1-μ2|² + Tr(Σ1) + Tr(Σ2) - 2 * Tr_cross
    float bw_sq = dot(delta_v, delta_v) + tr1 + tr2 - 2.0 * cross_trace;
    
    // 保护截断输出
    return max(0.0, bw_sq);
}

// ------------------------------------------------------------
// 特化 Bures-Wasserstein 距离平方 (预计算 eigen_std)
//
// 与 maxent_bures_distance_sq 完全等价，但接收预计算的 eigen_std
// (σ_⊥, σ_∥) = maxent_eigen_std(omega, kappa)，跳过 4 次 sqrt。
// 调用方在共享内存加载阶段批量预计算 eigen_std，采样循环中直接
// 查表使用，大幅降低 atrous 内核的 ALU 调度压力。
// ------------------------------------------------------------
float maxent_bures_distance_sq_precomputed(vec3 v1, vec2 std1, vec3 v2, vec2 std2) {
    vec3 delta_v = v1 - v2;

    // 分别计算两分布协方差矩阵的迹: Tr(Σ) = 2σ_⊥² + σ_∥²
    float tr1 = 2.0 * std1.x * std1.x + std1.y * std1.y;
    float tr2 = 2.0 * std2.x * std2.x + std2.y * std2.y;

    // 计算两分布均值方向的夹角余弦平方: c² = (v1·v2)² / (|v1|²|v2|²)
    float len1_sq = dot(v1, v1);
    float len2_sq = dot(v2, v2);

    float c_sq = 0.0;
    if (len1_sq > 1e-16 && len2_sq > 1e-16) {
        float dot_v = dot(v1, v2);
        c_sq = min(1.0, (dot_v * dot_v) / (len1_sq * len2_sq));
    }

    // 交叉混合项: a1*b2 + a2*b1   (a = σ_⊥, b = σ_∥)
    float cross_ab = std1.x * std2.y + std2.x * std1.y;

    // 协方差各向异性强度: b² - a²
    float diff1 = std1.y * std1.y - std1.x * std1.x;
    float diff2 = std2.y * std2.y - std2.x * std2.x;

    // 2D 子空间内的开方迹
    float T2D_sq = cross_ab * cross_ab + c_sq * diff1 * diff2;
    float T2D = sqrt(max(0.0, T2D_sq));

    // 总体交叉迹
    float cross_trace = std1.x * std2.x + T2D;

    // 最终 Bures 距离公式
    float bw_sq = dot(delta_v, delta_v) + tr1 + tr2 - 2.0 * cross_trace;

    return max(0.0, bw_sq);
}


// 计算最大熵分布的自然参数 (θ, β) 用于散度计算
// 返回 vec4(theta.xyz, beta), 其中 θ = ( (3+κ²)² / (4 ω² (1-κ²)) ) * v
// 如果 ω 太小或 κ 趋近 1 会导致 β 发散, 调用方需注意上限
vec4 maxent_theta_beta(vec4 encoded) {
    vec3 v = encoded.xyz;
    float omega = encoded.w;
    float len_v = length(v);

    float kappa = maxent_kappa(len_v, omega);
    float kappa_sq = kappa * kappa;
    float one_minus_kappa_sq = max(0.0, 1.0 - kappa_sq);
    // 避免除零: one_minus_kappa_sq 很小则参数很大，但 kappa 被限制在 0.999999，分母仍安全
    float denom = omega * one_minus_kappa_sq;
    float three_plus_kappa_sq = 3.0 + kappa_sq;

    // β = (3+κ²) / (ω * (1-κ²))
    float beta = three_plus_kappa_sq / max(denom, 1e-20);

    // θ = ((3+κ²)² / (4 ω² (1-κ²))) * v / ? 见文档: θ = (3+κ²)²/(4 ω² (1-κ²)) * v
    // 等价于 beta * (3+κ²)/(4 ω) * v
    float theta_scale = beta * three_plus_kappa_sq / (4.0 * max(omega, 1e-20));
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

    // 散度公式: cross_term - 2n, 对于 n=3 常数为 6.0
    float d = cross_term - 6.0;

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
    return sqrt(W_eff * D_J * sample1.w * sample2.w * (2.0 / 3.0));
}

float maxent_normalized_distance_fast(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = maxent_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);

    // 调和有效样本数 (Wald Effective Sample Size)
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);

    float v1 = maxent_estimator_variance(sample1, N1);
    float v2 = maxent_estimator_variance(sample2, N2);
    return sqrt(W_eff * D_J * sample1.w * sample2.w * (2.0 / 3.0) / max(v1 + v2, 1e-6));
}

// ------------------------------------------------------------
// 漫反射辐照度重建 (核心)
// 基于三维最大熵分布的半球余弦投影解析逼近
// ------------------------------------------------------------
float maxent_irradiance(vec4 encoded, vec3 n)
{
    float omega = encoded.w;
    vec3 v = encoded.xyz;
    if (omega <= 1e-6) return 0.0;

    float lenV = length(v);
    if (lenV <= 1e-6) return 0.25 * omega;

    float rho = clamp(lenV / omega, 0.0, 1.0);
    float kappa = (3.0 * rho) / 
        (2.0 + sqrt(max(0.0, 4.0 - 3.0 * rho * rho)));
    float mu = clamp(dot(v / lenV, n), -1.0, 1.0);

    // kappa = 1 时一般闭式在 mu = 0 处呈 0/0，直接采用其连续极限。
    if (1.0 - kappa <= 1e-5)
        return omega * max(mu, 0.0);

    float k2 = kappa * kappa;
    float mu2 = mu * mu;
    float d = max(1e-12, 1.0 - k2 + k2 * mu2);
    float d32 = d * sqrt(d);
    float nSym = 3.0 + 6.0 * k2 * (-1.0 + 2.0 * mu2)
        + k2 * k2 * (3.0 - 12.0 * mu2 + 8.0 * mu2 * mu2);
    float e = (nSym + 8.0 * kappa * mu * d32)
        / (4.0 * (3.0 + k2) * d32);
    return omega * max(e, 0.0);
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
    float norm = 3.0 * pow(max(0.0, 1.0 - k2), 3.0)
            / (4.0 * PI * (3.0 + k2));
    float d2 = d * d;
    return norm / max(1e-6, d2 * d2);
}

// ------------------------------------------------------------
// MaxEnt 指导采样
// ------------------------------------------------------------
vec3 sample_maxent_guiding(vec3 axis, float kappa, vec2 xi)
{
    vec3 T = normalize(cross(abs(axis.y) < 0.99999 ? vec3(0, 1, 0) : vec3(1, 0, 0), axis));
    vec3 B = cross(axis, T);

    float mu;
    if (kappa < 1e-4) {
        mu = 1.0 - 2.0 * xi.x;
    } else {
        float a = 1.0 / pow(1.0 + kappa, 3.0);
        float b = 1.0 / pow(max(1e-4, 1.0 - kappa), 3.0);
        float invCube = mix(a, b, xi.x);
        float t = pow(invCube, -1.0 / 3.0);
        mu = clamp((1.0 - t) / kappa, -1.0, 1.0);
    }

    float phi = 2.0 * PI * xi.y;
    float s = sqrt(max(0.0, 1.0 - mu * mu));
    return mu * axis + s * (cos(phi) * T + sin(phi) * B);
}
#endif // MAXENT_GLSL
