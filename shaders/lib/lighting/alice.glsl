#ifndef ALICE_GLSL
#define ALICE_GLSL
// ============================================================
// ALICE (Asymmetric Laplace Isomorphic Conic Encoding) 光照库
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
// 命名空间前缀: alice_
// 性能策略: 无超前clamp, 多使用内联与代数化简, 尽量避免分支
// ============================================================

// ------------------------------------------------------------
// 编码/解码
// ------------------------------------------------------------

// 将原始光照样本编码为线性嵌入表示
vec4 alice_encode(vec3 v, float I) {
    // ω = |v| + I
    float omega = length(v) + I;
    return vec4(v, omega);
}

// 编码 1spp 蒙特卡洛样本 (direction, radiance) 为嵌入表示
vec4 alice_encode_sample(vec3 direction, float radiance) {
    // 直接使用方向向量和辐照度作为输入，编码为嵌入表示
    return vec4(normalize(direction) * radiance, radiance);
}

// 从嵌入表示中提取方向向量 v 和各向同性强度 I
// 注意: 可能因数值误差 ω < |v|, 此时将 I 钳制为 0
void alice_decode(vec4 encoded, out vec3 v, out float I) {
    v = encoded.xyz;
    float len_v = length(v);
    // I = ω - |v|, 强制非负
    I = max(0.0, encoded.w - len_v);
}

// ------------------------------------------------------------
// 线性合成 (累加器接口)
// ------------------------------------------------------------
// 因为 T* 在嵌入空间就是简单线性混合，直接使用乘加即可。
// 下列函数仅提供方便的语义封装。

// 两样本加权混合 (返回混合后的嵌入表示)
vec4 alice_mix(vec4 a, float weight_a, vec4 b, float weight_b) {
    return a * weight_a + b * weight_b;
}

// 累加一个样本到现有均值 (常用于时域递归)
vec4 alice_accumulate(vec4 accum, vec4 new_sample, float sample_weight) {
    // 线性累加
    return accum + new_sample * sample_weight;
}


// ------------------------------------------------------------
// 最大熵统计特征提取
// ------------------------------------------------------------

// 计算组合方向参数 kappa ∈ [0, 1)
// 输入: len_v = |v|, omega = ω
// 对于 n=3 的闭式解
float alice_kappa(float len_v, float omega) {
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
float alice_variance(vec4 encoded) {
    float v2 = dot(encoded.xyz, encoded.xyz);
    float omega2 = encoded.w * encoded.w;
    float variance = (2.0 * omega2 + encoded.w * sqrt(max(4.0 * omega2 - 3.0 * v2, 0.0))) / 3.0 - 0.5 * v2;
    return max(0.0, variance);
}

// ------------------------------------------------------------
// 径向能量不确定度方差
// ------------------------------------------------------------
float alice_radial_variance(vec4 encoded) {
    float kappa = alice_kappa(length(encoded.xyz), encoded.w);
    float kappa_sq = kappa * kappa;
    float omega2 = encoded.w * encoded.w;
    float kappa2_3 = kappa_sq + 3.0;
    kappa2_3 *= kappa2_3;
    float radial_variance = omega2 / kappa2_3 * (3.0 + 6.0 * kappa_sq - kappa_sq * kappa_sq);
    return max(0.0, radial_variance);
}

// 估计方差 (用于时域累积)
float alice_estimator_variance(vec4 encoded, float N) {
    return alice_variance(encoded) / max(N, 1e-6);
}

// 径向估计方差 (用于时域累积)
float alice_radial_estimator_variance(vec4 encoded, float N) {
    return alice_radial_variance(encoded) / max(N, 1e-6);
}

// 计算最大熵分布的自然参数 (θ, β) 用于散度计算
// 返回 vec4(theta.xyz, beta), 其中 θ = ( (3+κ²)² / (4 ω² (1-κ²)) ) * v
// 如果 ω 太小或 κ 趋近 1 会导致 β 发散, 调用方需注意上限
vec4 alice_theta_beta(vec4 encoded) {
    vec3 v = encoded.xyz;
    float omega = encoded.w;
    float len_v = length(v);

    float kappa = alice_kappa(len_v, omega);
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
float alice_weighted_jeffreys_fast(vec4 sample1, vec4 sample2, vec4 alice_theta_beta1, vec4 alice_theta_beta2) {
    // 捕捉退化情况: 任一总能量为零
    if (sample1.w < 1e-20 || sample2.w < 1e-20) {
        // 若一方无能量，散度趋于无穷，返回极大值以拒绝
        return 1e10;
    }

    // 提取每个样本的自然参数 (θ, β)
    vec3 theta1 = alice_theta_beta1.xyz;
    float beta1 = alice_theta_beta1.w;
    vec3 theta2 = alice_theta_beta2.xyz;
    float beta2 = alice_theta_beta2.w;

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
float alice_weighted_jeffreys_divergence(vec4 sample1, vec4 sample2) {
    return alice_weighted_jeffreys_fast(sample1, sample2, alice_theta_beta(sample1), alice_theta_beta(sample2));
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
float alice_weighted_jeffreys_with_N(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = alice_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);
    return W_eff * D_J;
}

// ------------------------------------------------------------
// ALICE 近似圆锥测地线距离 (Fast Riemannian Cone Geodesic)
// 在同向对齐极限下，该测地距离严格塌缩为 |sample1.w - sample2.w| * sqrt(2 W_eff)
// ------------------------------------------------------------
float alice_distance_fast(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = alice_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);

    // 调和有效样本数 (Wald Effective Sample Size)
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);

    // 经黎曼锥度规映射，将无量纲散度拉回绝对辐射度尺度
    return sqrt(W_eff * D_J * sample1.w * sample2.w * (2.0 / 3.0));
}

float alice_normalized_distance_fast(vec4 sample1, vec4 sample2, vec4 dual1, vec4 dual2, float N1, float N2) {
    float D_J = alice_weighted_jeffreys_fast(sample1, sample2, dual1, dual2);

    // 调和有效样本数 (Wald Effective Sample Size)
    float W_eff = (N1 * N2) / max(N1 + N2, 1e-20);

    float v1 = alice_estimator_variance(sample1, N1);
    float v2 = alice_estimator_variance(sample2, N2);
    return sqrt(W_eff * D_J * sample1.w * sample2.w * (2.0 / 3.0) / max(v1 + v2, 1e-6));
}

// ------------------------------------------------------------
// 漫反射辐照度重建 (核心)
// 基于三维最大熵分布的半球余弦投影解析逼近
// ------------------------------------------------------------
float alice_irradiance(vec4 encoded, vec3 N) {
    float omega = encoded.w;

    // 极小能量直接返回 0
    if (omega < 1e-6) return 0.0;

    float len_v = length(encoded.xyz);
    // 退化至各向同性: v 极小时直接使用各向同性解 e = 1/4
    if (len_v < 1e-6) return omega * 0.25;

    vec3 v_hat = encoded.xyz / len_v;

    // 计算 kappa
    float rho = min(len_v / omega, 0.999999);
    float sqrt_term = sqrt(max(0.0, 16.0 - 12.0 * rho * rho));
    float kappa = (6.0 * rho) / (4.0 + sqrt_term);

    // 余弦投影
    float mu_0 = dot(v_hat, N);
    float abs_mu_0 = abs(mu_0);

    // 中间变量
    float kappa_sq = kappa * kappa;
    float one_minus_kappa_sq = max(0.0, 1.0 - kappa_sq);
    float sqrt_one_minus_kappa_sq = sqrt(one_minus_kappa_sq);

    float denom_shared = 3.0 + kappa_sq;

    // 对称部分边界值
    float e_S0_num = 3.0 * sqrt_one_minus_kappa_sq;
    float e_S1_num = 3.0 + 6.0 * kappa_sq - kappa_sq * kappa_sq;

    // 高阶光滑插值: 用 κ⁴ 权重压制小 κ 时的折角
    float kappa_fourth = kappa_sq * kappa_sq;
    // t = (1-κ⁴)*μ₀² + κ⁴*|μ₀|
    float t = mix(mu_0 * mu_0, abs_mu_0, kappa_fourth);

    // 对称分量
    float e_S_num = mix(e_S0_num, e_S1_num, t);

    // 最终辐照度公式:
    // E = ω / (4*(3+κ²)) * ( e_S_num + 8*κ*μ₀ )
    float final_numerator = e_S_num + 8.0 * kappa * mu_0;
    float irradiance = omega * final_numerator / (4.0 * denom_shared);

    // 数值安全 (确保非负)
    return max(0.0, irradiance);
}

// ------------------------------------------------------------
// 辅助函数：仅从嵌入表示获取总能量和方向
// ------------------------------------------------------------
float alice_total_energy(vec4 encoded) {
    return encoded.w;
}

vec3 alice_direction(vec4 encoded) {
    return encoded.xyz;
}
#endif // ALICE_GLSL
