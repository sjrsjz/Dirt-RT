#set text(font: "Microsoft YaHei", lang: "zh")
#set page(numbering: "1", header: "Asymmetric Laplace Isomorphic Conic Encoding 技术文档", footer: link(
  "https://github.com/sjrsjz",
  [GitHub: sjrsjz],
))
#set document(
  date: datetime.today(),
  author: (
    "https://github.com/sjrsjz",
    "sjrsjz@gmail.com",
  ),
  title: "Asymmetric Laplace Isomorphic Conic Encoding 技术文档",
  description: [
    Asymmetric Laplace Isomorphic Conic Encoding 技术文档
  ],
  keywords: (
    "Real-time Path Tracing",
    "Spatiotemporal Denoising",
    "Maximum Entropy Principle",
    "Directional Moment Closure",
    "Path Guiding",
    "Information Geometry",
  ),
)
#show raw: it => text(it, font: ("Consolas", "Microsoft YaHei"), size: 10pt, lang: "zh")

#v(10em)
#align(center)[#text(size: 18pt)[Asymmetric Laplace Isomorphic Conic Encoding \ 技术文档]]

#v(5em)

#figure(caption: [ALICE 编码通过类SVGF降噪器降噪后的结果，5次弹射])[
  #align(center)[
    #image("./assets/image-1.png")
  ]
]

#pagebreak()
#outline()
#pagebreak()

#align(center)[
  #rect(width: 100%, stroke: 0.5pt + luma(200), radius: 4pt, inset: 12pt)[
    #set align(left)
    #text(size: 10pt, fill: luma(80))[
      *开源声明 (License)* \
      *文档本体*（文字内容、插图与理论阐述）采用 *知识共享 署名 4.0 国际（CC BY 4.0）* 协议 \
      （#link("https://creativecommons.org/licenses/by/4.0/")[https://creativecommons.org/licenses/by/4.0/]）： \
      任何人均可在署名作者的前提下自由复制、修改、再分发与商业使用本文档本体；再分发或改编时须保留本署名及许可声明，并注明是否对原文档作出修改。\
      *配套代码*（GLSL / HLSL 着色器实现）仍采用 *MIT License* 开源，允许任何个人与企业免费用于学术研究、商业游戏引擎及离线渲染器开发；分发或使用核心算法代码时请保留原作者署名及本声明。\
      *署名 (Attribution)*：作者 sjrsjz（#link("https://github.com/sjrsjz")[GitHub: sjrsjz] / sjrsjz\@gmail.com），标题《Asymmetric Laplace Isomorphic Conic Encoding 技术文档》。
    ]
  ]
]
= 前言

== 背景介绍

漫反射光照重建是计算机图形学中的一个重要问题，其目的是从一组光照样本中重建出一个光照向量，以便于后续的渲染。在实际应用中，由于光照样本的噪声，重建出的光照向量往往会受到噪声的影响，因此需要对光照向量进行降噪处理。

注: 本质上是我认为 SVGF @schied2017svgf 太差劲难以调参还有糊的一批的冷启动模糊，SH 又重又会产生负数和振铃效应，SG 慢的爆炸，ReSTIR @bitterli2020spatiotemporal 又不是真正的降噪 Pass，NRD @nvidia2021nrd 我又不会也不适合 MC shader（因为过于重量级）。这直接性的导致我逼出了下面的文章。

== 目的

本文档旨在介绍一种新的光照编码以及其对应的降噪方案与重建算法。

= 建模

== 信号视角的出发点

在实时蒙特卡洛路径追踪管线中，单次射线弹射返回的原始信号为方向-辐射率乘积：

$ bold(x)_i = hat(bold(d))_i dot L_i in RR^n $

其中 $hat(bold(d))_i$ 为探针射线方向的单位向量，$L_i$ 为该方向入射辐射率（incident radiance）的亮度标量。对于单条射线探针，BRDF 调制延迟至着色阶段统一施加，因此 $L_i$ 是材质无关的纯入射光场测量值。该信号的天然结构为：

- *方向信息*：$hat(bold(d))_i = bold(x)_i / |bold(x)_i|$（信号朝向）
- *辐射率幅度*：$L_i = |bold(x)_i|$（信号 L1 范数）

我们将光照状态空间直接定义为增广锥：

$ cal(C) = {(bold(v), omega) in RR^n times RR_(>=0) | omega >= |bold(v)|} $

任意光照编码表示为二元组 $L = (bold(v), omega) in cal(C)$，其中 $bold(v) in RR^n$ 为方向矩，$omega in RR_(>=0)$ 为总入射辐射率（total incident radiance）。光照合成算子 $T: "List"(cal(C)) -> cal(C)$ 将一组信号样本映射为合成光照状态，其中 $cal(L) in "List"(RR^n)$ 为输入的蒙特卡洛样本列表。

#text(size: 10pt, fill: luma(100))[
  *设计依据：* 状态空间 $cal(C)$ 的选择并非人为规定，而是由信号结构与概率论共同决定的自然结果（详见「锥约束的概率论来源」一节）。
]

== 第一性原理

本建模完全从代数第一性原理与信号处理视角出发（排除额外物理假设），推导光照合成算子 $T$ 的解析形式。其满足的代数公理如下：

+ *公理 1 — 交换结合不变性（信号无序性）* \
  光照的合成结果与样本的输入顺序及计算分包无关。对于任意样本列表 $cal(L)$ 及其任意无交划分（Partition）$cal(L) = union.big_k cal(L)_k$，算子满足：
  $ T(cal(L)) = T({ T(cal(L)_k) }) $

  #text(size: 10pt, fill: luma(100))[
    *信号视角：* 蒙特卡洛样本是一条无序数据流，统计提取量不应依赖样本到达顺序。该公理在数学上提供了累加操作底层拓扑的李群对称性保障。
  ]

+ *公理 2 — 辐射率守恒（L1 幅度可加性）* \
  合成前后系统的总辐射率严格守恒。辐射率定义为信号的 L1 范数，对于任意样本列表 $cal(L) = {bold(x)_i}$，满足：
  $ omega(T(cal(L))) = sum_(bold(x)_i in cal(L)) |bold(x)_i| $

  #text(size: 10pt, fill: luma(100))[
    *信号视角：* 这是 L1 范数的可加性——信号处理中最基本的幅度守恒律。在蒙特卡洛语境下，这意味着合成光照的总辐射率等于所有采样射线的辐射率之和。
  ]

+ *公理 3 — 方向矩保真（一阶矩可加性）* \
  组合后的方向矩等于各样本方向矩的向量叠加。对于任意样本列表 $cal(L) = {bold(x)_i}$，满足：
  $ bold(v)(T(cal(L))) = sum_(bold(x)_i in cal(L)) bold(x)_i $

  等价表述：对于任意线性探针方向 $bold(a) in RR^n$，其对合成光照的线性响应量等于对所有输入样本响应量的叠加：
  $ bold(a) dot bold(v)(T(cal(L))) = sum_(bold(x)_i in cal(L)) bold(a) dot bold(x)_i $

  #text(size: 10pt, fill: luma(100))[
    *信号视角：* 这是信号一阶矩（均值）的可加性。方向矩 $bold(v)$ 即信号在 $RR^n$ 中的向量均值乘以样本数，其线性可加性是信号空间的基本结构性质。
  ]

+ *公理 4 — 正一阶齐次性* \
  当所有输入等比例缩放时，合成后的结果亦等比例缩放。对于任意非负标量 $lambda >= 0$，满足：
  $ T({lambda bold(x)_i}) = lambda T({bold(x)_i}) $

  #text(size: 10pt, fill: luma(100))[
    *信号视角：* 信号整体放大 $lambda$ 倍，所有统计提取量等比例放大。这是信号处理中的尺度协变性公理。
  ]

我们辅助定义 $ T_"avg" (cal(L)) = 1/(|cal(L)|) T(cal(L)) $ 以表示输入样本的平均光照合成结果。

== 算子的导出

=== 直接推导

由公理 2（辐射率守恒）与公理 3（方向矩保真），合成算子 $T$ 的解析形式被*直接且唯一地确定*：

$ T(cal(L)) = (sum_(bold(x)_i in cal(L)) bold(x)_i, sum_(bold(x)_i in cal(L)) |bold(x)_i|) in cal(C) $

其对应的均值算子为：

$ T_"avg"(cal(L)) = (E[bold(x)], E[ |bold(x)| ]) $

=== 锥封闭性证明

需验证 $T(cal(L)) in cal(C)$，即证明 $omega >= |bold(v)|$：

$ omega = sum_i |bold(x)_i| >= |sum_i bold(x)_i| = |bold(v)| $

上式由三角不等式直接保证。$square$

=== 线性性

引入增广信号表示 $tilde(bold(x)) = (bold(x), |bold(x)|) in RR^(n+1)$，则合成算子退化为纯向量加法：

$ T(cal(L)) = sum_(bold(x)_i in cal(L)) tilde(bold(x))_i $

均值算子即为增广信号的数学期望：

$ T_"avg"(cal(L)) = E[tilde(bold(x))] $

=== 唯一性证明

合成算子的唯一性由公理 2 与公理 3 *独立且完全地*保证：

- *公理 2* 独立且完全地锁定了标量分量：$omega(T(cal(L))) = sum |bold(x)_i|$ 是唯一满足辐射率守恒的标量赋值。
- *公理 3* 独立且完全地锁定了向量分量：$bold(v)(T(cal(L))) = sum bold(x)_i$ 是唯一满足方向矩保真的向量赋值。
- 公理 1（交换结合不变性）提供了累加操作的李群对称性保障，排除了任何非交换或非结合的合成方案。
- 公理 4（正齐次性）排除了任何对样本数量或幅度的非线性依赖。

两条核心公理（2 与 3）各自独立地唯一确定了输出状态的一个分量，不存在任何其他算子能同时满足这两条公理。因此 $T$ 的解析形式唯一。$square$

== 锥约束的概率论来源

本节阐述状态空间 $cal(C)$ 的锥约束 $omega >= |bold(v)|$ 为何不是人为规定，而是概率论的必然结果。

=== Jensen 不等式保证

对于任意定义在 $RR^n$ 上的概率测度 $mu$（具有有限一阶矩与一阶绝对矩），由 Jensen 不等式：

$ E_mu[ |bold(x)| ] >= |E_mu[bold(x)]| $

即 $omega >= |bold(v)|$。因此，任何由 $(bold(v), omega) = (E[bold(x)], E[ |bold(x)| ])$ 参数化的光照状态*自动落入锥* $cal(C)$ 中。

=== 边界态的重新解释

锥边界 $omega = |bold(v)|$ 对应 Jensen 不等式取等，当且仅当分布退化为 Dirac $delta$ 函数——即所有信号样本指向同一方向。这精确对应蒙特卡洛管线中单样本未经任何空间融合的原始射线态：

$ "单样本: " quad (bold(v), omega) = (bold(x)_i, |bold(x)_i|), quad omega = |bold(v)| quad "（锥边界）" $

而经过空间滤波累汇后的多样本状态，由于经验分布的弥散性，Jensen 不等式严格成立：

$ "多样本: " quad omega = E[ |bold(x)| ] > |E[bold(x)]| = |bold(v)| quad "（锥内部）" $

=== Jensen 差的统计含义

定义 Jensen 差：

$ I = omega - |bold(v)| = E[ |bold(x)| ] - |E[bold(x)]| >= 0 $

$I$ 度量信号分布偏离 Dirac 态（完全定向）的弥散程度。在当前实现中它是保证 $omega >= |bold(v)|$ 的线性锥坐标：$I = 0$ 表示锥边界上的纯方向状态，$I > 0$ 表示存在方向弥散。


== 算法意义

本节阐述上述代数结构对实机渲染管线的工程价值。

在计算机图形学与实时渲染中，空间滤波与降噪算法（如 SVGF 及其衍生架构）本质上高频依赖各种线性或凸组合操作（加权求和、卷积等）。

本建模的数学推导给出了强有力的证明：*合成算子 $T$ 在增广信号空间中即为纯向量加法*。在实机管线中，我们仅需在增广表示 $tilde(bold(x)) = (bold(x), |bold(x)|)$ 下对样本执行极低成本的常规线性混合（求取 $T_"avg"$），即可严格保证最终的滤波结果绝对吻合所有的代数公理与辐射率守恒律。

由于 $T$ 本身就是线性的，*无需任何后置非线性修正*——滤波循环中的每一步操作都是严格的线性组合，辐照度重建 $E(hat(arrow(n)))$ 延迟至着色输出阶段一次性完成。这为兼顾线性矩保持与着色执行效率的光照降噪管线提供了基础。

= 统计模型

== 运行时闭包

ALICE 的缓冲状态仍为线性一阶矩
$ arrow(L) = (bold(v), omega), quad |bold(v)| <= omega $，
其中 $omega$ 是总入射能量，$bold(v)$ 是方向一阶矩。混合、时域累积与空间滤波都直接在该四维锥中执行。

运行时闭包采用径向参考测度 $d nu = r d r d Omega$。令
$kappa = rho = |bold(v)| / omega$、$hat(bold(v)) = bold(v) / |bold(v)|$，则联合密度为
$
  p_L(r, arrow(u)) =
  (beta^2 (1-kappa^2)) / (4 pi)
  exp(-beta r (1-kappa hat(bold(v)) dot arrow(u))),
  quad beta = 2 / (omega(1-kappa^2)).
$
对半径积分后得到单位球面上的归一化能量密度
$
  p_E(arrow(u)) =
  (1-kappa^2)^2 /
  (4 pi (1-kappa hat(bold(v)) dot arrow(u))^3).
$
它严格满足 $integral_(S^2) p_E d Omega = 1$ 与
$integral_(S^2) arrow(u) p_E d Omega = kappa hat(bold(v))$，因此
$kappa$ 直接等于归一化一阶矩长度，不需要数值反演。$kappa = 0$ 是均匀球面分布；$kappa arrow.r 1$ 时闭包收敛到沿主轴的方向原子。

== 二阶统计与对偶参数

以主轴为平行方向，单样本的空间协方差分量为
$
  sigma_"perp"^2 &= (1-kappa^2) omega^2 / 2,   sigma_"parallel"^2 &= (1+kappa^2) omega^2 / 2.
$
代码中使用的标量方差与径向方差分别为
$
  "Var"_"scalar" &= (3-kappa^2) omega^2 / 2
    = 3 omega^2 / 2 - |bold(v)|^2 / 2,   "Var"(R) &= (1+kappa^2) omega^2 / 2.
$
若缓冲存储的是 $"Var"_"scalar" / N_"eff"$，则径向估计量方差通过
$
  ("Var"(R)) / N_"eff" =
  ("Var"_"scalar") / N_"eff" dot
  (1+kappa^2) / (3-kappa^2)
$
恢复。

用于散度计算的自然参数是
$
  bold(theta) = beta kappa hat(bold(v)) = beta bold(v) / omega,
  quad beta = 2 / (omega(1-kappa^2)).
$
记 $bold(phi) = (bold(theta), -beta)$、$bold(psi) = (bold(v), omega)$，则
$bold(phi) dot bold(psi) = -2$。两个状态的对称 Jeffreys 散度在实现中写为
$
  D_J = beta_1 omega_2 + beta_2 omega_1
    - bold(theta)_1 dot bold(v)_2
    - bold(theta)_2 dot bold(v)_1 - 4.
$
时空滤波再以
$W_"eff" = N_1 N_2 / (N_1+N_2)$
对其加权。

== Lambert 查询

令 $mu = hat(bold(v)) dot arrow(n)$，
$d = sqrt(1-kappa^2+kappa^2 mu^2)$。归一化半球余弦响应具有解析形式
$
  e(kappa, mu) =
  (1-kappa^2+2 kappa^2 mu^2) / (4 d)
  + kappa mu / 2,
  quad E = omega e.
$
为避免背向、高集中状态下的消减误差，$kappa mu < 0$ 时实现改用等价表达式
$
  e(kappa, mu) =
  (1-kappa^2)^2 /
  (4 d (d-kappa mu)^2).
$
边界行为为 $e(0,mu)=1/4$，以及
$lim_(kappa arrow.r 1) e(kappa,mu)=max(mu,0)$。

== EON 查询

粗糙漫反射路径沿用同一 $(bold(v), omega)$ 状态。实现把 EON 响应拆成解析 Lambert 项、Fujii--Oren--Nayar 方向分区和多次散射补偿。方向分区由 Iris 自定义的三维 `RGBA16F` LUT 重建，四个通道存储变换后 $kappa$ 的分段三次 Bernstein 控制值；缺失能量项使用闭式稳定分支，避免第二次纹理访问。均匀态与方向原子态直接走精确边界路径。

== GLSL 接口

核心实现在 `shaders/lib/lighting/maxent.glsl` 与
`shaders/lib/lighting/eon.glsl`。调用者继续传入
`vec4(v, omega)`，无需修改缓冲布局：

```glsl
float kappa = clamp(length(v) / omega, 0.0, 1.0 - 1e-6);
float response = maxent_irradiance(vec4(v, omega), normal);
vec3 outgoing = eon_project_maxent(
    maxEntY, CoCg, normal, wo, roughness, albedo);
```

= 降噪管线实现

== 管线总览

ALICE 降噪管线采用多 Pass 架构，在时空域上对漫反射光照信号进行递进式滤波。整体管线由以下核心 Pass 组成（参考 #link("https://github.com/sjrsjz/Dirt-RT/tree/better-denoiser-dev/shaders/post", [Dirt RT/shaders/post])）：

#figure(caption: [ALICE 降噪管线数据流])[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      [*Pass*], [*着色器*], [*功能*],
      [100], [fragment], [时域累积：利用运动矢量重投影历史帧，执行 ALICE 增广空间的时域递归滤波],
      [swap2], [compute], [方差预滤波：$5 times 5$ 几何双边滤波器平滑原始 ALICE 方差；$3 sigma$ 能量钳位],
      [300_cs], [compute], [空间滤波 L1—L3：à-trous 小波分解 $R_0 in {1, 2, 4}$，利用共享内存加速],
      [300], [fragment], [空间滤波 L4—L6：à-trous 小波分解 $R_0 in {8, 16, 32}$，旋转抖动去相关],
      [swap3], [compute], [缓冲交换：将滤波结果写回主缓冲区，完成双缓冲 Flip，保存历史统计信息],
    )
  ]
]

其中 Pass 100（时域累积）利用运动矢量将历史帧的 ALICE 编码重投影至当前帧，在增广空间中执行指数滑动平均。该 Pass 的详细内容超出本节范围，本节重点阐述空间域降噪的三个核心 Pass：swap2（方差预滤波）、300/300_cs（空间滤波）与 swap3（缓冲交换）。

#text(size: 11pt, fill: rgb("#8B0000"))[
  *⚠ 关键提示：* 尽管本降噪器在架构上参考了 SVGF @schied2017svgf 的 à-trous 小波分解框架，但其在*信号域、权重函数、方差估计与预处理策略*等关键维度上与标准 SVGF 存在本质性差异。以下各节将在相应位置显式标明这些差异。核心差异概览见下表。
]

#figure(caption: [ALICE 降噪器与标准 SVGF 的关键差异])[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      [*维度*], [*标准 SVGF*], [*本降噪器*],
      [*信号域*],
      [RGB 三通道颜色（辐射度量空间）],
      [ALICE 增广表示 $(bold(v), omega)$（四维线性空间），色度 $("Co", "Cg")$ 独立滤波],

      [*合成算子*],
      [非线性（需处理颜色空间的非线性组合）],
      [纯线性 $T$ 算子——增广空间中样本累加即为向量加法，由 ALICE 代数结构保证],

      [*方差估计*],
      [基于颜色通道的局部经验方差],
      [ALICE 最大熵分布的闭型理论方差 $"Var"_"scalar"(bold(X))$（仅依赖 $(bold(v), omega)$ 一阶矩），含 $5 times 5$ 预滤波 Pass],

      [*亮度权重*],
      [RGB 亮度梯度 + 方差归一化],
      [ALICE 向量空间距离 $|bold(v)_"center" - bold(v)_"sample"|$ + 预滤波方差归一化],

      [*能量钳位*], [无], [$3 sigma$ 能量钳位（swap2），保留 $rho = (|bold(v)|)/omega$ 不变],
      [*时域累积*], [RGB 颜色空间的指数滑动平均], [ALICE 增广空间的直接线性累加，$w$ 即有效帧数 $N_"eff"$],
      [*输出信号*],
      [滤波后 RGB 颜色],
      [滤波后 $(bold(v), omega) + ("Co", "Cg")$，辐照度重建（$E(arrow(n))$）延迟至着色阶段一次性完成],
    )
  ]
]

#text(size: 12pt, fill: rgb("#1A5276"))[
  *ALICE 降噪器的工程特性：* \
  ALICE 降噪器在*漫反射通道内*无需区分直接光照（DI）与间接光照（GI）即可获得极佳的联合降噪结果——ALICE 探针编码的是材质无关的入射光场，DI 的太阳光照和 GI 的间接弹射光在增广空间中自然累加。镜面反射与折射通道使用独立的降噪管线（reflectIlluminationBuffer / refractIlluminationBuffer），其架构与 ALICE 解耦。 \
  对于纯漫反射或不透明非金属材质，漫反射通道承担绝大部分降噪负载；对于纯金属等无漫反射分量的材质，镜面反射通道独立处理。 \
  实际表现：*接触阴影在 5—10 帧内收敛至可清晰识别的程度*（60 fps 下约 80—170 ms），在人眼感知中几乎无法察觉收敛过程。
]

== 统一漫反射缓冲区

ALICE 降噪管线在宿主端（Host-side）采用统一 SSBO（Shader Storage Buffer Object）管理漫反射数据，替代传统的多纹理方案。单个像素的数据结构 `UnifiedDiffuseElement` 包含 18 个 `float`（72 字节），涵盖以下功能域：

+ *RT 输出域 (12B)*：光线追踪当前帧的原始 ALICE 编码（`rt_aliceY_xy`, `rt_aliceY_zw`, `rt_CoCg`），由 `ray0.rgen` 写入，Pass 100 读取。
+ *当前几何域 (20B)*：世界空间位置 $(p_x, p_y, p_z)$ + 八面体压缩法线 `oct_n` + 第二法线 `oct_n2`。
+ *历史几何域 (16B)*：上一帧的世界空间位置与压缩法线，用于时域重投影的边缘停止判定。此域独立于当前几何域，因为 `ray0.rgen` 每帧覆写当前几何域而历史几何必须跨帧保留。
+ *时域历史域 (14B)*：上一帧累积的 ALICE 编码与累积权重，由 swap3 写入，Pass 100 读取。
+ *交换缓冲域 (14B)*：当前帧待滤波/已滤波的 ALICE 编码与权重，作为 Pass 100 → swap2 → 300 → swap3 之间的数据总线。

ALICE 编码采用半精度浮点（float16）压缩存储：每个光照状态 $(bold(v), omega)$ 的 `vec4` 打包为两个 `float`（通过 `packHalf2x16`/`unpackHalf2x16`），色度分量 `CoCg` 打包为一个 `float`，总计 3 个 `float` 即可完整表示一个 ALICE 光照状态。压缩/解压接口如下：

```hlsl
vec3 packAlice(AliceEncoding encoded) {
    float s0 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.x, encoded.aliceY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.z, encoded.aliceY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(encoded.CoCg.x, encoded.CoCg.y)));
    return vec3(s0, s1, s2);
}

AliceEncoding unpackAlice(float s0, float s1, float s2) {
    AliceEncoding encoded;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    encoded.aliceY = vec4(v0.x, v0.y, v1.x, v1.y);
    encoded.CoCg = v2;
    return encoded;
}
```

== 方差预滤波 (swap2)

=== 设计动机

本降噪器的空间滤波器核心依赖是逐像素的方差估计 $sigma^2$。然而，单帧蒙特卡洛采样产生的原始方差包含极高频率的噪声，直接用于指导 à-trous 滤波会导致：(1) 亮度权重 $w_"luma" = Delta E / sigma$ 在噪声像素处失稳，产生块状伪影；(2) 低方差区域被误判为高置信度，导致边缘模糊。

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 标准 SVGF 使用颜色通道的局部经验方差，且在 à-trous 的每一级通过 $sigma^2 = max(0, 1/n sum x_i^2 - mu^2)$ 实时计算。本降噪器则利用 ALICE 最大熵分布的闭型理论方差 $"Var"_"scalar"(bold(X))$，该方差仅依赖一阶矩 $(bold(v), omega)$ 和有效帧数 $N_"eff"$，无需存储二阶矩。更重要的是，我们引入了一个独立的*方差预滤波 Pass*（swap2），在进入 à-trous 迭代之前对原始方差进行 $5 times 5$ 几何感知双边平滑。这一预处理是本降噪器区别于标准 SVGF 的关键设计之一。
]

因此在进入 à-trous 迭代之前，需要对原始 ALICE 方差进行一次温和的预平滑。swap2 在 $16 times 16$ 工作组上执行 $5 times 5$ 几何感知双边滤波，同时完成 $3 sigma$ 能量钳位与向 colortex 的数据推送。

=== 共享内存协作加载

swap2 采用 $20 times 20$ 的共享内存 Tile（$16 times 16$ 工作组 + 2px Halo），256 个线程通过 Round-Robin 分摊 400 个像素的加载任务。每个 Tile 元素包含位置、法线、原始方差与总能量 $omega$。

原始方差由 `alice_estimator_variance(aliceY, weight)` 实时计算，其中 `weight` 为时域累积的有效帧数 $N_"eff"$。方差公式采用前文推导的三维标量方差闭型解（参见"原始样本空间 $RR^n$ 的方差结构"一节）：

$ "Var"_"scalar"(bold(X)) = (2omega^2 + omega sqrt(4omega^2 - 3|bold(v)|^2)) / 3 - 1/2 |bold(v)|^2 $

$ sigma_"raw"^2 = "Var"_"scalar"(bold(X)) / max(N_"eff", 1) $

=== 3-sigma 能量钳位

在对 ALICE 编码执行空间滤波之前，swap2 先对中心像素的输出能量 $omega$ 施加 $3 sigma$ 钳位。在 $5 times 5$ 邻域内计算加权均值 $mu_omega$ 与标准差 $sigma_omega$（权重为 $w_"kernel" dot w_"geom"$），然后将中心像素的 $omega$ 钳制到 $[mu_omega - 3 sigma_omega, mu_omega + 3 sigma_omega]$：

$
  "scale" = omega_"clamped" / max(omega_"center", 10^(-8)), quad bold(v)' = bold(v) dot "scale", quad ("Co", "Cg")' = ("Co", "Cg") dot "scale"
$

钳位后按比例缩放整个 ALICE 向量以保持各向异性度 $rho = (|bold(v)|) / omega$ 不变。这一操作保留了锥约束 $omega >= |bold(v)|$，同时防止了因时域累积不足导致的极端能量离群值在后续 à-trous 滤波中扩散。

=== 5×5 几何感知方差滤波

方差滤波采用 B-spline 核 $h(x) in {1.0, 0.66667, 0.44444}$ 的 $5 times 5$ 可分离权重：

$ w_"kernel"(k_x, k_y) = h(|k_x|) dot h(|k_y|) $

几何权重采用 SVGF 标准公式：

$ w_"normal" = (op("clamp")(bold(n)_"center" dot bold(n)_"sample", 0, 1))^(gamma_n) $
$ w_"depth" = exp(-(|(bold(p)_"sample" - bold(p)_"center") dot bold(n)_"center"|) / (sigma_p dot "footprint")) $

总权重为三因子乘积 $w = w_"kernel" dot w_"normal" dot w_"depth"$，滤波后方差为加权平均（可选保守模式取 $max("filtered", "center")$ 以保证方差不被低估）。如果为了减少计算开销，可以把法线权重替换为 $exp$ 形式近似。

=== 高斯曲率边缘标记 （可选，非必要）

#text(fill: rgb("#8B0000"))[
  *⚠ 可选特性：* 高斯曲率边缘标记是本降噪器的一个*实验性辅助特性*，由编译宏 `ENABLE_GAUSSIAN_FILTER` 控制开关。在多数场景下，基础的几何权重（法线 + 深度）已足以提供可靠的边缘停止。该特性并非降噪器的必要组成部分，关闭后不影响核心降噪质量。该过程仅适用于TAA无法正确处理的破碎亚像素几何，并在实际应用中避免使用。
]

在几何边缘（如方块棱角、不连续边界），法线与深度的边缘停止无法可靠分离两侧的光照信号——因为几何缓冲区本身在边缘处已经断裂。swap2 利用 $3 times 3$ 有限差分计算局部表面的高斯曲率 $K$：

$
  E = (D_x bold(p))^2, quad F = (D_x bold(p)) dot (D_y bold(p)), quad G = (D_y bold(p))^2
$
$
  L = D_(x x) bold(p) dot bold(n), quad M = D_(x y) bold(p) dot bold(n), quad N = D_(y y) bold(p) dot bold(n)
$
$ K = (L N - M^2) / (E G - F^2) $

其中一阶/二阶偏导数由中心差分估计（$D_x$ 取 $plus.minus 1$ 邻域差之半，$D_(x x)$ 取 $+1, 0, -1$ 三点的二阶中心差分）。当 $|K| > tau$（曲率阈值，由 `CURVATURE_THRESHOLD` 宏定义）时，将该像素的 $omega$ 标记为负值。后续的 300/300_cs Pass 检测到 $omega < 0$ 时自动将几何权重置零（`geomValid = 0`），仅依赖亮度权重进行降噪，避免在几何不连续处错误地混合两侧信号。

== 空间滤波器 (300 / 300_cs)

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 本降噪器仅在 à-trous 小波分解的*迭代架构*上参考了 SVGF。权重函数的核心——信号表示、距离度量、方差来源——均基于 ALICE 编码重新设计，与标准 SVGF 有根本性区别。下文详述。
]

=== à-trous 小波分解架构

空间滤波器采用 6 级 à-trous 小波迭代（迭代架构参考 @schied2017svgf），每级使用 $3 times 3$ 滤波核，步长 $R_0$ 逐级翻倍：

#figure(caption: [à-trous 迭代级别配置])[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      [*级别 (STEP)*], [*步长 $R_0$*], [*着色器*], [*有效半径*],
      [1], [$1$], [300_cs (Compute)], [1 px],
      [2], [$2$], [300_cs (Compute)], [3 px],
      [3], [$4$], [300_cs (Compute)], [7 px],
      [4], [$8$], [300 (Fragment)], [15 px],
      [5], [$16$], [300 (Fragment)], [31 px],
      [6], [$32$], [300 (Fragment)], [63 px],
    )
  ]
]

前 3 级（$R_0 <= 4$）在计算着色器（300_cs）中执行，利用共享内存（LDS）减少冗余纹理读取；后 3 级（$R_0 >= 8$）回退到片段着色器（300），因为大步长下邻域 Tile 重叠率低（线程间纹理读取不再显著重叠），LDS 收益递减。

=== 计算着色器共享内存优化

300_cs 的核心优化在于利用工作组内的数据复用。对于 $16 times 16$ 工作组和 $R_0$ 步长，所需纹理 Tile 尺寸为 $(16 + 2R_0)^2$：

#figure(caption: [各 $R_0$ 级别的 Tile 尺寸与共享内存用量])[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      [*$R_0$*], [*Tile 尺寸*], [*几何 (vec4)*], [*光照 (vec4)*],
      [1], [$18 times 18$], [324], [324],
      [2], [$20 times 20$], [400], [400],
      [4], [$24 times 24$], [576], [576],
    )
  ]
]

所有 Tile 尺寸对应的共享内存用量（最大 $576 times 2 times 16 = 18.0$ KB）均远低于典型 GPU 的 32—64 KB LDS 限制。256 个线程通过 Round-Robin 分摊加载任务（每个线程加载 $ceil("TILE_AREA" / 256)$ 个元素），越界像素的方差写入负值作为天空标记。加载完成后通过 `barrier()` + `memoryBarrierShared()` 同步，后续的 $3 times 3$ à-trous 采样循环完全从共享内存读取，将全局纹理读取次数降低约 3—6 倍。

=== 三边权重函数

每个邻域样本的综合权重由三项因子乘积构成。

*贴图空间权重 (Kernel Weight)*：B-spline 核 $h in {1.0, 0.66667}$，仅区分中心与十字邻域：

$ w_"kernel"(i, j) = h(|i|) dot h(|j|) $

*几何权重 (Geometry Weight)*：

$
  w_"geom" = gamma_n (1 - bold(n)_"center" dot bold(n)_"sample") + (|(bold(p)_"sample" - bold(p)_"center") dot bold(n)_"center"|) / (sigma_p dot "footprint") dot "geomValid"
$

其中：
- $gamma_n$ 为法线灵敏度参数（`SVGF_NORMAL_POWER`），控制法线差异的惩罚强度；
- $sigma_p$ 为位置灵敏度参数（`SVGF_POSITION_PARAM`）；
- $"footprint" = max("distToCam" / "resolution.y", 10^(-4))$ 为像素的世界空间足印尺寸，使深度项与屏幕分辨率无关；
- `geomValid` 由高斯曲率标记控制：曲率超阈值时此项归零，滤波器降级为纯亮度驱动。

*亮度权重 (Luminance Weight)*：

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 标准 SVGF 的亮度权重基于 RGB 颜色空间的梯度 $|L_i - L_j|$（$L_i$ 为像素亮度）。本降噪器则在 ALICE 增广空间中度量差异——使用方向向量 $bold(v)$ 的欧氏距离 $|bold(v)_"center" - bold(v)_"sample"|$。这一选择的数学依据是：$bold(v)$ 在增广空间 $cal(C)$ 中即为线性可加的信号分量，其欧氏距离直接度量了光照在方向-能量联合空间中的差异，无需经过辐照度重建步骤。此外，由于方差预滤波（swap2）已对 $sigma^2$ 进行了平滑，此处直接使用中心像素的预滤波方差作为归一化基准（标准 SVGF 在每级 à-trous 中使用局部计算的 $sigma_"center"^2 + sigma_"sample"^2$）：
]

$ w_"luma" = (|bold(v)_"center" - bold(v)_"sample"|) / sqrt(sigma_"center"^2) dot phi_l $

其中 $phi_l$ 为亮度灵敏度参数（`SVGF_PHI_L`）。

=== 关于非对称 $sigma^2$ 选择的关键说明

此处使用非对称形式 $sigma_"center"^2$ 而非对称形式 $sigma_"center"^2 + sigma_"sample"^2$，是本降噪器经过大量实验验证后的核心设计决策，其背后的逻辑链高度依赖方差预滤波（swap2）的存在：

*情形一：无方差预滤波（标准 SVGF 方案）*

若直接在 à-trous 中使用未经预平滑的原始方差，则必须采用对称形式 $max(sigma_"center"^2 + sigma_"sample"^2, 10^(-8))$。原因在于：原始方差信号的噪化程度极高，单像素方差估计极不稳定。若仅取 $sigma_"center"^2$，中心像素的方差可能在噪声峰谷之间剧烈震荡——当 $sigma_"center"^2$ 偶然极小（方差低谷）而 $|bold(v)_"center" - bold(v)_"sample"|$ 因噪声较大时，亮度权重 $w_"luma"$ 会异常膨胀，导致邻域样本被过度信任，出现*黑斑噪点与严重的能量丢失*（降噪器崩溃）。$sigma_"sample"^2$ 的引入本质上是一种统计平滑——两个独立噪声项的叠加降低了权重对单点方差异常的敏感度。

*情形二：有方差预滤波（本降噪器方案）*

swap2 的 $5 times 5$ 双边平滑已大幅消除了方差信号中的高频噪声，$sigma_"center"^2$ 本身已足够稳定。此时若继续使用对称形式 $sigma_"center"^2 + sigma_"sample"^2$，不仅 $sigma_"sample"^2$ 的稳定化作用已微乎其微，更严重的是——$sigma_"sample"^2$ 与 $sigma_"center"^2$ 虽经预滤波，但在空间上仍残留微小差异。这些差异在帧间随采样模式变化而波动，经双样本叠加后在 $w_"luma"$ 中产生*极其严重的时域频闪副作用*。这种频闪无法被时域累积完全消除，因为亮度权重的帧间波动直接改变了每一级 à-trous 的有效核形状。

*结论*：方差预滤波与单样本 $sigma_"center"^2$ 是一对高度耦合的设计——预滤波消除了 $sigma_"center"^2$ 的不稳定性，从而允许安全地移除 $sigma_"sample"^2$ 项，进而根除双样本方差叠加引发的时域频闪。这是本降噪器区别于标准 SVGF 的又一关键且非显见的差异点。

=== 方差下界的参数敏感性

在实机代码中，$sigma^2$ 在进入 $1/sqrt(sigma^2)$ 之前被钳制到下界 $epsilon$：

$ sigma^2 = max(sigma_"center"^2, epsilon), quad epsilon = 10^(-9) $

（对称方案中则为 $max(sigma_"center"^2 + sigma_"sample"^2, 10^(-8))$，原理相同。）

$epsilon$ 是本降噪器中对*接触阴影质量*与*暗区稳定性*之间权衡的最敏感参数。其作用机制如下：

$epsilon$ 控制的是 $1/sqrt(sigma^2)$ 的上界——因为 $1/sqrt(sigma^2) <= 1/sqrt(epsilon)$。当 $sigma_"center"^2$ 在暗区自然趋于极小时，$1/sqrt(sigma^2)$ 若不设上限将趋向无穷大，导致 $w_"luma"$ 完全支配组合权重，降噪器对任何微小的 $bold(v)$ 差异都过度响应。

- *$epsilon$ 偏大（如 $10^(-6)$）*：降噪器"迟钝"——$1/sqrt(sigma^2)$ 被压低，$w_"luma"$ 在暗区无力区分真正的光照边界与噪声，*低能量区域接触阴影被模糊*。
- *$epsilon$ 偏小（如 $10^(-12)$）*：降噪器"过度灵敏"——$1/sqrt(sigma^2)$ 在暗区极高，$w_"luma"$ 对噪声起伏过度响应，表现为*降噪崩溃*（暗区出现结构性黑斑）或*阴影蠕动*（帧间噪声被权重放大后形成缓慢漂移的伪影）。
- *$epsilon = 10^(-9)$*：经大量场景（室内暗角、密林阴影、洞穴）实验后选定的平衡点，在保留接触阴影锐度的前提下为暗区提供足够的数值稳定性。

*组合权重*：亮度权重以"双边增强"形式出现在乘积的两个位置——作为指数衰减项惩罚大能量差，同时作为前置因子提供温和的自适应增强：

$ w_0 = w_"kernel" dot (1 + w_"luma") dot exp(-(w_"geom" + w_"luma")) $

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 标准 SVGF 的组合权重为 $w_"kernel" dot exp(-(w_"geom" + w_"luma"))$，即仅依赖指数衰减。本降噪器引入的 $(1 + w_"luma")$ 前置因子额外提供了一阶亮度自适应增强，在高方差区域（$w_"luma"$ 大）提供更快的收敛速度与更强的去噪能力。这一修改是 ALICE 增广空间下特有的设计——因为 $w_"luma"$ 基于 $bold(v)$ 的欧氏距离而非 RGB 梯度，其数值范围与统计特性不同于标准 SVGF。
]

=== ALICE 增广空间的样本累积

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的本质差异：* 这是本降噪器与标准 SVGF 最根本的分歧点。标准 SVGF 在 RGB 颜色空间中累积加权颜色值 $sum_i w_i bold(c)_i$——颜色空间中的线性组合并不对应物理上的光照合成。本降噪器则在 ALICE 增广空间 $cal(C)$ 中累积——由 ALICE 第一性原理保证，$T$ 算子在增广空间中即为纯向量加法，因此 $sum_i w_i tilde(bold(x))_i$ 在数学上严格等价于光照的物理合成。降噪器无需在每步处理非线性合成逻辑，这是 ALICE 编码相对于传统 RGB 降噪的核心优势。
]

邻域样本通过 ALICE 增广空间的纯线性累加进行合成（对应前文 $T$ 算子的线性性质）：

$
  bold(v)_"accum" = sum_i w_i bold(v)_i, quad omega_"accum" = sum_i w_i omega_i, quad ("Co", "Cg")_"accum" = sum_i w_i ("Co", "Cg")_i
$

归一化后 $bold(v)_"out" = bold(v)_"accum" / sum_i w_i$，同理 $omega$ 与色度均除以总权重。方差通过独立样本加权均值的方差传播公式传递至下一级：

$ sigma_"out"^2 = (sum_i w_i^2 sigma_i^2) / ((sum_i w_i)^2) $

=== 旋转抖动

在 $R_0 >= 8$（STEP ≥ 4）的级别中，固定的轴对齐采样模式会产生结构化的栅格伪影（Grid Artifacts）。为消除此伪影，atrous_denoise_diffuse.glsl 在每级引入随机旋转（Rotation Jitter）：

$ bold(d)_"rotated" = R(theta) dot bold(d)_"aligned", quad theta = 2 pi dot "rand"("pix" + R_0) $

旋转矩阵 $R(theta)$ 为标准的 2D 旋转，随机种子由像素坐标与当级步长 $R_0$ 联合哈希生成，确保：(1) 不同像素的旋转角独立分布；(2) 不同级别的旋转角互不相关；(3) 同一像素同级别的旋转角帧间保持固定（避免时域闪烁）。采样坐标取整（`round`）以防止浮点截断误差导致越界。

=== 最终 Pass 特殊处理

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 标准 SVGF 的最终级仅输出滤波结果。本降噪器额外输出一份大半径模糊版本至独立纹理（`colortex5`），供 swap3 用于低置信度像素的收敛加速。这一双输出设计在标准 SVGF 中不存在。
]

最后一级（STEP = 6）在输出常规滤波结果（`colortex4`）的同时，额外输出一份"大半径模糊后"的结果至 `colortex5`（`out_light_sample_blurred`）。该结果在 swap3 中被读取，用于低权重像素的快速收敛。此外，高斯曲率标记（$omega$ 的正负号，非必要特性）仅在非最终 Pass 中向后传递，最终 Pass 输出时强制取绝对值，确保后续的辐照度重建不受到符号标记的干扰。

== 缓冲交换 (swap3)

swap3 是连接当前帧与下一帧的桥梁，负责完成以下关键任务：

*双缓冲 Flip*：将 `data_swap`（当前帧的滤波结果）复制到 `data`（将作为下一帧的历史帧），供 Pass 100 在下一帧进行时域重投影读取。这一操作实现了时间维度的数据流闭环。

*历史统计保存*：$w_"prev" = w$，将当前帧的有效累积帧数保存为历史权重。该权重在下一帧的 Pass 100 中用于：(1) 计算时域混合因子 $alpha$；(2) 评估重投影样本的可信度。

*滤波结果读回*：从 colortex5 读取最终 Pass 输出的模糊后 ALICE 编码，写入 `data_swap` 作为下一帧空间滤波的起点。

*低权重混合加速（受 NRD 启发 @nvidia2021nrd，非 SVGF 标准流程）*：

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的差异：* 标准 SVGF 的时域—空间关系是单向的（时域累积 → 空间滤波），时域历史不会在空间滤波后被修改。本降噪器在 swap3 中执行反向混合——利用大半径空间滤波结果回补低置信度的时域历史。这一设计借鉴了 NRD（NVIDIA Real-time Denoisers）@nvidia2021nrd 的思想，在标准 SVGF 中不存在。
]

当历史累积不足时（$w < 1.0$），将当前历史数据向空间滤波结果进行温和混合：

$ "data" = "mix"("data", "blurred_alice", "clamp"(1 / max(w, 1.0), 0, 1)) $

此操作在冷启动、去遮挡（Disocclusion）或快速相机运动导致的时域累积断裂时提供额外的收敛加速。实践中该混合对降噪质量的影响轻微但有助于消除孤立的闪烁像素。参数 $1/max(w, 1.0)$ 确保混合强度与信度成反比：$w$ 越小则越倾向于信任空间滤波结果。同样地，该操作并非必须的，在大多数场景中关闭后不会显著影响最终质量。

== 数据流总结

综合上述三个 Pass，ALICE 降噪管线的完整数据流如下：

1. *Pass 100 (时域累积)*：从 SSBO 读取历史 ALICE 编码 $arrow(L)_"hist"$，与当前帧 RT 输出 $arrow(L)_"curr"$ 在增广空间中执行加权平均，输出 $arrow(L)_"temp"$ 及累积权重 $w$。
2. *swap2 (方差预滤波)*：读取 $arrow(L)_"temp"$ 及 $w$，计算原始方差 $sigma_"raw"^2$，执行 $5 times 5$ 双边平滑 + $3 sigma$ 钳位 + 高斯曲率标记，推送至 colortex3（几何）与 colortex4（ALICE + 方差）。
3. *300_cs (空间滤波 L1—L3)*：从 colortex3/4 协作加载至 LDS，执行 $R_0 in {1, 2, 4}$ 的 à-trous 滤波，结果写回 colortex4。
4. *300 (空间滤波 L4—L6)*：从 colortex3/4 直接 texelFetch，执行 $R_0 in {8, 16, 32}$ 的 à-trous 滤波（含旋转抖动），最终级额外输出至 colortex5。
5. *swap3 (缓冲交换)*：双缓冲 Flip，保存历史统计，读回模糊结果，完成数据流闭环。下一帧从步骤 1 重新开始。

在压缩表示下，ALICE 降噪器的空间滤波阶段仅需要两个 $"vec4"$（共 32 字节）即可完整表示所有必要的几何信息（一个 $"vec4"$）与光照信息（一个 $"vec4"$）

= 命名与适用范围

该光照编码继续称为 Asymmetric Laplace Isomorphic Conic Encoding（ALICE）。其中 “Conic” 指线性状态空间
$cal(C) = {(bold(v), omega) | omega >= |bold(v)|}$；
“Isomorphic” 指原始表示 $(bold(v), I)$ 与嵌入表示
$(bold(v), omega=|bold(v)|+I)$ 之间的可逆映射。

当前运行时闭包是针对方向能量重建所选择的统计模型。由于其参考测度不是三维笛卡尔 Lebesgue 测度，本文不再把它解释为 Maxwell--Jüttner 光子气模型。$kappa$ 仅表示归一化一阶矩长度和方向集中度；$beta$ 是闭包的尺度参数，不作为物理漂移速度或热力学温度使用。


= 屏幕空间光路重建重要性采样

#figure(caption: [降噪得到的光场一阶矩分布])[
  #align(center)[
    #image("./assets/image-2.png")
  ]
]

== 物理动机与前置分布

在实时路径追踪（Real-time Path Tracing）中，尽管通过下一次事件估计（Next Event Estimation, NEE）可以有效降低直接光照的方差，但对于复杂的次级反弹（如长廊深处、极小窗口的室内），盲目的余弦重要性采样（Cosine-weighted Sampling）极难命中有效光源，导致间接光照产生极具破坏性的高频长尾噪声（Fireflies）。

既然我们在降噪管线中已经利用 ALICE 编码在时空域上提取并重构了光场的最大熵分布状态 $(bold(v), omega)$，我们自然可以将其作为*先验知识（Prior）*，在下一帧发射光线时对半球空间进行路径引导（Path Guiding）。

基于闭包的角向能量密度，引导 PDF 直接取为
$
  p_"ALICE"(arrow(u)) =
  (1-kappa^2)^2 /
  (4 pi (1-kappa hat(bold(v)) dot arrow(u))^3).
$
该表达式已经在完整单位球面 $S^2$ 上归一化，并且其一阶矩正好是
$kappa hat(bold(v))$。

== 解析逆变换采样

令 $mu = hat(bold(v)) dot arrow(u)$，则边缘分布满足
$
  (1-kappa mu)^(-2) =
  op("lerp")((1+kappa)^(-2), (1-kappa)^(-2), xi_1).
$
因此在 $kappa > 0$ 时可直接计算
$
  mu =
  (1 -
    [op("lerp")((1+kappa)^(-2), (1-kappa)^(-2), xi_1)]^(-1/2))
  / kappa.
$
$kappa$ 接近零时使用 $mu=2 xi_1-1$；方位角仍为
$phi=2 pi xi_2$。该采样器与上式 PDF 完全配对，无需拒绝采样。


== 动态多重重要性采样

虽然 ALICE 提供了极其逼近真实光场的引导，但在遮挡剧烈变化的动态场景中，前一帧的引导先验可能失效（例如光源突然移动或相机瞬移）。为了保证渲染方程的绝对无偏性（Unbiasedness）并避免除零方差爆炸，我们将 ALICE 采样与经典的余弦重要性采样（Cosine-weighted Sampling）进行多重重要性采样（MIS）@veach1995optimally 混合。

在闭包中，归一化一阶矩长度 $rho = (|bold(v)|) / omega$ 反映光场的“定向确信度”。因此，我们将 ALICE 的混合概率权重 $P_"guide"$ 直接与 $rho$ 挂钩：
$
  P_"guide" = cases(
    0.975 dot rho & "if" |bold(v)| > 10^(-8),
    0 & "if" |bold(v)| <= 10^(-8)
  )
$
*设计意图：* 当光场趋于环境漫反射（$rho arrow.r 0$）时，系统自动退化为余弦采样；当光场表现出强方向性（$rho arrow.r 1$）时，系统将 $97.5%$ 的算力投入 ALICE 引导，保留 $2.5%$ 的余弦采样作为安全底线。

最终，下一级射线的混合概率密度函数为：
$ p_"mix"(arrow(u)) = (1 - P_"guide") dot p_"cos"(arrow(u)) + P_"guide" dot p_"ALICE"(arrow(u)) $
其中 $p_"cos"(arrow(u)) = max(0, arrow(n) dot arrow(u)) / pi$。

在渲染方程的蒙特卡洛积分器中，漫反射 BSDF 贡献度（乘积因子）为：
$ W_"BSDF" = (f_r dot max(0, arrow(n) dot arrow(u))) / p_"mix"(arrow(u)) = (p_"cos"(arrow(u))) / p_"mix"(arrow(u)) $

上述理论最终收敛于极简的代码架构中：实机管线在无需任何复杂的八叉树或神经网格缓存的前提下，仅利用上一帧时空滤波残留的副产物 $(bold(v), omega)$，便以极低的 ALU 开销完成了一次理论完备、物理无偏的路径引导。

#bibliography("./references.bib", title: "参考文献", style: "ieee")
