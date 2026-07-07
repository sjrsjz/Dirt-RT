#set text(font: "Microsoft YaHei", lang: "zh")
#set page(numbering: "1", header: "Asymmetric Laplace Isomorphic Conic Encoding 技术文档", footer: link(
  "https://github.com/sjrsjz",
  [GitHub: sjrsjz],
))
#set document(
  date: datetime(year: 2026, month: 7, day: 4),
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
    "Maxwell-Jüttner Distribution",
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

== 第一性原理
我们将光照状态空间定义为 $S = RR^n times RR_(>=0)$。
任意光照编码表示为二元组 $L = (arrow(v), I) in S$，其中 $arrow(v) in RR^n$，$I in RR_(>=0)$。

定义能量度量函数 $omega(L) : S -> RR_(>=0)$ 满足：
$ omega(L) := abs(arrow(v)) + I $

定义分量函数 $ v(L) := bold(v), quad I(L) := I $。

本建模完全从代数第一性原理出发（排除额外物理假设），推导光照合成算子 $T: "List"(S) -> S$ 的解析形式（其中 $cal(L) in "List"(S)$ 为输入的光照样本列表）。

其满足的代数第一性原理公理如下：

+ *结合与交换不变性 (Associativity & Commutativity)* \
  光照的合成结果与样本的输入顺序及计算分包无关。对于任意样本列表 $cal(L)$ 及其任意无交划分（Partition）$cal(L) = union.big_k cal(L)_k$，算子满足：
  $ T(cal(L)) = T({ T(cal(L)_k) }) $

+ *能量守恒性 (Energy Conservation)* \
  合成前后系统的总能量严格守恒。对于任意样本列表 $cal(L)$，满足：
  $ omega(T(cal(L))) = sum_(L_i in cal(L)) omega(L_i) $

+ *正一阶齐次性 (Positive Homogeneity)* \
  当所有输入等比例缩放时，合成后的结果亦等比例缩放。对于任意非负标量 $lambda >= 0$，满足：
  $ T({lambda L_i}) = lambda T({L_i}) $

+ *方向响应可加性 / 一阶矩守恒 (Directional Response Additivity)* \
  对于任意方向探测器（即任意线性投射方向 $bold(a) in RR^n$），其对合成光照的线性响应量必须等于对所有输入局部样本响应量的叠加。即：
  $ bold(a) dot v(T(cal(L))) = sum_(L_i in cal(L)) bold(a) dot v_i $

我们辅助定义 $ T_"avg" (cal(L)) = 1/(|cal(L)|) T(cal(L)) $ 以表示输入样本的平均光照合成结果。

== 算子的导出

=== 代数同构与形式解构
我们将光照状态空间 $S = RR^n times RR_(>=0)$ 通过双射 $phi: S -> C$ 映射到更易处理的凸锥 $C subset RR^(n+1)$ 中：
$C = { (bold(x), y) in RR^(n+1) | y >= |bold(x)| }$。

映射与逆映射分别定义为：
$
  phi(bold(v), I) := (bold(v), |bold(v)| + I) \
  phi^(-1)(bold(x), y) := (bold(x), y - |bold(x)|)
$

在同构空间 $C$ 中，诱导的合成算子 $T_C$ 需要满足原始公理的对应形式：
- 由 *能量守恒性* 直接锁定第二个坐标（即纵轴能量分量）为常规累加：$Omega_"res" = sum (|bold(v)_i| + I_i)$；
- 由新引入的 *方向响应可加性* ，由于公理对所有探测方向 $bold(a)$ 均成立，故等式成立的充要条件严格限定了其底流形上的向量分量合成必须为普通向量求和：$bold(V)_"res" = sum bold(v)_i$。

该求和结果自然满足锥的封闭性（三角不等式保证了 $|sum bold(v)_i| <= sum |bold(v)_i| <= Omega_"res"$）。

将 $T_C$ 的结果经 $phi^(-1)$ 还原回 $S$，即得光照合成算子的唯一解析形式：
#set math.equation(numbering: none)
$
  T(cal(L)) = ( sum_(L_i in cal(L)) bold(v)_i, sum_(L_i in cal(L)) I_i + sum_(L_i in cal(L)) |bold(v)_i| - |sum_(L_i in cal(L)) bold(v)_i| )
$

其对应的均值算子为：
$ T_"avg"(cal(L)) = ( E[bold(v)], E[I] + E[ |bold(v)| ] - |E[bold(v)]| ) $

=== 唯一性证明
合成算子的唯一性严格由双射同构 $phi$ 与第一性公理共同保证：
“能量守恒性”独立且完全决定了标量分量（总测度空间维度）的封闭特征，从而锁死标量积；“方向响应可加性”在测度基础上显式消除了非恒等的各向同性重放缩解（例如避免了对方向矢量强度的凸显或抑制造成的非线性偏置）。此外，结合与交换不变性在数学上提供了累加操作底层拓扑的李群对称性保障。反向拉回到 $S$ 后，$T$ 解析形式必须唯一，不可能存在违背该显式而又同时满足上述公理的其他算子。

== 算子的线性嵌入与算法引入

为剥离原始算子 $T$ 中由绝对泛数带来的非线性耦合，我们沿用前文的同构思想，在算法层面上引入算子 $T$ 的同构诱导线性表示（Isomorphism-induced Linear Operator）$T^*: "List"(S) -> S$。

我们将光照样本映射为其线性嵌入表示（Linear Embedded Representation）$cal(S)[L]$，将其重新参数化为“向量-总能量”的联合形式：
$ cal(S)[L] := (bold(v), omega(L)) = (bold(v), I + |bold(v)|) $

在此嵌入空间下，诱导算子 $T^*$ 退化为优雅的纯线性累加：
$ T^*(cal(L)) = sum_(L_i in cal(L)) cal(S)[L_i] $

相对应的诱导平均算子即为对联合样本的线性数学期望：
$ T^*_"avg"(cal(L)) = E[cal(S)[L]] $

基于上述代数结构的嵌入与解构，原始必定包含复杂非线性项的平均算子 $T_"avg"(cal(L))$，可被严格且唯一地解耦为“线性表示期望”与“后置能量回拉（Pull-back）补偿”的差值形式：
$
  T_"avg"(cal(L)) = ( E[bold(v)], E[omega(L)] - |E[bold(v)]| ) = T^*_"avg"(cal(L)) - (bold(0), |bold(v)(T^*_"avg"(cal(L)))|)
$

*算法意义（Algorithmic Significance）：*
这一代数解耦特性展现了极具穿透力的工程价值。在计算机图形学与实时渲染中，空间滤波与降噪算法（如 SVGF 及其衍生架构）本质上高频依赖各种线性或凸组合操作（加权求和、卷积等）。
我们的数学推导给出了强有力的证明：算法在微观的滤波循环中，完全无需处理复杂的非线性光照合成。在实机管线中，我们仅需在嵌入空间 $cal(S)$ 中对样本执行极低成本的常规线性混合（求取 $T^*_"avg"$），最后在着色输出阶段仅进行一次 $O(1)$ 复杂度的整体泛数修正。此架构即可严格保证最终的滤波结果绝对吻合所有的代数分布公理与物理能量守恒不变律。这为设计兼顾“数学严格无偏性”与“极高着色执行效率”的光照降噪管线奠定了坚实的理论基石。

= 统计模型

== 蒙特卡洛表示与最大熵编码

合成算子 $T$ 的第一性原理只约束代数结构，本身不包含任何统计学语义。然而，实机降噪管线在本质上需要处理基于蒙特卡洛（Monte Carlo, MC）采样的随机信号，为了将其接入降噪管线，并在统计层面最少地引入人为偏见先验（Ad-hoc），我们从信息论角度采用最大熵原理（Maximum Entropy Principle）@jaynes1957information 为单个光照状态 $L$ 构造其所隐式编码的概率分布。

在实机 MC 采样管线中，单次发射射线所携带的纯原始辐射度是不包含各向同性低频能量的界端状态 $L_("raw") = (bold(x), 0)$，其线性嵌入表示等于 $phi(L_("raw")) = (bold(x), |bold(x)|)$，严格位于同构锥的边界上。而在经过算子滤波（累汇）后得到的内部局部光照状态 $L = (bold(v), I)$，对应总能量 $omega = |bold(v)| + I$（此时通常 $omega > |bold(v)|$）。

我们将 $cal(S)[L] = (bold(v), omega)$ 视作对局部光照场的充分观测约束。基于最大熵原理，我们寻找一个定义在连续动量空间 $bold(x) in RR^n$ 上的最高熵概率密度分布 $p(bold(x))$，使其满足：

$
    "Maximize" quad & H[p] = - integral_(RR^n) p(bold(x)) log p(bold(x)) d bold(x) \
  "Subject to" quad & integral_(RR^n) p(bold(x)) d bold(x) = 1 \
                    & integral_(RR^n) bold(x) p(bold(x)) d bold(x) = bold(v) \
                    & integral_(RR^n) |bold(x)| p(bold(x)) d bold(x) = omega
$

=== 解析分布形式与边界处理

利用变分法（拉格朗日乘子法），在非奇异域 $omega > |bold(v)| > 0$ 内可严格导出该最大熵问题的唯一解析解，属于指数族分布（Exponential Family）：

$
  p_L (bold(x)) = (beta^n (1 - kappa^2)^((n+1)/2)) / (|S^(n-1)| Gamma(n)) exp(-beta(|bold(x)| - kappa hat(bold(v)) dot bold(x)))
$

其中 $|S^(n-1)|$ 为 $n-1$ 维单位球面面积（对于三维空间 $n=3$，恒有 $|S^2| Gamma(3) = 8pi$）。参数解析形式完全由一阶观测矩阵确定：

$ rho = (|bold(v)|) / omega $
$ kappa = (2n rho) / ( (n+1) + sqrt((n+1)^2 - 4 n rho^2) ) $
$ beta = (n + kappa^2) / (omega (1 - kappa^2)) $
$
  hat(bold(v)) = cases(
    bold(v) / (|bold(v)|) & "if" bold(v) != bold(0),
    bold(h)_("any") & "if" bold(v) = bold(0)
  )
$

*边界极限行为的数值保障*：自然物理下会出现以上连续分布的退化极限：
1. *零向量无偏衰减（$bold(v) = bold(0)$）*：此时 $rho = 0, kappa = 0$，分布退化为各向同性的拉普拉斯衰减场 $p(bold(x)) prop exp(-beta |bold(x)|)$。
2. *绝对黑暗态（$omega = 0$）*：系统处于严格能量断绝态，方差彻底收缩为点质量分布（即狄拉克 $delta(bold(x))$ 函数），在 Shader 实现时以 `omega` 防除零机制直接越过评估。
3. *未被降噪的原始射线态（$omega = |bold(v)|$）*：此时连通流形边界 $rho=1, kappa=1$ 导致 $beta$ 趋向无穷，使得分布成为沿 $hat(bold(v))$ 轴线的极度窄分布（狄拉克异化）。这精确反映了单样本未经任何空间融合时极度缺乏低频信息的事实。在实机应用中可硬性阈值截断使其始终落在非奇异测度域内：$rho < 1 - epsilon$。

=== 原始样本空间 $RR^n$ 的方差结构

非奇异极大熵分布在原始空间 $RR^n$ 中的协方差矩阵 $op("Cov")(bold(X))$ 具有完美的轴对称几何特征（不确定度旋转体）：

$
  op("Cov")(bold(X)) = sigma_(perp, bold(X))^2 (bold(I)_n - hat(bold(v))hat(bold(v))^T) + sigma_(parallel, bold(X))^2 hat(bold(v))hat(bold(v))^T
$

其中垂直主方向与平行主方向的本征方差分别为：
$ sigma_(perp, bold(X))^2 = ((n+1) omega^2 (1 - kappa^2)) / (n + kappa^2)^2 $
$ sigma_(parallel, bold(X))^2 = ((n+1) omega^2 (1 + kappa^2)) / (n + kappa^2)^2 $

其对应的系统本征均质总标量方差（Scalar Variance）为：
$ "Var"_("scalar")(bold(X)) = op("tr")(op("Cov")(bold(X))) = ((n+1) omega^2) / (n+kappa^2)^2 [ n + (2-n)kappa^2 ] $

在消除中间参数 $kappa$ 后，其仅由输入一阶统计量决定的纯闭合解析式为：
$
  "Var"_("scalar")(bold(X)) = (omega ( (n+1)omega + sqrt((n+1)^2 omega^2 - 4 n |bold(v)|^2) )) / (2n) - (n-1)/(n+1) |bold(v)|^2
$

对于三维渲染场景（$n=3$），标量方差可极简化求值为：
$ "Var"_("scalar")(bold(X)) = (2omega^2 + omega sqrt(4omega^2 - 3|bold(v)|^2)) / 3 - 1/2 |bold(v)|^2 $

在 Shader 中，考虑平摊效应后，若样本的时域有效累积期望帧数为 $N_("eff")$，则当前像素 Estimator 的残留方差为 $"Var"_("estimator") = "Var"_("scalar")(bold(X)) / N_("eff")$。此项可直接作为双边滤波器（Bilateral Filter）等价执行的自适应动态带宽 $sigma_c^2$。

=== $T^*$ 嵌入空间 $(bold(X), |bold(X)|)$ 的联合方差

由于光照合成的算法载体实际在线性嵌入空间 $bold(Y) = (bold(X), R) = (bold(X), |bold(X)|)$ 中执行运算，其完整的联合协方差块矩阵对时空滤波器至关重要（其通过 $delta$-method 支持更复杂的协方差评估）：

$ op("Cov")(bold(Y)) = mat(op("Cov")(bold(X)), op("Cov")(bold(X), R); op("Cov")(R, bold(X)), op("Var")(R)) $

其中，方向与能量的交叉协方差向量为：
$ op("Cov")(bold(X), R) = (2(n+1) kappa omega^2) / (n + kappa^2)^2 hat(bold(v)) $

径向能量不确定度方差为：
$ op("Var")(|bold(X)|) = omega^2 / (n+kappa^2)^2 [ n + (n+3) kappa^2 - kappa^4 ] $


=== 信息几何对偶空间与散度测度

为了度量空间邻域或时域历史中两个光照状态 $L_1(bold(v)_1, omega_1, N_1)$ 与 $L_2(bold(v)_2, omega_2, N_2)$ 之间的核心相似度，我们将光照状态嵌入至连续概率流形的信息几何表面中@amari2016information。基于勒让德对偶性（Legendre Duality），单个光照状态 $L = (bold(v), omega)$ 在超空间 $RR^(n+1)$ 存在唯一的双向对偶表示轴系：

- *原始坐标向量（期望参数空间 / 广度分布）：*
  $ bold(psi)(L) := vec(bold(v), omega) in RR^(n+1) $

- *对偶坐标向量（自然参数空间 / 强度分布）：*
  $
    bold(phi)(L) := vec(bold(theta), -beta) = (n+kappa^2) / (omega(1-kappa^2)) vec((n+kappa^2) / ((n+1)omega) bold(v), -1) in RR^(n+1)
  $

上述对偶坐标严格等于系统负香农熵关于原始坐标的梯度：$bold(phi)(L) = - nabla_(bold(psi)) H(p)$。在标准内积作用下，此模型存在优雅的*零和守恒定理*：
$ bold(phi)(L) dot bold(psi)(L) = bold(theta) dot bold(v) - beta omega equiv -n $

根据对称 Bregman 散度的勒让德坐标内积定理，指数族分布的双样本杰弗里斯散度（Jeffreys Divergence）无配分函数抵消残片，等于其纯净坐标差的对向内积：
$ D_J(L_1, L_2) = (bold(phi)(L_1) - bold(phi)(L_2)) dot (bold(psi)(L_1) - bold(psi)(L_2)) $

理论散度并未囊括降噪算法中的样本信度。当用于降噪管线中的双边拒绝权重时，我们额外引入基于有效信噪比（Fisher 信息量对角线加权）的调和累积估计算子 $W_("eff") = (N_1 N_2) / (N_1 + N_2)$，由此导出了一套适合实机且无分支求值的加权相似度测距量（Weighted Jeffreys Divergence）：

$
  D_("WJ")(L_1, L_2) &= W_("eff") dot D_J(L_1, L_2) \
  &= - (N_1 N_2) / (N_1 + N_2) [ bold(phi)(L_1) dot bold(psi)(L_2) + bold(phi)(L_2) dot bold(psi)(L_1) + 2n ]
$

展开上述基于调和权重交叉项的对跖标量分量即为：
$
  D_("WJ")(L_1, L_2) = (N_1 N_2) / (N_1 + N_2) [ beta_1 omega_2 + beta_2 omega_1 - bold(theta)_1 dot bold(v)_2 - bold(theta)_2 dot bold(v)_1 - 2n ]
$

对于三维渲染空间（$n=3$），常数项 $-2n$ 恒为 $-6$。在实际管线执行中，该量测器兼具严把关的代数动态范围特质：即在低时空 SPP 积累期，测度值天然具备较高度自适应软包容使得信号快速融合重建，而在高 SPP 或结构反差明显时，测距量极其严苛迅速转为硬核分治策略防残影伪像。

= 光照重建

== 积分建模与半球投影

在三维渲染场景（$n=3$）中，设局部着色点表面的单位法向量为 $arrow(n) in S^2$。假设材质为理想漫反射（Lambertian）材质，根据辐照度（Irradiance）的定义，我们需要对前文推导出的最大熵分布 $p_L (bold(x))$ 在半球空间进行余弦加权投影积分。

我们将 $E(arrow(n))$ 定义为半球夹角余弦投影的数学期望：
$
  E(arrow(n)) = E_(p_L) [ max(0, bold(x) dot arrow(n)) ] = integral_(RR^3) max(0, bold(x) dot arrow(n)) p_L (bold(x)) d bold(x)
$

代入三维极大熵分布的解析形式（其中单位球面面积与伽马函数乘积 $|S^2| Gamma(3) = 8pi$）：
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(RR^3) max(0, bold(x) dot arrow(n)) exp(-beta (abs(bold(x)) - kappa hat(bold(v)) dot bold(x))) d bold(x)
$

== 方位角解析积出与一维化简

为简化该三维空间积分，我们引入球坐标系。令 $bold(x) = r arrow(u)$，其中 $r = abs(bold(x)) in [0, oo)$，而 $arrow(u) in S^2$ 为单位方向向量（体积元满足 $d bold(x) = r^2 d r d arrow(u)$）。

利用正齐次性将积分剥离为径向与角向的双重形式：
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(S^2) max(0, arrow(u) dot arrow(n)) [ integral_0^oo r^3 exp(-beta (1 - kappa hat(bold(v)) dot arrow(u)) r) d r ] d arrow(u)
$

利用定积分关系 $integral_0^oo r^3 e^(-a r) d r = Gamma(4) / a^4 = 6 / a^4$（由于 $kappa in [0, 1)$，其径向收敛因子 $a = beta(1 - kappa hat(bold(v)) dot arrow(u)) > 0$ 恒成立），积分式中的径向分布被严格积出：
$
  E(arrow(n)) = (3 (1 - kappa^2)^2) / (4 pi beta) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

消去中间参数 $beta = (3 + kappa^2) / (omega (1 - kappa^2))$，可得仅与宏观总能量 $omega$ 及方向分布特征相关的投影积分：
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 pi (3 + kappa^2)) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

我们建立局部坐标系，令法线 $arrow(n)$ 为 $z$ 轴，则 $cos theta = arrow(u) dot arrow(n)$。半球截断算子 $max(0, cos theta)$ 将积分域严格限制在朝上的半球 $Omega_+ = { arrow(u) in S^2 | cos theta >= 0 }$。

令 $mu_0 = hat(bold(v)) dot arrow(n)$。在此基底下，我们将主要光轴方向 $hat(bold(v))$ 投影表示为 $(sin theta_0, 0, mu_0)^T$。对方位角 $phi in [0, 2pi]$ 进行积分（应用一阶导数递推）：
$ integral_0^(2pi) d phi / (A - B cos phi)^4 = pi (2A^3 + 3A B^2) / (A^2 - B^2)^(7/2) $

其中辅元定义为：
$ A(z) = 1 - kappa mu_0 z, quad B(z)^2 = kappa^2 (1 - mu_0^2)(1 - z^2) $
$ A(z)^2 - B(z)^2 = kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) $

令 $z = cos theta in [0, 1]$，带入整理后，即可消去方位角，得到关于天顶角余弦 $z$ 的*最简化一元解析积分形式*：
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 (3 + kappa^2)) integral_0^1 (z (1 - kappa mu_0 z) [ 2 (1 - kappa mu_0 z)^2 + 3 kappa^2 (1 - mu_0^2)(1 - z^2) ]) / ([ kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) ]^(7/2)) d z
$

== 边界行为分析与对称/反对称解耦

由于上式分母含有分数阶代数项 $Q(z)^(7/2)$，其原函数形式在一般域内极其繁琐。为了构造高效的实时重建方案，我们定义归一化辐照度响应函数为 $e(mu_0, kappa) := E(arrow(n)) / omega$，并将其拆解为关于余弦角 $mu_0$ 的对称分量 $e_S$ 与反对称分量 $e_A$：
$
  e_S (mu_0, kappa) = (e(mu_0, kappa) + e(-mu_0, kappa)) / 2, quad e_A (mu_0, kappa) = (e(mu_0, kappa) - e(-mu_0, kappa)) / 2
$

通过对上述一维积分进行边界极限推导，该系统表现出以下极为高雅且对称的数学边界闭合解：

1. *全向各向同性极限 ($kappa arrow.r 0$)*：
  $ e(mu_0, 0) equiv 1/4 $
2. *极致定向极限 ($kappa arrow.r 1$)*：
  $ e(mu_0, 1) = max(0, mu_0) $
3. *光轴与法线完全同向共线 ($mu_0 = 1$)*：
  $ e(1, kappa) = ((1+kappa)^3 (3 - kappa)) / (4 (3 + kappa^2)) $
4. *光轴与法线完全反向共线 ($mu_0 = -1$)*：
  $ e(-1, kappa) = ((1-kappa)^3 (3 + kappa)) / (4 (3 + kappa^2)) $
5. *光轴与表面切面完全平行共面 ($mu_0 = 0$)*：
  $ e(0, kappa) = (3 sqrt(1 - kappa^2)) / (4 (3 + kappa^2)) $

*反对称部分的解析唯一性定理：*
进一步分析表明，反对称分量 $e_A$ 的物理本质是将半球投影还原为全球投影。通过在单位球 $S^2$ 上进行无截断积分，可严格证明该分量对任意 $mu_0$ 与 $kappa$ 均呈*严格线性关系*，且完全不存在近似误差：
$ e_A (mu_0, kappa) equiv mu_0 dot (2 kappa) / (3 + kappa^2) $

由于 $e_A$ 已被严格积出，整个光照重建的拟合误差将完全退化并收拢至对称部分 $e_S$。

== 物理光滑性修正与高精度逼近

对称分量 $e_S$ 描述了从各向同性边缘 $e_S(0, kappa)$ 演变至共线对齐边缘 $e_S(1, kappa)$ 的过程。由于当 $kappa < 1$ 时，极大熵概率密度场在局部流形上光滑可微（$C^oo$ 连续），其产生的光照响应在 $\mu_0 = 0$ 处的一阶导数必须严格为 $0$。

只有当系统退化至极端的狄拉克极限（$kappa arrow.r 1$）时，折角项 $| \mu_0 |$ 的非连续一阶特征才会显现。基于此物理先验，线性过渡函数 $t$ 中对折角项的混合权重不应是线性的，而应随着 $kappa$ 的弱化呈现出高阶衰减特征。

我们引入高阶特征权重 $kappa^4$ 来压制中低频段的折角响应，构造以下过渡函数 $t$ 与对称部分近似：
$ t = (1 - kappa^4) mu_0^2 + kappa^4 | mu_0 | $
$ e_S (mu_0, kappa) approx e_S (0, kappa) + (e_S (1, kappa) - e_S (0, kappa)) dot t $

我们将 $e_S (0, kappa)$ 与 $e_S (1, kappa)$ 的边界解析值带入并合并，即可得到*兼顾各极限边界严格精确、物理场光滑连续且全域最大相对误差控制在 $0.4%$ 以内*的最终重建公式：

$
  E(arrow(n)) approx (omega) / (4(3+kappa^2)) [ 3 sqrt(1 - kappa^2) + (3 + 6 kappa^2 - kappa^4 - 3 sqrt(1 - kappa^2)) dot ((1 - kappa^4) mu_0^2 + kappa^4 | mu_0 |) + 8 kappa mu_0 ]
$

#image("./assets/image.png")

== 辐照度重建 HLSL 实现

上述代数重组公式仅包含基础算术指令，避免了昂贵的超越函数（如 $sin, cos$）或数值积分开销，非常契合现代 GPU 渲染架构。以下为实机 Shader 执行的核心逻辑：

```hlsl
// 基于最大熵分布的高精度 O(1) 漫反射光照重建 (Irradiance Reconstruction)
// 参数说明:
//   v     - 空间滤波后得到的光照方向向量 (v = L.v)
//   omega - 空间滤波后得到的联合总能量 (omega = L.I + |L.v|)
//   N     - 当前像素的表面单位法向量
float ReconstructDiffuseLighting(float3 v, float omega, float3 N)
{
    // 0. 极小能量边界保护
    if (omega < 1e-6f) return 0.0f;

    float len_v = length(v);
    if (len_v < 1e-6f)
    {
        // 对应各向同性极限情况 (e_isotropic = 0.25)
        return omega * 0.25f;
    }

    float3 v_hat = v / len_v;
    float rho = min(len_v / omega, 0.999f); // 截断防除零

    // 1. 快速拟合特征参数 kappa (针对三维测度空间 n = 3)
    float sqrt_term = sqrt(16.0f - 12.0f * rho * rho);
    float kappa = (6.0f * rho) / (4.0f + sqrt_term);

    // 2. 余弦投影关系
    float mu_0 = dot(v_hat, N);
    float abs_mu_0 = abs(mu_0);

    // 3. 提取特征项与公共分母
    float kappa_sq = kappa * kappa;
    float one_minus_kappa_sq = max(0.0f, 1.0f - kappa_sq);
    float sqrt_one_minus_kappa_sq = sqrt(one_minus_kappa_sq);

    float denom_shared = 3.0f + kappa_sq;

    // 4. 计算对称部分的边界分量
    float e_S0_num = 3.0f * sqrt_one_minus_kappa_sq;
    float e_S1_num = 3.0f + 6.0f * kappa_sq - kappa_sq * kappa_sq;

    // 5. 应用高阶光滑插值函数过渡 (物理 C1/C2 连续保障)
    float kappa_fourth = kappa_sq * kappa_sq;
    float t = (1.0f - kappa_fourth) * mu_0 * mu_0 + kappa_fourth * abs_mu_0;

    // 6. 合并对称部分与无偏反对称部分，计算最终辐照度
    float e_S_num = lerp(e_S0_num, e_S1_num, t);
    float final_numerator = e_S_num + 8.0f * kappa * mu_0;
    float irradiance = omega * (final_numerator / (4.0f * denom_shared));

    return max(0.0f, irradiance);
}
```

= 降噪管线实现

== 管线总览

ALICE 降噪管线采用多 Pass 架构，在时空域上对漫反射光照信号进行递进式滤波。整体管线由以下核心 Pass 组成（参考 #link("https://github.com/sjrsjz/Dirt-RT/tree/better-denoiser-dev/shaders/post", [Dirt RT/shaders/post])）：

#figure(caption: [ALICE 降噪管线数据流])[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      [*Pass*], [*着色器*], [*功能*],
      [100], [fragment], [时域累积：利用运动矢量重投影历史帧，执行 ALICE 嵌入空间的时域递归滤波],
      [swap2], [compute], [方差预滤波：$5 times 5$ 几何双边滤波器平滑原始 ALICE 方差；$3 sigma$ 能量钳位],
      [300_cs], [compute], [空间滤波 L1—L3：à-trous 小波分解 $R_0 in {1, 2, 4}$，利用共享内存加速],
      [300], [fragment], [空间滤波 L4—L6：à-trous 小波分解 $R_0 in {8, 16, 32}$，旋转抖动去相关],
      [swap3], [compute], [缓冲交换：将滤波结果写回主缓冲区，完成双缓冲 Flip，保存历史统计信息],
    )
  ]
]

其中 Pass 100（时域累积）利用运动矢量将历史帧的 ALICE 编码重投影至当前帧，在嵌入空间中执行指数滑动平均。该 Pass 的详细内容超出本节范围，本节重点阐述空间域降噪的三个核心 Pass：swap2（方差预滤波）、300/300_cs（空间滤波）与 swap3（缓冲交换）。

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
      [ALICE 嵌入表示 $cal(S)[L] = (bold(v), omega)$（四维线性空间），色度 $("Co", "Cg")$ 独立滤波],

      [*合成算子*],
      [非线性（需处理颜色空间的非线性组合）],
      [纯线性 $T^*$ 算子——嵌入空间中样本累加即为向量加法，由 ALICE 代数结构保证],

      [*方差估计*],
      [基于颜色通道的局部经验方差],
      [ALICE 最大熵分布的闭型理论方差 $"Var"_"scalar"(bold(X))$（仅依赖 $(bold(v), omega)$ 一阶矩），含 $5 times 5$ 预滤波 Pass],

      [*亮度权重*],
      [RGB 亮度梯度 + 方差归一化],
      [ALICE 向量空间距离 $|bold(v)_"center" - bold(v)_"sample"|$ + 预滤波方差归一化],

      [*能量钳位*], [无], [$3 sigma$ 能量钳位（swap2），保留 $rho = (|bold(v)|)/omega$ 不变],
      [*时域累积*], [RGB 颜色空间的指数滑动平均], [ALICE 嵌入空间的直接线性累加，$w$ 即有效帧数 $N_"eff"$],
      [*输出信号*],
      [滤波后 RGB 颜色],
      [滤波后 $(bold(v), omega) + ("Co", "Cg")$，辐照度重建（$E(arrow(n))$）延迟至着色阶段一次性完成],
    )
  ]
]

#text(size: 12pt, fill: rgb("#1A5276"))[
  *ALICE 降噪器的工程特性：* \
  ALICE 降噪器的一个显著工程优势在于——*无需区分直接光照（DI）与间接光照（GI）即可获得极佳的联合降噪结果*。这直接简化了管线架构：不需要独立的 DI Pass 与 GI Pass，不需要分别维护两套时域历史与方差估计，单条 ALICE 编码管线同时对 DI 的高频接触阴影与 GI 的低频漫反射环境光进行降噪。 \
  实际表现：*接触阴影在 5—10 帧内收敛至可清晰识别的程度*（60 fps 下约 80—170 ms），在人眼感知中几乎无法察觉收敛过程。
]

== 统一漫反射缓冲区

ALICE 降噪管线在宿主端（Host-side）采用统一 SSBO（Shader Storage Buffer Object）管理漫反射数据，替代传统的多纹理方案。单个像素的数据结构 `UnifiedDiffuseElement` 包含 18 个 `float`（72 字节），涵盖以下功能域：

+ *RT 输出域 (12B)*：光线追踪当前帧的原始 ALICE 编码（`rt_aliceY_xy`, `rt_aliceY_zw`, `rt_CoCg`），由 `ray0.rgen` 写入，Pass 100 读取。
+ *当前几何域 (20B)*：世界空间位置 $(p_x, p_y, p_z)$ + 八面体压缩法线 `oct_n` + 第二法线 `oct_n2`。
+ *历史几何域 (16B)*：上一帧的世界空间位置与压缩法线，用于时域重投影的边缘停止判定。此域独立于当前几何域，因为 `ray0.rgen` 每帧覆写当前几何域而历史几何必须跨帧保留。
+ *时域历史域 (14B)*：上一帧累积的 ALICE 编码与累积权重，由 swap3 写入，Pass 100 读取。
+ *交换缓冲域 (14B)*：当前帧待滤波/已滤波的 ALICE 编码与权重，作为 Pass 100 → swap2 → 300 → swap3 之间的数据总线。

ALICE 编码采用半精度浮点（float16）压缩存储：每个光照状态 $cal(S)[L] = (bold(v), omega)$ 的 `vec4` 打包为两个 `float`（通过 `packHalf2x16`/`unpackHalf2x16`），色度分量 `CoCg` 打包为一个 `float`，总计 3 个 `float` 即可完整表示一个 ALICE 光照状态。压缩/解压接口如下：

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
  *与标准 SVGF 的差异：* 标准 SVGF 的亮度权重基于 RGB 颜色空间的梯度 $|L_i - L_j|$（$L_i$ 为像素亮度）。本降噪器则在 ALICE 嵌入空间中度量差异——使用方向向量 $bold(v)$ 的欧氏距离 $|bold(v)_"center" - bold(v)_"sample"|$。这一选择的数学依据是：$bold(v)$ 在嵌入空间 $cal(S)$ 中即为线性可加的信号分量，其欧氏距离直接度量了光照在方向-能量联合空间中的差异，无需经过辐照度重建步骤。此外，由于方差预滤波（swap2）已对 $sigma^2$ 进行了平滑，此处直接使用中心像素的预滤波方差作为归一化基准（标准 SVGF 在每级 à-trous 中使用局部计算的 $sigma_"center"^2 + sigma_"sample"^2$）：
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
  *与标准 SVGF 的差异：* 标准 SVGF 的组合权重为 $w_"kernel" dot exp(-(w_"geom" + w_"luma"))$，即仅依赖指数衰减。本降噪器引入的 $(1 + w_"luma")$ 前置因子额外提供了一阶亮度自适应增强，在高方差区域（$w_"luma"$ 大）提供更快的收敛速度与更强的去噪能力。这一修改是 ALICE 嵌入空间下特有的设计——因为 $w_"luma"$ 基于 $bold(v)$ 的欧氏距离而非 RGB 梯度，其数值范围与统计特性不同于标准 SVGF。
]

=== ALICE 嵌入空间的样本累积

#text(fill: rgb("#8B0000"))[
  *与标准 SVGF 的本质差异：* 这是本降噪器与标准 SVGF 最根本的分歧点。标准 SVGF 在 RGB 颜色空间中累积加权颜色值 $sum_i w_i bold(c)_i$——颜色空间中的线性组合并不对应物理上的光照合成。本降噪器则在 ALICE 嵌入空间 $cal(S)$ 中累积——由 ALICE 第一性原理保证，$T^*$ 算子在嵌入空间中即为纯向量加法，因此 $sum_i w_i cal(S)[L_i]$ 在数学上严格等价于光照的物理合成。降噪器无需在每步处理非线性合成逻辑，这是 ALICE 编码相对于传统 RGB 降噪的核心优势。
]

邻域样本通过 ALICE 嵌入空间的纯线性累加进行合成（对应前文 $T^*$ 算子的线性性质）：

$
  bold(v)_"accum" = sum_i w_i bold(v)_i, quad omega_"accum" = sum_i w_i omega_i, quad ("Co", "Cg")_"accum" = sum_i w_i ("Co", "Cg")_i
$

归一化后 $bold(v)_"out" = bold(v)_"accum" / sum_i w_i$，同理 $omega$ 与色度均除以总权重。方差通过独立样本加权均值的方差传播公式传递至下一级：

$ sigma_"out"^2 = (sum_i w_i^2 sigma_i^2) / ((sum_i w_i)^2) $

=== 旋转抖动

在 $R_0 >= 8$（STEP ≥ 4）的级别中，固定的轴对齐采样模式会产生结构化的栅格伪影（Grid Artifacts）。为消除此伪影，300.glsl 在每级引入随机旋转（Rotation Jitter）：

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

1. *Pass 100 (时域累积)*：从 SSBO 读取历史 ALICE 编码 $arrow(L)_"hist"$，与当前帧 RT 输出 $arrow(L)_"curr"$ 在嵌入空间中执行加权平均，输出 $arrow(L)_"temp"$ 及累积权重 $w$。
2. *swap2 (方差预滤波)*：读取 $arrow(L)_"temp"$ 及 $w$，计算原始方差 $sigma_"raw"^2$，执行 $5 times 5$ 双边平滑 + $3 sigma$ 钳位 + 高斯曲率标记，推送至 colortex3（几何）与 colortex4（ALICE + 方差）。
3. *300_cs (空间滤波 L1—L3)*：从 colortex3/4 协作加载至 LDS，执行 $R_0 in {1, 2, 4}$ 的 à-trous 滤波，结果写回 colortex4。
4. *300 (空间滤波 L4—L6)*：从 colortex3/4 直接 texelFetch，执行 $R_0 in {8, 16, 32}$ 的 à-trous 滤波（含旋转抖动），最终级额外输出至 colortex5。
5. *swap3 (缓冲交换)*：双缓冲 Flip，保存历史统计，读回模糊结果，完成数据流闭环。下一帧从步骤 1 重新开始。

在压缩表示下，ALICE 降噪器的空间滤波阶段仅需要两个 $"vec4"$（共 32 字节）即可完整表示所有必要的几何信息（一个 $"vec4"$）与光照信息（一个 $"vec4"$）

= 命名
根据 AI 的建议，暂时将该光照编码方案命名为 `Asymmetric Laplace Isomorphic Conic Encoding`（简称 `ALICE`）。该名称反映了其核心数学结构：基于最大熵原理的非对称拉普拉斯分布（Asymmetric Laplace）在同构空间中的锥形编码特征。

= 物理对应
虽然 ALICE 完全由第一性原理推导而来，但其严格等效于物理学中的*相对论统计力学*（Relativistic Statistical Mechanics）模型。具体而言，ALICE 的极大熵分布在物理上精确对应于处于局部热力学平衡态的*无静止质量漂移气体*（Drifting Massless Gas，即漂移光子气）。

== 麦克斯韦-朱特纳分布 (Maxwell-Jüttner Distribution)
当我们不对光子施加单色（固定波长）约束，而是允许其在三维连续动量空间 $RR^3$ 中自由分布时，对其宏观能量 $omega$ 与宏观动量 $bold(v)$ 施加最大熵约束，所导出的分布正是狭义相对论中的*带有漂移速度的麦克斯韦-朱特纳分布*（Maxwell-Jüttner Distribution）@juttner1911maxwellsche。

在 ALICE 的数学公式中，存在着一套极其严密且优美的物理量映射字典：

+ *相空间与动量* \
  数学状态向量 $bold(x)$ 严格对应于单个光子的动量 $bold(p)$（或等效的能量 $E/c$）。

+ *集体漂移速度* \
  各向异性度 $rho = (|bold(v)|) / omega$ 与极大熵特征参数 $kappa$ 在物理上等价。它们代表了这群光子气体作为整体在空间中运动的*无量纲集体漂移速度（Drift Velocity Ratio, $v_"drift" / c$）*。

+ *热力学温度* \
  自然参数 $beta$ 对应于实验室（摄像机）参考系下该光子气的*有效运动温度的倒数*，即 $beta = c / (k_B T_"lab")$。

+ *相对论多普勒效应* \
  角向积分中出现的核心代数项 $1 - kappa hat(bold(v)) dot arrow(u)$，正是狭义相对论中的*多普勒收缩因子（Relativistic Doppler Factor）*。由于光子气的高速集体漂移，使得光场能量在前方产生了相对论性的极度汇聚（Relativistic Beaming）。

== 光场的形态演化与热力学解释
通过这套物理映射，实机渲染中各种复杂的宏观光照现象，都可以被赋予直观且严密的微观热力学解释：

- *完全漫反射环境光 ($rho = 0, kappa = 0$)* \
  此时漂移速度为零，系统处于全局热平衡的各向同性状态。光子仅有无规则的“热运动”，对应 ALICE 编码中的纯标量底光强度 $I$。这等价于无向的均匀天光或极其充分的多次反弹低频 GI。

- *完全定向光 ($rho arrow.r 1, kappa arrow.r 1$)* \
  光子气的集体漂移速度趋近于光速。此时系统的相对温度向绝对零度收缩，所有的无规则热运动（各向同性分量 $I$）被全部冻结，转化为一致的定向动能。这在宏观上表现为一束绝对平行的强直射光（如高频太阳光束或激光）。

- *软阴影与半影过渡 ($0 < kappa < 1$)* \
  在实际场景的软阴影边缘，光场处于定向流动与热散射的中间非平衡态。ALICE 能够通过参数 $kappa$ 极其平滑地桥接这两种极端状态，实现物理自洽的接触硬化（Contact Hardening）与软阴影平滑渐变。


= 屏幕空间光路重建重要性采样

#figure(caption: [降噪得到的光场一阶矩分布])[
  #align(center)[
    #image("./assets/image-2.png")
  ]
]

== 物理动机与前置分布

在实时路径追踪（Real-time Path Tracing）中，尽管通过下一次事件估计（Next Event Estimation, NEE）可以有效降低直接光照的方差，但对于复杂的次级反弹（如长廊深处、极小窗口的室内），盲目的余弦重要性采样（Cosine-weighted Sampling）极难命中有效光源，导致间接光照产生极具破坏性的高频长尾噪声（Fireflies）。

既然我们在降噪管线中已经利用 ALICE 编码在时空域上提取并重构了光场的最大熵分布状态 $cal(S)[L] = (bold(v), omega)$，我们自然可以将其作为*先验知识（Prior）*，在下一帧发射光线时对半球空间进行路径引导（Path Guiding）。

基于极大熵的角向能量密度，我们构造定义在完整单位球面 $S^2$ 上的引导概率密度函数（PDF）：
$ p_"ALICE" (arrow(u)) = C / ((1 - kappa hat(bold(v)) dot arrow(u))^4) $
其中 $hat(bold(v))$ 为上一帧重建的引导主轴，$kappa$ 为对应的特征参数。

== 全球面积分与规范化常数

为了使其成为一个严格的概率密度函数，我们需要在全立体角上求解规范化常数 $C$。令 $mu = hat(bold(v)) dot arrow(u) = cos theta$，方位角为 $phi$，积分如下：
$
                      integral_(S^2) p_"ALICE" (arrow(u)) d arrow(u) & = 1 \
  C integral_0^(2pi) d phi integral_(-1)^1 1 / (1 - kappa mu)^4 d mu & = 1
$

对方位角积分得到 $2pi$，对 $mu$ 求定积分：
$ 2pi C [ 1 / (3 kappa (1 - kappa mu)^3) ]_(-1)^1 = 1 $
$ (2pi C) / (3 kappa) ( 1 / (1 - kappa)^3 - 1 / (1 + kappa)^3 ) = 1 $

通分化简括号内的项：
$
  ((1+kappa)^3 - (1-kappa)^3) / ((1-kappa^2)^3) = (2kappa^3 + 6kappa) / ((1-kappa^2)^3) = (2kappa(kappa^2 + 3)) / ((1-kappa^2)^3)
$

代回原式解得代数规范化常数：
$ C = (3(1-kappa^2)^3) / (4pi(3+kappa^2)) $


== 严格解析逆变换采样

为了在 GPU 中实现零舍弃率（Zero-Rejection）的高效重要性采样，我们对边缘概率密度分布 $p(mu) = 2pi C / (1 - kappa mu)^4$ 求解累积分布函数（CDF）：
$ F(mu) = integral_(-1)^mu p(x) d x = (2pi C) / (3 kappa) ( 1 / (1 - kappa mu)^3 - 1 / (1 + kappa)^3 ) $

由归一化条件可知 $F(1) = 1$。为了通过均匀分布的随机数 $xi_1 in [0, 1)$ 生成采样角 $mu$，我们令 $F(mu) / F(1) = xi_1$：
$ ( 1 / (1 - kappa mu)^3 - 1 / (1 + kappa)^3 ) / ( 1 / (1 - kappa)^3 - 1 / (1 + kappa)^3 ) = xi_1 $

为了在 Shader 中高效求解，我们定义边界常数 $a$ 与 $b$：
$ a = 1 / (1 + kappa)^3, quad b = 1 / (1 - kappa)^3 $
代入化简可得：
$ ( (1-kappa mu)^(-3) - a ) / (b - a) = xi_1 $
$ (1 - kappa mu)^(-3) = a + xi_1 (b - a) = op("lerp")(a, b, xi_1) $

对方程两边取 $-1/3$ 次幂，即可得到仅需两行代码即可在 GPU 上完成的解析逆映射方程：
$ mu = (1 - [ op("lerp")(a, b, xi_1) ]^(-1/3)) / kappa $

结合由 $xi_2$ 均匀生成的方位角 $phi = 2pi xi_2$，我们能够在 $O(1)$ 时间内直接采样出完全符合 ALICE 概率分布的射线方向。

== 动态多重重要性采样

虽然 ALICE 提供了极其逼近真实光场的引导，但在遮挡剧烈变化的动态场景中，前一帧的引导先验可能失效（例如光源突然移动或相机瞬移）。为了保证渲染方程的绝对无偏性（Unbiasedness）并避免除零方差爆炸，我们将 ALICE 采样与经典的余弦重要性采样（Cosine-weighted Sampling）进行多重重要性采样（MIS）@veach1995optimally 混合。

在代数物理上，ALICE 的无量纲漂移速度 $rho = (|bold(v)|) / omega$ 反映了光场的“定向确信度”。因此，我们将 ALICE 的混合概率权重 $P_"guide"$ 直接与 $rho$ 挂钩：
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