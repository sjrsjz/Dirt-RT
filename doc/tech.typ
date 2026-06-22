#set text(font: "Microsoft YaHei")
#set page(numbering: "1")

= 技术文档

== 前言

=== 背景介绍

漫反射光照重建是计算机图形学中的一个重要问题，其目的是从一组光照样本中重建出一个光照向量，以便于后续的渲染。在实际应用中，由于光照样本的噪声，重建出的光照向量往往会受到噪声的影响，因此需要对光照向量进行降噪处理。

注: 本质上是我认为 SVGF 太差劲难以调参，SH 又重又会产生负数和振铃效应，SG 慢的爆炸，ReSTIR 又不是真正的降噪 Pass，NRD 我又不会也不适合 MC shader。这直接性的导致我逼出了下面的文章。

=== 目的

本文档旨在介绍一种新的光照编码以及其对应的降噪方案与重建算法。

== 建模

=== 第一性原理
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

=== 算子的导出

==== 代数同构与形式解构
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
$ T_"avg"(cal(L)) = ( E[bold(v)], E[I] + E[|bold(v)|] - |E[bold(v)]| ) $

==== 唯一性证明
合成算子的唯一性严格由双射同构 $phi$ 与第一性公理共同保证：
“能量守恒性”独立且完全决定了标量分量（总测度空间维度）的封闭特征，从而锁死标量积；“方向响应可加性”在测度基础上显式消除了非恒等的各向同性重放缩解（例如避免了对方向矢量强度的凸显或抑制造成的非线性偏置）。此外，结合与交换不变性在数学上提供了累加操作底层拓扑的李群对称性保障。反向拉回到 $S$ 后，$T$ 解析形式必须唯一，不可能存在违背该显式而又同时满足上述公理的其他算子。

=== 算子的线性嵌入与算法引入

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

== 统计模型

=== 蒙特卡洛表示与最大熵编码

合成算子 $T$ 的第一性原理只约束代数结构，本身不包含任何统计学语义。然而，实机降噪管线在本质上需要处理基于蒙特卡洛（Monte Carlo, MC）采样的随机信号，为了将其接入降噪管线，并在统计层面最少地引入人为偏见先验（Ad-hoc），我们从信息论角度采用最大熵原理（Maximum Entropy Principle）为单个光照状态 $L$ 构造其所隐式编码的概率分布。

在实机 MC 采样管线中，单次发射射线所携带的纯原始辐射度是不包含各向同性低频能量的界端状态 $L_("raw") = (bold(x), 0)$，其线性嵌入表示等于 $phi(L_("raw")) = (bold(x), |bold(x)|)$，严格位于同构锥的边界上。而在经过算子滤波（累汇）后得到的内部局部光照状态 $L = (bold(v), I)$，对应总能量 $omega = |bold(v)| + I$（此时通常 $omega > |bold(v)|$）。

我们将 $cal(S)[L] = (bold(v), omega)$ 视作对局部光照场的充分观测约束。基于最大熵原理，我们寻找一个定义在连续动量空间 $bold(x) in RR^n$ 上的最高熵概率密度分布 $p(bold(x))$，使其满足：

$
    "Maximize" quad & H[p] = - integral_(RR^n) p(bold(x)) log p(bold(x)) d bold(x) \
  "Subject to" quad & integral_(RR^n) p(bold(x)) d bold(x) = 1 \
                    & integral_(RR^n) bold(x) p(bold(x)) d bold(x) = bold(v) \
                    & integral_(RR^n) |bold(x)| p(bold(x)) d bold(x) = omega
$

==== 解析分布形式与边界处理

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

==== 原始样本空间 $RR^n$ 的方差结构

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

==== $T^*$ 嵌入空间 $(bold(X), |bold(X)|)$ 的联合方差

由于光照合成的算法载体实际在线性嵌入空间 $bold(Y) = (bold(X), R) = (bold(X), |bold(X)|)$ 中执行运算，其完整的联合协方差块矩阵对时空滤波器至关重要（其通过 $delta$-method 支持更复杂的协方差评估）：

$ op("Cov")(bold(Y)) = mat(op("Cov")(bold(X)), op("Cov")(bold(X), R); op("Cov")(R, bold(X)), op("Var")(R)) $

其中，方向与能量的交叉协方差向量为：
$ op("Cov")(bold(X), R) = (2(n+1) kappa omega^2) / (n + kappa^2)^2 hat(bold(v)) $

径向能量不确定度方差为：
$ op("Var")(|bold(X)|) = omega^2 / (n+kappa^2)^2 [ n + (n+3) kappa^2 - kappa^4 ] $


==== 信息几何对偶空间与散度测度

为了度量空间邻域或时域历史中两个光照状态 $L_1(bold(v)_1, omega_1, N_1)$ 与 $L_2(bold(v)_2, omega_2, N_2)$ 之间的核心相似度，我们将光照状态嵌入至连续概率流形的信息几何表面中。基于勒让德对偶性（Legendre Duality），单个光照状态 $L = (bold(v), omega)$ 在超空间 $RR^(n+1)$ 存在唯一的双向对偶表示轴系：

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

== 光照重建

=== 积分建模与半球投影

在三维渲染场景（$n=3$）中，设局部着色点表面的单位法向量为 $arrow(n) in S^2$。假设材质为理想漫反射（Lambertian）材质，根据辐照度（Irradiance）的定义，我们需要对前文推导出的最大熵分布 $p_L (bold(x))$ 在半球空间进行余弦加权投影积分。

我们将 $E(arrow(n))$ 定义为半球夹角余弦投影的数学期望：
$
  E(arrow(n)) = E_(p_L) [ max(0, bold(x) dot arrow(n)) ] = integral_(RR^3) max(0, bold(x) dot arrow(n)) p_L (bold(x)) d bold(x)
$

代入三维极大熵分布的解析形式（其中单位球面面积与伽马函数乘积 $|S^2| Gamma(3) = 8pi$）：
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(RR^3) max(0, bold(x) dot arrow(n)) exp(-beta (abs(bold(x)) - kappa hat(bold(v)) dot bold(x))) d bold(x)
$

=== 方位角解析积出与一维化简

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

=== 边界行为分析与完美对称/反对称解耦

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

=== 物理光滑性修正与高精度逼近

对称分量 $e_S$ 描述了从各向同性边缘 $e_S(0, kappa)$ 演变至共线对齐边缘 $e_S(1, kappa)$ 的过程。由于当 $kappa < 1$ 时，极大熵概率密度场在局部流形上光滑可微（$C^oo$ 连续），其产生的光照响应在 $\mu_0 = 0$ 处的一阶导数必须严格为 $0$。

只有当系统退化至极端的狄拉克极限（$kappa arrow.r 1$）时，折角项 $| \mu_0 |$ 的非连续一阶特征才会显现。基于此物理先验，线性过渡函数 $t$ 中对折角项的混合权重不应是线性的，而应随着 $kappa$ 的弱化呈现出高阶衰减特征。

我们引入高阶特征权重 $kappa^4$ 来压制中低频段的折角响应，构造以下过渡函数 $t$ 与对称部分近似：
$ t = (1 - kappa^4) mu_0^2 + kappa^4 | mu_0 | $
$ e_S (mu_0, kappa) approx e_S (0, kappa) + (e_S (1, kappa) - e_S (0, kappa)) dot t $

我们将 $e_S (0, kappa)$ 与 $e_S (1, kappa)$ 的边界解析值带入并合并，即可得到*兼顾各极限边界严格精确、物理场光滑连续且全域最大相对误差控制在 $0.4%$ 以内*的最终重建公式：

$
  E(arrow(n)) approx (omega) / (4(3+kappa^2)) [ 3 sqrt(1 - kappa^2) + (3 + 6 kappa^2 - kappa^4 - 3 sqrt(1 - kappa^2)) dot ((1 - kappa^4) mu_0^2 + kappa^4 | mu_0 |) + 8 kappa mu_0 ]
$

#image("/assets/image.png")

=== 实机 HLSL 降噪管线实现

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

== 命名
根据 AI 的建议，暂时将该光照编码方案命名为 `Asymmetric Laplace Isomorphic Conic Encoding`（简称 `ALICE`）。该名称反映了其核心数学结构：基于最大熵原理的非对称拉普拉斯分布（Asymmetric Laplace）在同构空间中的锥形编码特征。