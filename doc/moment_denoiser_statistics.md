# Bures 矩降噪器的统计口径

实景出现明显噪声回归后，实验性置信域 resolve 已默认关闭（`MAXENT_TEMPORAL_CONFIDENCE_CLAMP=0`），恢复本文原时域链。失败试验的接口、假设和测试结果保留于 [置信域解算说明](temporal_confidence_resolve.md)。

本文档描述 encoder 与 decoder 之间的 latent 滤波。均值状态始终在线性矩空间中累积；距离使用该状态的 \(2\times2\) PSD Bures--Wasserstein 几何。alpha=1 的 \(g^{-3}\) 联合 MaxEnt 族只负责补全未存储的 \(R^2\) 加权角向矩，不参与最终光照解码。

## 1. 编码与可识别量

对非负亮度样本 \(R\) 和单位入射方向 \(\mathbf u\)，encoder 输出

\[
Z=(R\mathbf u,R).
\]

时域状态保存

\[
\mathbf v=E[R\mathbf u],\qquad
w=E[R],\qquad
e_2=E[R^2],\qquad N_{\rm eff}.
\]

一阶状态对应 Hermitian 矩阵

\[
M(\mathbf v,w)=\frac12
\begin{pmatrix}
w+v_z&v_x-i v_y\\
v_x+i v_y&w-v_z
\end{pmatrix}.
\]

可实现条件 \(\lVert\mathbf v\rVert\leq w\) 等价于 \(M\succeq0\)。滤波、重投影和历史更新仍直接作用于 \((\mathbf v,w,e_2)\) 的线性坐标；PSD 投影只在距离与方差求值的局部副本上执行，不回写历史。

## 2. Bures 光场距离

对两个状态 \((\mathbf v_i,w_i)\)，定义

\[
q_i=\sqrt{\max(w_i^2-\lVert\mathbf v_i\rVert^2,0)},
\]

\[
c=\sqrt{\frac{w_1w_2+\mathbf v_1\cdot\mathbf v_2+q_1q_2}{2}}.
\]

Bures--Wasserstein 距离平方为

\[
d_B^2=w_1+w_2-2c.
\]

shader 使用抗相消的等价形式

\[
d_B^2=
\frac{\lVert\mathbf v_1-\mathbf v_2\rVert^2+(q_1-q_2)^2}
{w_1+w_2+2c}.
\]

若两边均为零，距离取零。该量一次齐次：同时把两个光场乘以 \(s\geq0\) 时，\(d_B^2\) 乘以 \(s\)。

## 3. \(g^{-3}\) 补全的局部 Bures MC 方差

Bures 距离的局部二次展开还依赖 \(E[R^2\mathbf u]\) 和 \(E[R^2\mathbf u\mathbf u^T]\)。现有标量存储无法识别它们，因此实现显式采用 alpha=1 联合 MaxEnt 闭包。令

\[
\kappa=\frac{\lVert\mathbf v\rVert}{w},\qquad
\mathbf n=\frac{\mathbf v}{\lVert\mathbf v\rVert}.
\]

该族的 \(R\) 加权方向边缘正比于 \(g^{-3}\)。多乘一个 \(R\) 后，\(R^2\) 加权角密度正比于 \(g^{-4}\)，其所需矩为

\[
\frac{E[R^2\mathbf u]}{e_2}
=\frac{4\kappa}{3+\kappa^2}\mathbf n,
\]

\[
\frac{E[R^2\mathbf u\mathbf u^T]}{e_2}
=\frac{(1-\kappa^2)I+4\kappa^2\mathbf n\mathbf n^T}
{3+\kappa^2}.
\]

将它们代入 \(M(\mathbf v,w)\) 处的局部 Bures 度量并化简，得到

\[
\boxed{
V_B^{g^{-3}}=
\frac{2e_2(3-\kappa^2)/(3+\kappa^2)-w^2}{4w}}
\]

作为单个 MC 观测的局部 Bures 方差模型。shader 使用数值上更稳定的等价形式

\[
V_B^{g^{-3}}=
\frac{(e_2-w^2)+3(1-\kappa^2)e_2/(3+\kappa^2)}{4w},
\]

把非负的径向和角向项分开；其中径向差进一步按 \((\sqrt{e_2}-w)(\sqrt{e_2}+w)\) 求值，避免在 \(\kappa\to1\)、\(e_2\to w^2\) 时相消。它保留实测 \(e_2\)，因此径向 firefly 能量仍来自路径追踪器。边界连续为

\[
V_B(0)=\frac{2e_2-w^2}{4w},\qquad
V_B(1)=\frac{e_2-w^2}{4w}.
\]

实现先把 \(e_2\) 提升到物理下界 \(w^2\)，再计算闭式。若 \(w=e_2=0\)，方差为零；若 FP16 状态出现 \(w=0,e_2>0\)，则把标准差饱和到可存储上限，以免把未解析的稀疏能量误判为零噪声。对 \(N_{\rm eff}>1\)，使用 \(1/(1-1/N_{\rm eff})\) 修正有限样本的 plug-in 中心矩。该因子对径向中心矩严格成立；由于 \(\kappa\) 同样由有限历史估计，对角向闭包属于有限样本近似。

## 4. 空域冷启动方差

短历史阶段在 \(7\times7\) 几何核内先计算

\[
\bar{\mathbf v}=\frac{\sum_i a_i\mathbf v_i}{W},\quad
\bar w=\frac{\sum_i a_iw_i}{W},\quad
\bar e_2=\frac{\sum_i a_ie_{2,i}}{W},\quad W=\sum_i a_i.
\]

随后只在汇聚状态 \((\bar{\mathbf v},\bar w,\bar e_2)\) 上计算一次 \(V_B^{g^{-3}}\)。这个顺序把邻域样本之间的方向与亮度散布计入方差；逐像素先算 \(V_B\) 再平均会遗漏该项。

同一组权重对应的有效样本数单独重建为

\[
N_{\rm eff}=\frac{W^2}{\sum_i a_i^2/N_i}.
\]

空间估计与中心像素的时域估计按历史长度平滑切换。两者都表示单观测 Bures 方差，后续消费者才除以各自 \(N_{\rm eff}\)。单个样本无法从自身估计方差，因此 \(N_{\rm eff}=1\) 的中心时域方差为零，并由空间池化提供冷启动尺度。

## 5. A-Trous 传播与拒绝

standardDeviation² 在每个 A-Trous 边界表示局部单观测 Bures 方差。中心与 tap 的差异使用

\[
D=\sqrt{
\frac{d_B^2(M_c,M_s)}
{V_{B,c}/N_c+V_{B,s}/N_s}
}.
\]

由于分子和分母都随辐亮度一次缩放，该标准化差异对统一曝光缩放不变。

上式的空域拒绝计数取 \(\min(N_{\rm eff},16)\)，再乘逐级置信度尺度。每个 pass 用信号权重线性汇聚 \(V_B\)，并独立传播 Kish \(N_{\rm eff}\)。后者使用平面场景上标定的常相关闭包处理前序 pass 产生的样本重叠。标定保留实际 Bures 权重、FP16 格式及逐级拒绝尺度，覆盖 \(\kappa=0,0.7,0.98\) 和历史 \(N=1,4,16,64\)；配置、逐场数据与留出验证见 [标定说明](calibration/README.md)。

## 6. 时域响应

最终独立 current 与重投影 history 分别形成

\[
V_{\rm est,C}=V_{B,C}/N_C,\qquad
V_{\rm est,H}=V_{B,H}/N_H.
\]

时域响应使用 \(d_B^2/(V_{\rm est,C}+V_{\rm est,H})\) 调节当前帧权重。提交历史时，\((\mathbf v,w,e_2)\) 仍按同一个时域 alpha 线性更新，\(N_{\rm eff}\) 按 Kish 规则更新；过滤后的 Bures 方差字段只用于下一次显著性判断，不能反演成 \(e_2\)。

## 7. 适用边界

- \(e_2\) 必须与 \((\mathbf v,w)\) 使用同一组原始时域权重更新。
- \(g^{-3}\) 闭包恢复的是模型高阶矩。任意真实 PT 分布，特别是低概率、极亮且与主轴相反的样本，可以具有相同保存矩而产生不同的局部 Bures 方差。
- 局部 Bures 方差来自距离在均值处的二阶展开；它是用于标准化估计器差异的 delta-method 尺度，不等于任意大偏差下 \(E[d_B^2(Z,EZ)]\) 的精确值。
- 空域方差假设几何核内局部平稳；真实信号边缘会被解释为额外 MC 散布。这是保守偏差，仍需用独立时间样本校准。
- \(\kappa\to1\) 时闭式本身有限，无需运行时解析分支。距离函数仍须先投影 FP16 舍入造成的 \(\lVert\mathbf v\rVert>w\)。
- 当前六级传播常数为 \(0,0.09184833,0.12613998,0.13781854,0.14262104,0.14655028\)，是对平面测试域的递归重叠近似。固定权重下，Bures 二次型的公共因子 \(\operatorname{tr}(G\Sigma)\) 在归一化相关性中消去；实际信号拒绝会改变核重叠。独立探针冻结信号权重后标定该部分，独立 MC 重复采样另行验证完整自适应滤波的中心方差。常数的适用性依赖拒绝强度、采样模式和初始方差策略。
