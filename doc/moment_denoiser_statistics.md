# 矩空间降噪器的统计口径

本文档只描述 encoder 与 decoder 之间的 latent 滤波。MaxEnt 是 decoder 的闭包选择，不参与本节的距离、方差或相似度定义。

## 1. 编码与可识别量

对一个非负亮度样本 (R) 和单位入射方向 \(\mathbf u\)，encoder 输出

\[
Z=(R\mathbf u,R)\in\mathbb R^4.
\]

时域状态保存

\[
\widehat{\mathbf a}=\widehat{E[R\mathbf u]},\qquad
\widehat b=\widehat{E[R]},\qquad
\widehat q=\widehat{E[R^2]},\qquad N_{\rm eff}.
\]

因为 \(\|\mathbf u\|=1\)，所以 \(\|Z\|^2=2R^2\)。由此可精确恢复编码向量的总体中心散布迹

\[
S_Z=2\widehat q-\|\widehat{\mathbf a}\|^2-\widehat b^2.
\]

若归一化时域权重为 \(w_i\)，且 \(N_{\rm eff}=1/\sum_iw_i^2\)，则在独立同分布、权重视为给定的条件下，

\[
\widehat V_{\bar Z}=\frac{S_Z}{N_{\rm eff}-1}
\]

无偏估计加权均值误差的总方差 \(E\|\bar Z-EZ\|^2\)。当 \(N_{\rm eff}\le1\) 时，单个状态不能估计采样方差，必须使用冷启动空间估计或返回零可信度。

## 2. 矩空间差异

两个 latent 状态的 decoder 无关距离取为

\[
d_M^2=\|\widehat{\mathbf a}_1-\widehat{\mathbf a}_2\|^2
      +(\widehat b_1-\widehat b_2)^2.
\]

它是原始 encoder 坐标的旋转不变欧氏距离。它不使用 MaxEnt 分布、MaxEnt 协方差、Gaussian proxy 或 Bures/Wasserstein 距离。

只保存 \(E[R^2]\) 无法识别四维协方差矩阵的形状，因此不能构造精确 Mahalanobis、Wald 或卡方统计量。实现使用的是“距离平方除以误差总方差”的迹归一化差异；在正确的平稳误差模型下其期望为一，但不能宣称其服从卡方分布。

## 3. 未存储交叉协方差的闭包

令估计器误差向量为 \(\varepsilon_A,\varepsilon_B\)，总方差为

\[
V_A=E\|\varepsilon_A\|^2,\qquad V_B=E\|\varepsilon_B\|^2.
\]

无法存储迹交叉协方差时，对每个 pass 或混合上下文 \(c\) 采用显式闭包

\[
E[\varepsilon_A^T\varepsilon_B]
\approx p_c\sqrt{V_AV_B}.
\]

因此差值总方差必须包含协方差项：

\[
V_{A-B}=V_A+V_B-2p_c\sqrt{V_AV_B}.
\]

这是迹协方差闭包，不等价于恢复完整协方差矩阵。一次空间 pass 最多混合九个估计器；若该 pass 的所有非对角项采用同一个 \(p_c\)，为保证等相关矩阵半正定，需要

\[
-\frac1{8}\le p_c\le1.
\]

不能从单像素保存矩中识别 \(p_c\)，但可以给它一个不依赖 \(\phi\) 的、可重复推导的固定核口径。令原始像素误差 \(\varepsilon_q\) 相互独立且具有相同迹方差 \(\sigma^2\)，并令 pass \(\ell\) 之前像素 \(x\) 的累计线性冲激响应为 \(h_x^{(\ell-1)}(q)\)，则

\[
V_x=\sigma^2\sum_q h_x(q)^2,\qquad
C_{xy}=\sigma^2\sum_q h_x(q)h_y(q),\qquad
\rho_{xy}=\frac{C_{xy}}{\sqrt{V_xV_y}}.
\]

对当前 pass 的中心与八个 tap 记固定核权重为 \(a_i\)，实现所需的两个标量分别定义为

\[
p_{\rm diff}=\frac{\sum_x\sum_{i=1}^8a_iC_{0i}(x)}
{\sum_x\sum_{i=1}^8a_i\sqrt{V_0(x)V_i(x)}},
\]

\[
p_{\rm prop}=\frac{\sum_x\sum_{i<j}a_ia_jC_{ij}(x)}
{\sum_x\sum_{i<j}a_ia_j\sqrt{V_i(x)V_j(x)}}.
\]

前者使中心–tap 差值的核权重平均交叉协方差匹配，后者使该 pass 的完整非对角传播项匹配。前三个规则网格 pass 平移不变，可由离散卷积精确求值；后三个 pass 使用 shader 的逐像素 whash 旋转，表中值为空间采样平均，但每个被采样位置的核重叠仍是精确计算：

| 上下文 | \(p_{\rm diff}\) | \(p_{\rm prop}\) |
|---|---:|---:|
| à-trous step 1 | 0 | 0 |
| à-trous step 2 | 0.1633227 | 0.1060087 |
| à-trous step 4 | 0.2170925 | 0.1411698 |
| à-trous step 8 | 0.2507545 ± 0.0000606 | 0.1394126 ± 0.0000329 |
| à-trous step 16 | 0.2584918 ± 0.0001035 | 0.1470511 ± 0.0000270 |
| à-trous step 32 | 0.2665217 ± 0.0003202 | 0.1525312 ± 0.0001094 |
| 最终输出相邻相关 | 轴向 0.94461、对角 0.94434 | 均匀双线性相位 0.94456 |
| 新时域样本–历史创新 | 独立采样假设 | 0 |

这些数值只由采样位置、固定核权重与 pass 顺序决定，因此不含 \(\phi\)、\(N_{\rm eff}\)、光照分布、几何或 MaxEnt。代价是它明确把信号与几何权重条件化为固定值；真实自适应滤波器偏离该线性化时，这些常数不再是其精确相关。既要求对任意 \(\phi\) 的真实自适应算子精确，又要求估计完全不依赖 \(\phi\)，二者不可同时满足。原来的全局 0.2 则连 pass 的累计核重叠也没有表达。

## 4. 任意权重的协方差传播

对

\[
\bar X=\frac{\sum_i w_iX_i}{W},\qquad W=\sum_iw_i,
\]

完整展开为

\[
V_{\bar X}=\frac{1}{W^2}\left(
\sum_iw_i^2V_i+2\sum_{i<j}w_iw_j\operatorname{Cov}_{\rm tr}(X_i,X_j)
\right).
\]

代入常相关闭包后，令

\[
Q=\sum_iw_i^2V_i,\qquad S=\sum_iw_i\sqrt{V_i},
\]

得到

\[
V_{\bar X}=\frac{(1-p_c)Q+p_cS^2}{W^2}.
\]

该式同时覆盖 à-trous、双线性时域重投影和镜面双分支混合。原来的分数阶幂传播没有对应的协方差模型，已弃用。

## 5. 时域创新量

在线性矩空间中，若当前递推状态

\[
T=(1-\alpha)H+\alpha X,
\]

则

\[
\|T-H\|^2/\alpha^2=\|X-H\|^2
\]

在未经过后续非线性处理的 encoder 时域状态上是严格恒等式。当前实现比较的是又经过空间滤波的信号，因此实际使用时还包含“当前与历史共享同一个局部固定权重空间算子”的线性化；数据依赖权重变化会破坏严格恒等。若历史均值含 \(N_{\rm eff}\) 个等效样本，使用 \(V_X\approx N_{\rm eff}V_H\) 与上述闭包可得

\[
V_{X-H}\approx V_H\left(N_{\rm eff}+1
-2p_{\rm innovation}\sqrt{N_{\rm eff}}\right).
\]

这替代了把当前值与历史值无条件视为独立的 \((N_{\rm eff}+1)V_H\)。

实现还显式建模降噪器内蕴的矩空间误差。令
\(\sigma_D^2=10^{-5}\) 为该误差的总方差（单位是矩坐标的平方），则实际观测到的降噪状态差异使用

\[
V_{T-H}=\alpha^2V_{X-H}+\sigma_D^2,
\qquad
D=\sqrt{\frac{\|T-H\|^2}{V_{T-H}}}.
\]

\(\sigma_D^2\) 位于 \(\alpha^2\) 外部，因为它描述时域混合之后由降噪、重投影及有限表示共同留下的附加误差，而不是原始创新量 \(X-H\) 的采样方差。它同时避免零分母只是这一统计模型的结果，不能把它解释为普通数值 epsilon；修改它会直接改变标准化差异和历史样本上限的尺度。

## 6. 适用边界

- \(\widehat q\) 必须与 \((\widehat{\mathbf a},\widehat b)\) 使用同一组时域权重更新。
- 公式的无偏性以独立同分布且权重条件给定为前提。若权重直接依赖同一批亮度样本，仍可能有选择偏差。
- 固定核推导针对所有 tap 均存在的图像内部。图像边界、几何拒绝和信号拒绝都会改变实际归一化核，统计式本身不能修复这种模型失配。
- \(p_c\) 无法由单个运行状态的保存矩识别；表中数值由指定固定核推导。若要得到精确多维显著性检验，必须额外保存足够的二阶矩，例如 \(E[R^2\mathbf u]\) 与 \(E[R^2\mathbf u\mathbf u^T]\)。
- 每个 pass 内仍用一个标量近似不同 tap 对。\(p_{\rm diff}\) 与 \(p_{\rm prop}\) 是不同加权投影，不能互换；若该近似仍不够，必须把 tap/offset 类别纳入参数或保存交叉协方差。
- 镜面 surface/virtual 分支相关性取决于两个运动投影的实际间距，当前用固定核相邻输出值 0.9446 作为局部重叠近似；这不是由现有状态或固定核本身唯一确定的量。
- 当 \(p_c=1\) 且 \(V_A=V_B\) 时，闭包必然给出 \(V_{A-B}=0\)：模型此时声称两项误差完全相同，任何非零差异都应被拒绝。这是闭包端点的数学退化。
