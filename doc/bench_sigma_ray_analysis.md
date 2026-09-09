# 预计算射线与未知 sigma 的实测复核

后续在允许少量精度损失的条件下完成了历史提交消融、oct32 逐字段误差及八个 pass 净链计时，见 [提交消融与 oct32 净收益](bench_culprit_analysis.md)。该后续记录补齐了本报告尚缺的历史主因证据，未证明生产回归已经解决。

本轮结果支持将 **未知 sigma 的处理列为次因**，不能用它解释 `_9` 到 `_11` 前五级约 2.04 倍的周期差。`geometry.z` 确实已经空闲，容量并不妨碍预计算射线；但存储表示、解码访存、虚拟法线差分的精度和寄存器寿命需要一起验证。

本轮正式着色器只修改了共享估计器的饱和舍入边界，并补充统计量注释。性能候选和全部失败结果保留在 `temp/bench_sigma/`。**尚未证明 `_9` 的性能回归已解决。** 大核的逐像素 Poisson 旋转保持不变，两条路径继续使用统一降噪算法。

## 1. 实验范围与计时方法

- 本轮基线是 `temp/bench_sigma/baseline_shaders`，已经包含上一轮小核 FP32 射线共享缓存；不是 `_9` 的历史源码。
- 设备为 RTX 4060 Laptop / NVIDIA 610.62。连续空间链使用 1920×1080、两类合成输入、两域五级或六级派发；正式对照为 31 对 AB/BA。GPU 恢复输入在 query 外，完整批次结束后才读取 query 和输出。
- 独立 sigma 派发分别测 resolver-only 和九样本 accum-resolve，使用动态 SSBO 输入和可观察输出。另以 16 次动态内部计算减少输入带宽的相对影响。没有用可被编译器删掉的空循环计时。
- 独立派发含其输入输出、控制指令和 barrier，不是纯 ALU 延迟。1080p 原始输入较大，尤其可能受带宽限制；小工作集重复计算也不能代替真实 pass。
- AMD gfx1100 的 RGA 只用于相同工具链下的 ISA、循环、寄存器和 LDS 对照，不代表 NVIDIA 的真实寄存器或占用率。

`tools/benchmark_spatial_chain.py` 现用生产 GPU 编码函数在 query 外填充几何射线字段。旧版 fixture 的 z=0 无法正确测试读取预计算方向的候选。两棵树获得同一份输入，旧树忽略该字段；实验性的 view-scale 表示也由其生产函数生成。

## 2. 未知状态与正常统计不能混为一项成本

`0dac118` 同时引入/调整了 estimator 方差传播和未知状态协议。正常恒定相关性模型本来就需要：

```text
A = sum_known((w*sigma)^2)
S = sum_known(w*sigma)
K = sum_known(w)           // donor 额外状态
Q = sum_unknown(w^2)       // donor 额外状态
W = sum_all(w)            // 同时是光照归一化权重
d = S/K
V = ((1-rho)*(A+Q*d*d) + rho*(S+(W-K)*d)^2) / W^2
```

`sigma=-2` 既可能来自数值信息丢失，也可能来自非零单样本无法估计方差；历史失效是另一个状态。不能把 unknown 当零噪声，也不能把 A/S 的正常工作全部算作异常处理开销。既有 Jensen/cone 容差通过了 30,000 个单次 FP16 边界样本；没有证据说明普通单次舍入普遍制造 unknown。详见 [起源分析](../temp/bench_sigma/origin/report.md)。

在**全部有效 sigma 已知**的控制输入中，彻底删除 K/Q、对应分支和 donor 归约，五级链得到以下配对速度比（baseline/candidate）：

| 输入 / 路径 | 配对速度比 | 消费字段 |
|---|---:|---|
| coherent / diffuse | 1.0135 | 逐位一致 |
| coherent / reflection | 0.9956 | 逐位一致 |
| mixed / diffuse | 1.0248 | 逐位一致 |
| mixed / reflection | 1.0202 | 逐位一致 |

这约是 **慢 0.44% 到快 2.42%**，且只针对这些输入，不是任意场景的成本上界。该诊断不能处理未知输入，未采用。独立派发也未显示未知状态导致数量级退化：小工作集、16 次动态计算的九样本原始空间统计约为已知 42–43 µs、一项未知 43 µs、混合 43 µs、全部未知 38 µs。它们包含共同的动态控制工作，不能乘以像素数换算游戏耗时。

单独加全已知 fast 分支或解析折叠没有稳定改善连续链。RGA 显示 fast 分支没有降低 VGPR，反而增加部分指令；branchless 52/54 的 VGPR 从 62/69 增到 70/72。详见 [控制实验](../temp/bench_sigma/known_only_control.json)、[独立派发](../temp/bench_sigma/microbench/repeated/report.json)、[1080p 独立派发](../temp/bench_sigma/microbench/native_1080/report.json)。

## 3. 已修复的确切数值问题

有效输入的 sigma 与 donor 均不超过 M=65504。将 unknown 替为现有 donor 后，非负加权项满足 `sum(t_i^2) <= sum(t_i)^2`，因此任意 `rho in [0,1]` 的准确归一化估计均满足 `V <= M^2`。

原函数会把合法混合结果的微小 FP32 上界超调当成 unknown。本轮仅在 `denoiserResolveEstimatorSigma` 的最终转换中接受 `V <= M^2*(1+1e-5)` 的有限小幅超调，并投影回上界。累加、donor、mix、归一化的求值顺序全部保留；通用 `denoiserVarianceToSigma` 输入检查仍然严格。全 unknown、NaN、Inf、负方差及更大越界仍失败。

现有调用最多九项，普通 FP32 算术的保守误差预算小于该容差。极小 K 的 FTZ/下溢损失不在该证明内，本次没有用任意 clamp 隐藏它。详见 [上界证明与实际调用契约](../temp/bench_sigma/cap_only_proof.md)。

实际 GPU 的 48,406 组输入中，饱和双线性组 12,000 例原先有 **2,717** 个错误 unknown，现在为零；极小 donor 组另修复 25 例。所有原已知结果 FP32 逐位不变，严格方差入口不变。这个比例是专门构造的边界测试，**不是游戏中的发生频率**。测试入口是 [audit_estimator_sigma_gpu.py](../tools/audit_estimator_sigma_gpu.py)，结果见 [GPU 记录](../temp/bench_sigma/cap_gpu_check.json)。

## 4. 射线空间足够，但直接编码并非免费

`geometry.z` 的唯一有效生产者是共享 variance preparation；两个域都可以写入预计算方向。主 RT gbuffer.z 的纹理法线属于另一资源，不能混用。候选同时修改 50/65 和全部相应消费者，额外准备成本不能从整链预算里消失。

实测候选包括直接 oct32、小核缓存 oct32 解码结果、仅大核中心/tap 使用 oct32，以及 FP32 view-ray 归一化尺度配合解析像素坐标重建。最后一种避免方向量化，但增加参数存活和算术重排，测试中反而更慢。

真实 GPU 射线 probe 中，oct32 最大方向误差约 **0.00369°**；经过实际虚拟位置及四邻居法线差分，最大法线误差约 **6.06°**。FP32 scale 的最大方向误差约 0.0000158°，对应法线误差约 0.0231°。所以只报告编码射线角误差不足以评价降噪器误差。

仅大核使用 oct32 的五级链约快 3%–6%，但改变大量输出；mixed 输入的相对误差对局部接近零的分量尤其敏感，不能只用平均误差或将其描述为少量舍入差。该候选也未改变大核 Poisson 采样方式。这些方向预计算候选均未采用，geometry.z 仍空闲。详见 [方向/法线 probe](../temp/bench_sigma/ray_probe/report.json)、[仅大核 coherent](../temp/bench_sigma/large_only_coherent.json)、[仅大核 mixed](../temp/bench_sigma/large_only_mixed.json)。

## 5. 正常 Bures 热路径的额外工作

旧 small 曾把端点轴统计放进 LDS，因此普通每 tap 是 1 个标量 sqrt 加 1 次归一化除法；旧 large 每 tap 则是 4 个 sqrt 加 1 次除法。当前两者为 3 个 sqrt 加 2 次除法，另有条件 cone 修复。**只有 small 能据此明确说失去了端点复用，不能把所有 large 退化都归因于根号。** 两条 estimator 仍共用一次 proposal 权重计算。

本轮测试了保留稳定有理化分子、将 `(N/H)/S` 合并为 `N/(H*S)`：RGA 证实每个 known tap 少一次 RCP，但 52/54 各多一个 VGPR；连续链结果介于约慢 0.24% 与快 2.19%，且有 FP32 重关联传播到输出的差异，未采用。

另测试 small 的共享 q 缓存：保留实际 FP16 moment 的原始 q 结果及 cone 修复标记，不改 ABI、相关性或权重。两个输入、两个域的五级结果均逐位一致，但 **慢约 1%–4%**。RGA 显示 LDS 增长，52 又触发不同循环展开。这说明恢复一项预计算并不等于恢复旧版整套数据流的收益。

公式、源代码操作计数和输入范围详见 [metric 对照](../temp/bench_sigma/metric_cost.md)；实际 RGA 指令及自然循环分析见 [最后静态批次](../temp/bench_sigma/rga/final_batch/report.md)。

## 6. 正式修改验证与结论边界

- 82 个着色器入口，默认、EON 关闭及其它功能关闭三组编译/SPIR-V 检查全部通过，共 246 项。
- 17×9、127×73、1920×1080 的完整 temporal → preparation → 六级 spatial → persistent resolve 链，每组 37 项检查通过；普通信号、sigma、几何和历史交接保持逐位一致。
- 估计器方差、Bures、MC 合约、uncertainty ingress 与方差精度检查通过；饱和边界由新增 GPU oracle 单独覆盖，未放宽旧链的 sigma 比较规则。
- 最终生产代码的两域五级/六级、两类输入共八组 31 对测试，消费字段均逐位一致；配对速度比为 0.9849–1.0170。这是约百分之一的变化范围，**不作为加速宣称**。RGA 的 51/52/54 VGPR、LDS、RCP 与本轮基线相同。

最终计时文件为 `temp/bench_sigma/final_{coherent,mixed}_{5,6}.json`。这次落地的是数值纠错、可复现的独立派发测试与更精确的原因排除，不能宣称游戏帧时已恢复。

历史抓取的主因仍需在完整数据流内闭合：旧单流与当前双流的有效工作、缓存内容、寄存器寿命、延迟隐藏，以及 50/65 持续池化的成本彼此耦合。现有实验排除了“仅 unknown donor 很贵”这一充分解释，也没有证明其中任意剩余项单独解释全部差额。原始周期与资源指标继续以 [历史回归报告](bench9_regression_analysis.md) 为准。

复现示例（临时输出均在 temp）：

```powershell
python -X utf8 -B tools/audit_estimator_sigma_gpu.py --baseline-dir temp/bench_sigma/baseline_shaders
python -X utf8 -B temp/bench_sigma/benchmark_sigma.py --gpu --variants all_known_fast analytic_guarded --count 2073600 --inner 1 --trials 31 --dispatches 5 --correlations 0.14262104 --label native_1080
python -X utf8 -B tools/benchmark_spatial_chain.py --baseline-dir temp/bench_sigma/baseline_shaders --fixture coherent --steps 5 --trials 31 --output temp/bench_sigma/recheck.json
```
