# 历史提交消融与 oct32 净收益

原始抓取场景不完全相同，本轮使用固定输入、匹配各历史 ABI 的实际 GPU 派发建立对照。`_9` 的文件时间只能圈定历史候选，不能证明其精确源码哈希；下面定位的是可复现的提交成本增量，并未完成同一游戏帧的历史回放。

主要记录：[历史提交对照](../temp/bench_culprit/history_balanced_coherent.json)、[射线单项消融](../temp/bench_culprit/history_ray_isolated.json)、[八个 pass 净链](../temp/bench_culprit/net_oct_1080.json)、[oct32 逐字段误差](../temp/bench_culprit/oct_quality.json)、[RGA 指令分析](../temp/bench_culprit/rga/findings.md)。其余未带路径的实验文件均位于 `temp/bench_culprit/`。

本轮证据已经支持用户关于明显退化的判断，主要原因是**独立 current 双流实现和重复视线重建**。`fce9cd0 → 0219372` 引入双流，使受控 51 耗时增加约 **87–91%**、54 增加约 **78–80%**；紧接着 `0219372 → 933ec15` 的组合改动，使 51 再增加约 **56%**、54 增加约 **15–17%**。在排除 PDF 字段别名/CSE 差异的 933 控制中，恢复 oct 视线使 51 耗时减少 **42.43%**、54 减少 **6.05%**，把 933 小核的主要增量指向重复视线重建。这些百分比不能相加，也不能直接换算成游戏整帧退化比例。

当前证据不支持把主要原因概括为“unknown sigma 太复杂”或“FP32 累加太慢”。双流大幅跳变发生时，两侧都使用原有 FP16 光照累加、oct32 视线、相同逐像素 Poisson 调度；在 933 上仅恢复 FP16 后，51 只减少约 4.24%，54 反而增加约 2.99%。当前生产已经采用 FP32 shared ray cache，不能将历史恢复 oct 的 42.43% 小核收益沿用到当前代码；当前 oct 的收益需要单独测试。

**本轮尚未采用任何 shader 候选。** 下文区分了历史归因、保持输出的存储诊断、主动改变算法的成本诊断，以及准备供性能/质量选择的候选。所有树均在 `temp/bench_culprit`，未将这些实验写入生产代码。

当前 oct 的实际八阶段净链耗时变化只有 diffuse **-2.08%**、reflection **-0.73%**，95% 区间均跨零；同期独立 A/A 对照约有 +2.66%/+2.93% 的偏移。因此本轮确认了历史原因，但没有证明当前 oct 候选有足够稳定的净收益，不默认合入。

## 1. 计量口径与可靠性

设备为 RTX 4060 Laptop GPU，OpenGL 驱动报告 `4.3.0 NVIDIA 610.62`，分辨率均为 1920×1080。本轮数值来自实际 NVIDIA GL 执行，不从 AMD RGA 的 VGPR/SGPR 数量推导 NVIDIA 性能或寄存器分配。

本报告统一使用**耗时变化**：负数表示耗时下降，正数表示增加。原 JSON 的速度比定义为 `S = sqrt(T_A1*T_A2)/T_B` 时，本文换算为 `100*(1/S-1)%`；速度比区间 `[L,U]` 换算为 `[100*(1/U-1), 100*(1/L-1)]%`。A/A 列为第二个相同源码资源相对第一个的耗时变化。单独列出的绝对 ms 是边缘中位数，不能相除替代配对比值中位数。

历史平衡实验和当前链路平衡实验都使用两个相同源码、不同资源的基线 A1/A2，加一个候选 B。每个随机区组覆盖三个对象的全部六种执行顺序；11 个区组共 66 组。查询结果在整个计时批次结束后才读取，输入恢复在查询外。历史每次查询含四次固定输入 dispatch；链路查询含真实串联的指定空间阶段。区间来自 11 个区组中位数的 10,000 次 bootstrap，点值则使用所有组的比值中位数，两者统计口径有细微区别。

该设计降低顺序偏置并显示部分漂移，但不会消除驱动驻留策略、温度/频率状态或跨实验批次漂移。A/A 不是始终精确零：历史 constant 51 的 A/A 区间达到 `[-6.79%, +16.16%]`；coherent 54 的 A/A 为 `+1.68%`、区间 `[+0.48%, +4.08%]`。小于或接近这类变化的候选不能只凭一个正向点估计宣称有效。

当前链路的 `0:3` 表示前三个小核，`3:6` 为后三个大核，`0:5` 为前五级，`0:6` 为全部六级；diffuse 分别对应 51–53、54–56，reflection 对应 66–68、69–71。部分范围的前缀虽然在查询外执行，却是各候选运行自己的前缀，不能默认数值输入完全相同。`oct_large` 的前三小核未改，已有质量记录确认该前缀逐位一致；联合诊断和 FP16 候选则不能作此假设。

所有实验均为合成输入的单 pass 或 denoiser 子链，不是游戏帧率、同帧 RenderDoc/RGP 回放或完整渲染管线时间。历史 YAML 的 GR 周期另有计量域，不能与这些 ms 混合求和。

## 2. 主因：021 独立 current 双流实现的完整成本

来源：`history_balanced_constant.json`、`history_balanced_coherent.json`；确切源码差异和 ABI 见 `0219372_spatial.patch`、`ABI_AND_PAIRS.md`、`history_manifest.json`。

| 输入 | 相邻提交 | pass | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---|---:|---:|---:|---:|
| constant | fce9cd0 → 0219372 | 51 | +90.86% | [+86.35%, +96.36%] | +3.15% |
| constant | fce9cd0 → 0219372 | 54 | +77.86% | [+75.84%, +83.42%] | -1.21% |
| coherent | fce9cd0 → 0219372 | 51 | +87.05% | [+76.37%, +94.23%] | +0.09% |
| coherent | fce9cd0 → 0219372 | 54 | +80.14% | [+75.13%, +84.21%] | +1.68% |

这一相邻对的两侧均保留 FP16 光照累加和 1/8 缩放、oct32 primary ray、逐像素旋转及同样的八 tap、大核 8×8/小核 16×16 工作组；早期 small stddev LDS 在两侧都已删除。因此，FP32 恢复、射线重建、group rotation 改变和旧 LDS 移除不是此提交对的隐藏差异。

新增工作包括中心和每个接受 tap 的 independent-current 读取、有效性判断和解码，第二套光照/色度/方差/权重/距离累加，以及第二个 resolve/sanitize/store。proposal 权重和几何仍由两流共用。完整内点最多新增 9 次 16B 读取和 1 次 16B 写入，相当于 1080p 约 316.4 MiB 的逻辑请求量；这是缓存之前的载荷计算，不是实际显存流量。额外存活状态、调度和编译分配的代价也包含在观察到的提交增量里，尚未与纯算术独立分离。

constant fixture 使用相同量化 moment 和已知 sigma，真实 light distance 为零，避免提交中 phi 默认值变化影响权重。两侧 geometry/signal hash 相同，proposal 输出逐位相同，新 independent 输出也等于该 proposal；所有有效像素覆盖 poison，sigma 从输入 1.25 降为邻域混合后的值。因此旧 shader 不是全拒绝或只复制中心的假快路径。coherent fixture 同样观察到大幅退化，但提交还改变 phi 等细节，proposal 输出不再逐位一致；它是空间变化输入上的支持证据，不是数值完全等价的单变量证明。

因此最强结论是：**021 中独立 current 的完整实现是已定位的主要历史性能增量。** 这不等价于“必须移除第二个估计器”，也不能把当前已更换存储、角色裁剪和统计策略的代码，直接视为旧 021 的同一成本。

## 3. 预读陷阱与存储次因

早期 `history_constant.json` 在 warmup 前读取过输出 SSBO，021 的 51/54 边缘中位数曾达到 `9.170944 / 52.151296 ms`。只取消该输出预读、仍保留 GPU 初始化及批次后的输出验证后，`ssbo_no_pre_read.json` 中对应数值为 `1.085440 / 2.384896 ms`。这说明该预读显著扰动了实验；旧 9/52 ms 数据**不得用于提交成本归因**。

这些是不同批次，不能用其商估算一个精确的“读回惩罚”。没有采集 residency、PCIe 或迁移计数器，当前只证明对预读敏感；“CPU 读回改变驱动驻留策略”仍是机制假说。源代码有效、输出正确，并不能排除测量自身改变资源状态。

`ssbo_no_pre_read.json` 的两个受限存储干预保留 021 两个估计器及算法，所有输入和两类输出均逐位一致：

| 021 上的干预 | pass | 配对耗时变化 | 限制 |
|---|---:|---:|---|
| independent 输入 SSBO load 改为独立 RGBA32UI sampler；保留原 SSBO 输出与分配 | 51 | -4.39% | 单批 AB/BA，时间比 IQR 跨零改善边界 |
| 同上 | 54 | -8.61% | 同上 |
| current plane12/13 改为 plane0/1，并缩小分配 | 51 | -0.76% | 同上 |
| 同上 | 54 | -2.74% | 同上 |

这些较小正向点值未经过本轮相同的双基线区组确认，其 IQR 都跨过无变化。它们不能证明 SSBO 读取或大偏移解释了双流的全部 78–91% 增量；相反，保留第二套估计器后，大部分成本仍在。更完整说明见 `SSBO_DIAGNOSTICS.md`。

## 4. 另一个明确历史跳变：933 组合改动

来源：`history_balanced_constant.json`、`history_balanced_coherent.json`、`933ec15_spatial.patch`。

| 输入 | 相邻提交 | pass | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---|---:|---:|---:|---:|
| constant | 0219372 → 933ec15 | 51 | +55.96% | [+51.62%, +59.96%] | -1.13% |
| constant | 0219372 → 933ec15 | 54 | +14.95% | [+11.54%, +17.87%] | -0.09% |
| coherent | 0219372 → 933ec15 | 51 | +56.47% | [+54.19%, +59.01%] | -1.33% |
| coherent | 0219372 → 933ec15 | 54 | +16.65% | [+14.74%, +19.47%] | -0.99% |

933 同时恢复两套 FP32 光照/色度累加、把 c3.z 从 primary ray 改为 PDF 方向、改用中心/采样投影重建视线，并改变 PDF 与 moment 距离相关权重。对应 ABI 不同，因此 fixture 按各自源函数编码相同场景含义，原始 geometry hash 不相同；输出也不要求相同。这是提交整体成本，不是射线或精度的纯消融。

`history_half_balanced.json` 在 933 上只恢复原有 1/8 缩放的 native FP16 光照累加，其他计算不变：

| 干预 | pass | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---:|---:|---:|---:|
| 933 → 仅光照累加恢复 FP16 | 51 | -4.24% | [-6.18%, -1.83%] | +1.04% |
| 同上 | 54 | +2.99% | [+1.60%, +5.51%] | +0.00% |

恢复 FP16 的小核收益远小于历史小核增量，且大核方向相反，因此不能用 FP32 光照累加解释 933 的主要退化。该半精度候选会改变输出，只是成本控制，不是逐位替换。

### 933 视线重建拆分及 PDF 字段别名控制

恢复 oct 的树为了腾出 z，把 PDF 方向映射到 y，并依赖该 fixture 的 PDF direction 等于 surface normal。直接拿原 933 与该树比较，会同时改变 PDF 字段读取和可能的中心解码 CSE。因此另构建 `933_pdf_alias`：PDF 同样读 y，但保留原有 ray reconstruction，再与 oct 树配对。

| 相邻实验对象 | pass | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---:|---:|---:|---:|
| 原 933 → oct ray + PDF 读 y | 51 | -43.19% | [-43.37%, -41.51%] | -0.13% |
| 同上 | 54 | -7.07% | [-8.98%, -4.75%] | +0.32% |
| 933 PDF 读 y / reconstruct ray → 同 PDF 读 y / oct ray | 51 | -42.43% | [-43.51%, -39.34%] | +0.08% |
| 同上 | 54 | -6.05% | [-8.65%, -4.00%] | -0.05% |

来源：`history_oct_balanced.json`、`history_ray_isolated.json`。第二组把 PDF 读取放在两侧相同的字段，仍出现约 42% 小核耗时下降。因此重复视线重建是 933 小核增量的主要来源，不能用“只是 PDF 解码 CSE”解释。大核对应收益较小，不能拿小核结果套用到所有空间 pass。

这些树改了 geometry.z 的 ABI，fixture hash 不同是预期结果；oct 也改变方向精度，proposal/current 输出均非逐位一致。该控制是在明确 PDF==surface normal 的 fixture 上拆分成本，尚不是适用于任意 PDF 方向的通用生产布局。原 933 的每 tap 重建与当前已缓存小核射线的实现不同，历史收益不能作为当前 oct 候选的性能承诺。

## 5. 当前 oct32：小幅空间收益，需要单独核对净链和质量

以下均为 `oct_large`：small 维持原 FP32 ray tile；large center/tap 从 c3.z 解码 oct；large 四邻居 virtual-position helper 保持原重建。两域仍走相同模块。50/65 增加的编码未包含在这些空间范围内。

| 输入 / 域 | 范围 | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---|---:|---:|---:|
| coherent / diffuse | 前三小核 | +3.37% | [-0.21%, +8.78%] | -0.95% |
| coherent / diffuse | 后三大核 | -5.49% | [-6.35%, -3.82%] | -0.02% |
| coherent / diffuse | 前五级 | -4.84% | [-6.75%, -2.48%] | -1.14% |
| coherent / diffuse | 六级 | -2.09% | [-3.05%, -0.18%] | -1.71% |
| mixed / reflection | 后三大核 | -4.00% | [-5.69%, -0.46%] | -0.54% |
| mixed / reflection | 前五级 | -4.43% | [-6.41%, -0.35%] | -1.18% |
| mixed / reflection | 六级 | -2.69% | [-5.35%, -0.99%] | +0.90% |

来源分别为 `oct_balanced_coherent.json`、`oct_balanced_mixed.json`。域与 fixture 同时不同，不能把两行之差解释成纯材质或纯路径差异。small 代码未改变、已有数值结果也未改变，却仍测到包含零的 +3.37% 点值，这本身说明小幅差异要结合控制和区间解读。

50/65 已有 `decodeNormalU(encodeNormalU(primaryRay))` 的 tile 表达式，但原 uint 没有保留，且源 moment/Neff guard 比最终 geometry-valid store 窄。不能假设新增 c3.z 编码免费。空间表不支持整 denoiser 或整帧同幅提速；实际净链结果如下。

### 包含准备、六空间级和 resolve 的净链

`net_chain.py` 每次查询执行实际八个 dispatch：diffuse 是 `50 + 51–56 + 58`，reflection 是 `65 + 66–71 + 72`，包含生产 memory barriers。raw/temporal 前缀、GPU 输入恢复与 CPU readback 都在查询外。每个随机区组为 ABBA 或 BAAB，使用区组内各侧几何均值之比及区组 bootstrap 区间；这与前述三对象六顺序设计不同，没有同步的第三个 A/A runner。

| 八阶段实验 | 域 | 耗时变化 | 区组 95% 区间 |
|---|---|---:|---:|
| 相同基线 A/A | diffuse | +2.66% | [-0.01%, +6.21%] |
| 相同基线 A/A | reflection | +2.93% | [-1.88%, +5.62%] |
| baseline → oct_large | diffuse | -2.08% | [-6.04%, +4.08%] |
| baseline → oct_large | reflection | -0.73% | [-6.32%, +2.17%] |

来源：`net_aa_1080.json`、`net_oct_1080.json`。两对象的固定输入 hash 一致；每个对象在计时后 restore/replay 均逐位再现原输出。A/A 的 independent 和全部拥有的输出 plane 都逐位一致。候选输出差异被如实保留，执行成功不表示接受其精度。

当前 oct 的净区间跨零，点值也没有超过这组控制显示的波动，**不能证明稳定净收益**。GPU restore 会使缓存状态不同于游戏真实 raw/temporal 生产者；每个对象仍有约 0.6 GB 的资源，资源量及恢复字节数已记录，不能将此单域结果外推整帧。17×9 记录用于边界、输入恢复与重复输出检查，其短时调度结果不用于推导 1080p 收益。

### 已有 oct 质量记录

`oct_quality.json` 是不计时的实际生产 denoiser chain，覆盖两个候选各 37 项捕获。50/65 proposal 与 current 信号逐位一致；几何 x/y/w 精确一致，invalid z 为零，valid z 与实际 GPU encoder oracle 零失配。编码契约正确不表示解码视线无损。

| 候选 / 最终输出 | Y 相对 L2 | Y 相对误差 P99 / 最大 | sigma 相对误差 P99 / 最大 |
|---|---:|---:|---:|
| oct_large / diffuse | 0.376% | 2.09% / 22.85% | 2.37% / 26.27% |
| oct_large / reflection | 0.371% | 2.03% / 21.84% | 2.34% / 26.20% |
| oct_full / diffuse | 0.426% | 2.23% / 23.28% | 3.41% / 27.96% |
| oct_full / reflection | 0.423% | 2.18% / 22.19% | 3.40% / 27.97% |

large 的最终 Y 相对误差 P99.9 约 5%，说明平均损失小而局部尾部明显。large 色度向量误差除以端点亮度后 P99 约 0.45–0.46%；该 fixture 却固定 `Co=0.1Y, Cg=-0.2Y`，不足以验证独立色彩边界。全链被消费 VD 的最大绝对误差为 large `0.06640625`、full `0.080078125`。

所有捕获都没有 sigma 状态迁移、malformed sigma 或参与比较字段的非有限值。有效像素为 2,025,379，invalid 为 48,221；reflection 只有一个 unknown 像素，不能据此称全面覆盖 unknown 传播。

数值检查仍有限：未量化被消费的 `maxEntY.xyz`；最终 diffuse/specular 只详细解码 plane7/plane3，其余持久 plane 只有整个 buffer 是否相同的布尔；没有多帧闪烁、真实游戏图像或同输入虚拟法线的 GPU 分组误差。large 普通虚拟位置的射线表达式保留，但退化 normal fallback 会用 oct centerRay，后续 VD 差异也会传播到法线。因此不宜用一个角度量化上界代替输出质量审查。

后补的净链记录将 owned output plane 分开作 uint 比较，并给光照 plane 的全部 half 聚合差异及非有限计数；这补足了“哪些 plane 改变”的定位，但仍不等于对 directional moment 三个分量、能量与 sigma 的逐语义质量界。因此上面的 Y/CoCg/sigma/VD 质量结论仍以 `oct_quality.json` 为准，不能拿所有 half 的一个最大误差替代它。

## 6. 联合诊断：成本确实可以联合减少，但输出语义被主动改变

两个候选均以当前基线为比较对象，输入为 `all_known`、域为 diffuse。

| 诊断候选 | 范围 | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---|---:|---:|---:|
| group rotation + large oct + 完全移除 current 读取/累加 | 前五级 | -41.95% | [-42.80%, -38.40%] | -0.09% |
| 同上 | 六级 | -44.05% | [-47.10%, -41.83%] | +0.35% |
| 上述诊断再去掉完整 Bures 光照拒绝项 | 前五级 | -48.78% | [-50.27%, -47.86%] | -0.20% |
| 同上 | 六级 | -53.80% | [-54.05%, -53.12%] | +0.96% |

来源：`joint_balanced.json`、`joint_no_bures_balanced.json`。第一个候选不仅读 tap，还删除 center current 读取/有效性判断，最后用 proposal 复制到 current scratch；这明确破坏独立 current 语义。group rotation 改变采样关联和可能可见的网格结构，oct 改变几何精度。第二个候选完全删除光照拒绝，改变权重和后续输入。六级版本为了给占位 current 输出提供信号还恢复了末级 proposal 累加，生产代码原本不需要它。

因此它们是联合成本诊断，不是可直接合入的优化。两个 JSON 都各自对生产基线计时，基线绝对时间也发生变化；不能直接把 -48.78% 减去 -41.95% 当作 Bures 的独立可回收成本，更不能把此前 rotation/current/oct 的各自百分比相加。现有结果支持资源访问、状态存活和采样局部性具有联合影响，但尚未给出可保留原语义的同等收益实现。

源码中没有 weight==0/epsilon 后跳过累加的捷径，exp 下溢为零也执行已接受 tap 的累加。执行分支主要来自 bounds、geometry/signal/current validity、Neff validity 与 known-sigma。因此不能把输出非零权重数量当成执行 tap 数量；这些诊断的输入值与前缀变化仍需单独考虑。

## 7. 当前安全结构候选与 FP16 复验均无稳定链路收益

`metric_words` 复用准备后 proposal.z 保存最终 FP16 moment 对应的 FP32 q 和 cone-repair 标记；raw/temporal/current/persistent 的色度保持原 ABI。源码没有新增纹理对象、取样调用或 LDS，但已有取样开始消费 z 分量，不能认为返回数据的成本免费；原函数的 post-repair q 得以保留。`current_half_light` 则只把光照/色度累加改为 native FP16、使用 1/16 缩放；sigma/VD/权重计算本身仍为原代码。

| 候选 | 范围 | 耗时变化 | 区组 95% 区间 | A2 对 A1 耗时变化 |
|---|---|---:|---:|---:|
| metric_words | 前三小核 | -5.62% | [-11.74%, +0.42%] | +0.90% |
| metric_words | 后三大核 | -0.57% | [-1.91%, +1.24%] | +1.76% |
| metric_words | 前五级 | -0.30% | [-2.26%, +1.98%] | +1.80% |
| metric_words | 六级 | -0.86% | [-2.76%, +0.12%] | -0.46% |
| current_half_light | 前三小核 | +2.75% | [+1.70%, +6.45%] | -1.17% |
| current_half_light | 后三大核 | -0.65% | [-5.51%, +2.62%] | +1.42% |
| current_half_light | 前五级 | -1.32% | [-3.41%, +1.11%] | +1.23% |
| current_half_light | 六级 | -1.05% | [-2.68%, +0.19%] | +0.74% |

来源：`metric_balanced.json`、`current_half_balanced.json`；两者均为 coherent/diffuse。metric 的前五/六级区间跨零，且未加 50 准备成本，没有证明净收益。current half 的 small 明显变慢，整体小幅点值区间跨零，不能作为有效优化。

`balanced_chain.py` 只执行计时，不调用 `read_outputs`；因此这些 JSON 不证明 GPU 输出逐位正确。metric 的 CPU 20,006 条记录恢复 v/w/q 逐位一致，但跨实际 shader 的编译融合仍待验证；current half 的 CPU HDR 界和数值样例证明给定权重范围内不会溢出，不证明图像质量。由于未观察到有效链路收益，本轮不继续为这两项增加 GPU 质量验证，也不采用它们。

另测了只把 independent current 解包推迟到 weight/proposal 工作之后的 `late_current_unpack`，保留原始四个 uint 到最后使用，尝试缩短解码值存活期。`late_current_check.json` 的前五级在两个域均验证 proposal/current 原始输出逐位一致；`late_current_balanced.json` 的 diffuse/coherent 平衡耗时为前五级 **+0.49%**、区间 `[-2.24%, +3.22%]`，六级 **-1.17%**、区间 `[-3.28%, +1.09%]`，A/A 分别约 -0.34%/-0.16%。正确性成立，但未见稳定收益，因此也未采用。

## 8. 为什么目前不把 sigma 当作主因

除提交时间先后外，还有直接删除 unknown 专用工作的控制。既有 `temp/bench_sigma/known_only_control.json` 在全部有效 sigma 均已知的输入上，删除 K/Q 累加、unknown 分支和 donor 归约，保留正常已知统计所需的平方和、加权 sigma 和权重。两类 fixture、两个域的前五级输出均逐位一致：

| 全部有效 sigma 已知的控制输入 | 域 | 删除 unknown 专用工作后的耗时变化 |
|---|---|---:|
| coherent | diffuse | -1.33% |
| coherent | reflection | +0.44% |
| mixed | diffuse | -2.42% |
| mixed | reflection | -1.98% |

这是上一轮 31 对 AB/BA 的实际链路记录，基线与本轮冻结树不同，不能把其毫秒数与本轮拼接。变化约 -2.42% 到 +0.44%，支持“unknown 专用处理不是大幅退化的主要充分解释”。该诊断不能处理未知输入，也不是任意场景的成本上界；不能把所有正常 sigma 传播工作都算作 unknown 开销。更完整合同见 `doc/bench_sigma_ray_analysis.md` 第 2 节。

已确认的大幅 021 跳变是在旧有 sigma/统计路径上增加完整第二流；两侧并没有从 FP16 换成 FP32，也没有引入当前的 known/unknown 协议。后续 `892a822`/`0dac118` 的 Bures、estimator variance 和 unknown 改动比 021/933 晚，不能解释此前已经出现的两次大幅跳变。

联合 no-Bures 对照删除的是 endpoint preparation、距离/归一化、外层开方及整个光照拒绝项，不是“仅删除 sigma 传播”。它仍保留空间不确定性累加/resolve，所以不能拿该对照给 sigma 单独计时。q metadata 候选也只是 endpoint 公共计算缓存，整体不显著不能推出所有 metric 指令免费。

当前最有证据的两项主因是：021 的独立 current 完整实现，以及 933 小核中的重复视线重建。unknown 协议不是这两次已定位增量的主因。存储路径/偏移、采样局部性、额外存活状态和其余数学算术是还需进一步拆分的成本；保留统一架构和两个估计器语义，是寻找可采用实现的约束。

## 9. 与游戏 bench 的连接及尚缺证据

`profile_evidence.json` 保存原始 `_9/_10/_11` 计数器。以 54 为例，GR elapsed 从约 1.812M 周期升到 4.416M/4.348M，CS occupancy 从 56.60% 降到 32.14%/28.29%，L1 hit 从 68.52% 降到 32.42%/35.43%。这些现象与更大存活状态、访存和局部性问题相容，但不是机制的独立计时。

long scoreboard 等 stall 指标不能当作互斥耗时比例；驻留 warp 改变会改变其归一化值。51 的 `_9` all-stage occupancy 含 PS 成分，不能冒充纯 CS occupancy。98 在三个 capture 与当前生产中均为 fragment stage，约 2.835M 周期稳定；没有“98 改成 compute”这一解释。计数器不提供可直接恢复各 pass 毫秒数的完整时钟信息。

### RGA 的独立静态证据

`rga/comparison.md` 对齐了原始归档和当前源码的 gfx1100 编译选项与 half 支持。以下仅列 54：

| 源码 | VGPR | SGPR | LDS 字节 | scratch 字节 | ISA 字节 | 静态指令数 |
|---|---:|---:|---:|---:|---:|---:|
| fce9cd0 | 46 | 50 | 2048 | 0 | 4064 | 753 |
| 0219372 | 58 | 58 | 2048 | 0 | 5084 | 977 |
| 933ec15 | 71 | 74 | 2048 | 0 | 5080 | 984 |
| 当前 baseline | 69 | 78 | 2048 | 0 | 5324 | 1016 |
| 当前 oct_large | 63 | 66 | 2048 | 0 | 5280 | 1003 |

AMD 静态资源从 46→58→71 VGPR 的增长与新增流/更大状态的源码变化相容；当前 oct 将 69 降到 63，静态 RCP 从 21 降到 17，FP32 FMA 从 99 降到 87。所有这些编译产物 SCRATCH_MEM=0，scratch load/store 也为零，没有在该 AMD 离线产物观察到 spill。

这不能证明 NVIDIA 的实际物理寄存器数、spill 与 occupancy 阈值，更不能据 69→63 宣称 RTX 的净提速。当前 51 的离线 ISA 还有自动展开等编译策略差异，静态指令数与源码 tap 数都不能直接代表动态耗时。实际性能结论仍来自前述 NVIDIA 计时和控制。

### 方差准备 50/65 仍需单独定位

空间实验没有替代 50/65 的历史归因。源码已经确认 `e8acfd2` 取消了成熟历史跳过 49-tap 方差准备的 gate，使此前可跳过的大邻域工作更常执行；后续还改变了池化来源、Kish/Bessel 与 estimator/unknown 的统计准备。这是独立于空间双流/ray 访存的明确工作增加，**本轮尚未用该相邻提交对量化其历史耗时**。

净链已经把当前 50/65 成本包含进候选收益判断，但不能从八阶段总差倒推出 mature gate、sigma 或某个准备 helper 的独立成本。后续若优化准备阶段，应针对确切源码门控与数学需求作单变量验证。当前报告不声称恢复旧性能，也不把诊断候选认定为已完成的生产修复。

## 10. 本轮文件与验证

生产目录的 223 个文件与本轮开始的冻结树完全一致，见 [快照核对](../temp/bench_culprit/production_validation.json)。本轮更新了报告及两个基准工具：`audit_post_pipeline.py` 增加显式的逐字段捕获出口，不将该出口当作测试通过；`benchmark_spatial_chain.py` 修正了诊断版仅保留 scratch 写入时的资源识别。默认完整降噪链仍通过 17×9 的 37 项逐位检查，见 [默认验证](../temp/bench_culprit/tool_default_validation.json)。候选的精度和计时验证分别按上文标注，没有将修改统计或采样语义的诊断写入生产。

从项目根目录复现主要实验：

```powershell
python -X utf8 -B temp/bench_culprit/historical_balanced.py --pair fce9cd0:0219372 --pair 0219372:933ec15 --fixture coherent --output temp/bench_culprit/repeat_history.json
python -X utf8 -B temp/bench_culprit/historical_balanced.py --pair 933_pdf_alias:933_oct_rays --fixture coherent --output temp/bench_culprit/repeat_rays.json
python -X utf8 -B temp/bench_culprit/net_chain.py --candidate temp/bench_culprit/baseline_shaders --require-bitwise --output temp/bench_culprit/repeat_net_aa.json
python -X utf8 -B temp/bench_culprit/net_chain.py --candidate temp/bench_culprit/oct_large/shaders --output temp/bench_culprit/repeat_net_oct.json
```

这些命令应顺序执行，计时期间不同时运行另一份 GPU 实验或离线编译。历史适配合同见 [ABI 核对](../temp/bench_culprit/ABI_AND_PAIRS.md)，净链输入恢复及重复验证见 [恢复集设计](../temp/bench_culprit/NET_CHAIN_DESIGN.md)。
