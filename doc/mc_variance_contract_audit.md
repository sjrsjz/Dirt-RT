# MC 方差全通路审计（2026-09-06）

下文记录修复前的审计快照。后续已按用户要求实施
[异常方差入口隔离](uncertainty_ingress_guard.md)；旧数值反例仍保留用于回归对照。

本次为源码检查和 CPU 反例复算。没有修改生产 shader、切换游戏设置、增加 SSBO 或提交。
用户已经把 `variance_prepare.glsl` 的零均值／非零 RMS 返回值从 `65504²` 改为 `1.0`；审计保留此修改。
游戏对照已经表明原分支是爆红的直接来源，条带上每个像素的原始触发原因尚未捕获。

## 复算

运行 `python -B tools/audit_mc_variance_contract.py`。输出 JSON 包含源码 SHA-256、Python/NumPy 版本、种子和所有反例。
本机 Python 3.10.9、NumPy 1.26.4，种子 906202619。测试均使用 CPU IEEE FP16 舍入，不假定显卡开启 FTZ。
此脚本建立反例与代数不变量，不替代实际 GLSL 执行、帧抓取或独立 GPU 统计。

## 逐通路核验

| 环节 | 漫反射 | 反射 | 结论 |
| --- | --- | --- | --- |
| RT 输出 | NEE 与间接光的方向矩相加；RMS 为合并亮度绝对值 | 两项 qLi 矩相加；Raw adapter 用合并亮度绝对值作为 RMS | 对每帧合并样本的标量二阶矩定义一致；每帧不是必然单一方向原子 |
| Raw／历史打包 | `.w=(N_eff,RMS)` | 历史／proposal `.w=(RMS,N_eff)`；Raw `.w` 为距离元数据 | 两套 ABI 的对应消费者顺序匹配；反射 Raw 距离未被误当 RMS |
| 重投影 | 同一双线性权重累积一阶矩和 RMS² | 同一双线性权重累积一阶矩和 RMS²，再合并 surface/virtual 历史 | 有限合法输入下线性矩算术一致；反射中间多次 FP16 一阶矩打包可增加精度损失 |
| proposal／最终 Raw commit | 一阶矩与 RMS² 使用同一 alpha；Kish 由同一 alpha 更新 | 同左 | 保持代数混合关系；最终 alpha 依赖当前光照，不等于固定独立采样权重 |
| 共享方差 tile | 写入 `(N,RMS)`，读取 `.yx=(RMS,N)` | adapter 转换至相同共享接口 | 未发现字段互换；16×16 内区和 halo 索引覆盖一次，通过 CPU 检查 |
| Prepared | 空域池化矩后计算 V；与中心时域 V 按 N 混合 | 同左 | 两者调用同一闭包；闭包与有限历史修正的统计适用性另见下文 |
| independent-current | 当前帧 Raw 矩，N=1；方差来自邻域时域 proposal 的空间池化 | 同左 | 方差不是当前帧样本的独立经验方差；要求局部时空统计可转移 |
| A-Trous | V 线性平均；独立当前分支另行传播 Kish；proposal N 不更新 | 同左 | 存在异方差乘积近似和固定 rejection-confidence 缺口 |
| filtered-history | 平均历史／当前 V，再由 Kish 表示样本数 | 同左 | 相同异方差问题；权重选择和跨帧相关性额外影响 MC 校准 |
| Debug | Prepared 为 composite50；Filtered 为末级 A-Trous proposal | Prepared 为 composite65；Filtered 为末级 A-Trous proposal | Filtered 不是最终 resolve 的 filtered-history，也不是 independent-current 方差 |
| 折射 | PSR 端点／透射数据，由后续合成使用 | 无独立 MC 方差降噪链 | 不应将折射距离或透射字段当作本次 RMS |

源码中的资源顺序为 diffuse 1→50→51..56→58，reflection 59→61→63→65→66..71→72。
漫反射历史读取结束后 resolve 才写回；反射历史读取结束后 history_stage 才复用历史平面。
独立当前 scratch A/B 由两域串行复用。源码未发现同 pass 邻域读写同一 plane；这不证明运行时驱动屏障正确。
8×8 地址映射在 1269×744、1273×793、13×10 下无重复且在声明的 padded 层范围内；未验证运行时实际 SSBO 分配。

## 确认的问题

### 1. 零均值保护将精度失配变为约 43 亿方差

`variance_prepare.glsl:80` 的旧零均值分支在 RMS>0 时返回 `65504²`。
采用 alpha=.01、N=199、Raw R≈1.01328e-6：

- FP16 前 mean≈1.01328e-8、RMS≈1.01328e-7；FP16 后 mean=0、RMS≈1.19209e-7。
- 同一闭包在未量化状态给出 2.52053e-7；旧分支加有限样本修正给出 4.31244e9。
- 混入 0.1% 的亮度 1e-4 邻居矩后，量化均值非零，闭包返回 2.10044e-5。
- 在生成方差之后混合，则旧大值即使只获得 1e-5 权重，也贡献 43124.45。

这建立“矩先混合可消除异常、方差后混合可扩散异常”的具体机制。
返回 1 或 0 都是替代策略；0 对全黑状态正确，对残余二阶矩可能低估不确定性。
最终应统一失效处理和量化策略，不能把任意巨大常数作为 MC 数据传播。

### 2. 状态只清理了一半，并且可失去失效标记

- `diffuse_buffer.glsl:sanitizeDiffuseMaxEntEncoding` 和 `specular_buffer.glsl:sanitizeSpecularMaxEnt`：任一一阶矩或 CoCg 非有限，即清空全部一阶／色度状态。
- temporal packer 和 specular history writer 随后独立保留 RMS 与有效 N。注入 CoCg=NaN、mean=100、RMS=110、N=32，可得到 mean=0、RMS=110、N=32。
- `variance_prepare.glsl:denoiserVarianceFiniteMean` 同样将异常一阶矩清零，保留 RMS。中心 source 的统计有效性没有作为统一的 record-valid 状态传递。
- `pack_half.glsl:sanitizeRootMeanSquareFP16` 把正 Inf RMS 变为 65504；`variance_prepare.glsl:denoiserVarianceSanitizeNonnegative` 把 Inf 变为 0。相同异常经过不同入口后，可能成为极高不确定性或零不确定性。
- `signal.glsl:denoiserSanitizeMaxEntSignal` 把 NaN/Inf sigma 变为 0，而 signal-valid 只检查 sigma 有限且非负：异常记录可以重新表现为有效、零方差状态。
- 反射 `maxentGatherHistory` 在相加前检查元数据，但不检查一阶矩和色度；一个异常 tap 可污染混合，再被 packer 清空一阶矩，二阶矩仍被保留。漫反射检查 Raw tap 的一阶／色度，但 filtered tap 主要依赖 sigma-valid 检查。

上述为源码可达的故障路径和故障注入反例，不声称实际游戏已经产生 NaN。
合法有限矩的非负线性混合不会凭空得到 mean=0、E[R²]>0：对 R≥0，E[R]=0 蕴含 R=0 几乎处处。
因此必须区分真实零值、FP16 失配和异常记录，不能都打包成“有效黑色”。

### 3. 角向 MC 闭包不由当前存储矩唯一确定

RT 每帧合并 NEE 与间接方向原子。合并包记为 `(v,R)`，v 不必满足 |v|=R。
每帧完全相同的两个方向原子、每项能量 R/2、方向在法线两侧 ±60°，给出：

`v=(0,0,R/2), mean=R, E[R²]=R²`。

令 R=100、N=32，真实 MC 方差为 0，当前闭包给出 **17.8660**。
另一个过程每帧随机选择上述两个方向之一、能量始终为 R，具有相同存储的总体矩，真实局部 Bures 观测方差为 **18.75**。
因此这些存储矩无法唯一确定一般合并包的 MC 协方差。g^-3/g^-4 只能提供明确假设下的闭包。
此前公式审计验证了闭包模型内部的代数，不证明模型适合真实 PT 包。

闭包应用的 `N/(N-1)` 对完整非线性表达式不是普适无偏校正。
对上述常量包，N=2 时闭包给出 34.6154；相同包的 7×7 sigma=1 池化使 Kish N≈25.1004，闭包变成18.0258。
两者实际 MC 方差均为0，差异不需要几何差异或浮点误差。这不是此前“时域总是更低”的通用解释。

### 4. 线性方差字段与 Kish 的乘积不等于一般估计量方差

固定归一化权重 a_i、独立误差和共同局部度量下：

`Var(sum a_i X_i) = sum a_i² V_i/N_i`。

代码用线性平均的 `sum a_i V_i` 除以 Kish N。即使 N_i=1，得到的也是
`(sum a_i V_i)(sum a_i²)`，两者一般不同。
权重 (.9,.1)、V=(1,1000) 的 CPU 反例：当前乘积为82.738，真实独立估计方差为10.81，比值7.65384。
这是纯异方差反例，尚未加入自适应拒绝、核重叠和 Bures 度量随位置变化。
因此 Kish 的权重代数本身成立，不代表将任意线性 V 字段与它组合就得到正确 MC 估计方差。
此前同质平面相关系数校准没有覆盖这些条件。

### 5. 拒绝倍率与诊断的范围

proposal 的 sigma 是线性平均观测方差，拒绝分母使用原时域 N（并 cap16）；按级固定 C 从1增加到52.4633。
这模拟充分接受核的收缩，未随实际接受权重更新 proposal 不确定性；previous-pass 拒绝强时可继续过度收紧。
independent-current 的 Kish 另行更新，不用于这个分母。

Prepared/Filtered 都显示 proposal 的方差；最终 resolve 的独立当前输入始终使用空间池化方差。
所以增加“空域冷启动时长”不是修复全链路方差的保证，诊断图也不能直接代表最终 filtered-history 使用的 V。

## 排除范围与后续顺序

10000 组有限非负加权矩通过 Jensen／矩锥检查；相同 FP16 舍入后 RMS>=mean。
10000 对半精度元数据通过位级往返；未发现 RMS/方差或 N/RMS 的通用字段互换。
这些结论仅覆盖所列有限输入和源码接口，不排除 GPU FTZ、实际资源绑定、运行时分配／屏障或上游非有限数据。

优先处理记录级失效一致性与零均值异常分支，再标记异常首次出现位置；随后区分模型闭包误差和估计量方差传播误差。
不能通过换常数、加冷启动时长或重新拟合一个相关系数，宣称已经修复上述所有问题。
本次未改变统计定义、生产 shader、存储布局或用户设置。
