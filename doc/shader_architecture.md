# 着色器模块与数据契约

入口文件负责声明阶段和编译选项，算法放在模块中。修改时先确认信号的测度、坐标系和写入阶段，再修改计算；文件名中的 composite 编号是宿主调度接口。

## 从哪里开始读

| 层 | 文件或目录 | 职责 |
| --- | --- | --- |
| 调度和设置 | `shaders.properties`、`lib/settings.glsl` | RT 分组、资源分配、纹理绑定、用户选项 |
| RT 入口 | `ray0..5.rgen`、`lib/rt/raytrace_rgen.glsl` | 主可见性、续追、折射端点和缓存更新 |
| 场景接口 | `lib/rt/raytrace/scene.glsl` | 发射射线、解码材质、实体运动和材质签名 |
| 瞬时记录 | `lib/rt/raytrace/types.glsl` | 第一跳、引导、介质和折射链的寄存器数据 |
| 表面散射 | `lib/rt/raytrace/bsdf.glsl`、`lobe_selection.glsl` | BSDF/PDF/解析权重与离散分支概率 |
| 路径策略 | `lib/rt/raytrace/bounces.glsl`、`path_trace.glsl` | 第一跳采样、后续混合采样、NEE、缓存终止和轮盘赌 |
| 引导和折射 | `lib/rt/raytrace/guiding.glsl`、`refraction.glsl` | 历史重投影、混合 PDF、虚拟折射端点 |
| 介质与光源 | `lib/rt/raytrace/medium.glsl`、`lighting.glsl` | 段吸收/发射、太阳可见性、缓存查询 |
| 基础数学 | `lib/math/random.glsl`、`sampling.glsl`、`noise.glsl` | 随机序列、正交基和方向采样、程序噪声 |
| 微表面 | `lib/pbr/ggx.glsl`、`fresnel.glsl` | GGX、Smith 遮蔽和 Fresnel |
| 光照表示 | `lib/lighting/maxent.glsl`、`maxent_encode.glsl` | 方向矩、RGB 编码、余弦投影和引导分布 |
| 反射解码 | `lib/lighting/specular_maxent.glsl`、`specular_cdf/` | 材质调制、响应拟合与独立系数数据 |
| 缓冲与降噪 | `lib/buffers/`、`lib/lighting/denoiser/`、`post/denoiser/` | 存储 ABI、统计传播、阶段适配器 |

`lib/common.glsl` 和 `lib/rt/raytrace/transport.glsl` 保留为聚合入口。后者依赖先包含 `scene.glsl`；各 RT 阶段的资源声明由 raygen 聚合文件提供。纯数学模块不应引入屏幕缓冲或 RT 描述符。

## 一帧的数据流

| 阶段 | 工作和发布点 |
| --- | --- |
| prepare / gbuffers | 清除覆盖层，写入光栅覆盖颜色与深度 |
| ray0，组 0 | 主射线可见性、材质/几何、重投影反射历史 |
| ray1 / ray2 / ray3，组 1 | 漫反射背景续追、反射续追、几何折射端点；读取 ray0 发布的当前表面 |
| ray4，组 2 | 稀疏缓存分配；发布当前相机状态，续追完成前不能提前发布 |
| ray5，组 3 | 更新驻留体素；读取上一帧过滤后的缓存作为提案与递归光照 |
| composite1 / 3 | 漫反射时域提案、缓存原地时域处理；已移除空的 composite4 |
| composite50–58 | 漫反射方差准备、六级 A-Trous、最终历史提交 |
| composite59–72 | 反射输入、时域提案、历史暂存、方差准备、六级 A-Trous、最终历史提交 |
| composite81 | 解码漫反射/反射方向矩、解析折射端点、组合光照 |
| composite82 / 90–93 / 98 / 99 | 覆盖层合成与暗角、辉光金字塔与模糊、色调映射输出、曝光统计 |

RT 的 `rt.groups = 0 1 1 1 2 3` 表达跨组依赖。当前重构保留该顺序：ray0 的准备状态、ray4 的相机发布和 ray5 的缓存可见性具有不同生命周期。降噪共享 `c3..c6` 和 scratch A/B，因此两个降噪域不能直接并行。详见 [降噪阶段表](../shaders/post/denoiser/README.md)。

主折射由 ray3 的 `TraceRefractionPSR()` 处理。`Trace()` 只服务 ray1/ray2，后续顶点仍可选择反射、漫反射或透射。旧的主射线 fallback 和未使用的第一跳随机折射分支已经移除。

## 数学接口

| 对象 | 契约 |
| --- | --- |
| 方向 | `wo`、`wi` 从表面向外；`rd_i` 指向表面。方向和表面法线均为单位向量 |
| 粗糙度 | 材质 `R.x` 是 GGX alpha；Disney/EON 的感知粗糙度取其平方根 |
| 折射比 | `etaRatio = eta_i / eta_t`，与 GLSL `refract` 相同；辐亮度透射权重包含该比值的平方 |
| 反射求值 | `evaluateSpecularBRDF` 返回 `f * NoL`、VNDF 的方向 PDF；扩展重载同时返回 `f * NoL / q` |
| 透射采样 | `sampleTransmissionWeight` 返回已约分的 `f_t * abs(NoL) / q_t`，调用者再除以离散透射概率 |
| 拒绝样本 | 越出有效几何半球时贡献为零，不能重新归一化剩余方向的 PDF；透射拒绝不能重标成缓存反射 |
| 混合提案 | 选择概率独立于随后采样的微表面法线；PDF 包含所有有权重的方向提案 |
| 漫反射矩 | EON 模式记录 `Li / p` 的方向能量矩，合成阶段施加 BRDF、余弦和材质颜色；旧 Disney 模式保留 `Fd/pi` 调制 |
| 反射矩 | 记录 `(q_vndf / p) * Li` 的方向矩；合成查询使用 `F * G2/G1` 响应。不能按漫反射矩直接解码 |
| 统计 | 滤波的均值保持线性；`sqrt(E[R²])`、估计量标准差和 Kish 有效样本数是不同量 |

方向矩满足 `length(v) <= omega`，在该线性空间累积和插值。EON 与基础余弦投影复用同一标量响应。系数文件只保存运行时所需常量；实现注释描述输入域、输出意义和数值边界，避免加入未公开稿件身份或推导文本。已有许可证仍保留。

## 存储边界

屏幕 SSBO 按 8×8 tile 存储 `uvec4`。分配大小和层号由 `shaders.properties`、`lib/buffers/addr.glsl` 及各缓冲模块共同定义。漫反射缓冲现为 N0–N7 八个平面，每个分配像素 128 字节；独立当前帧的两个临时平面迁出后，比原先十个平面的 160 字节减少 32 字节。保留的持久历史平面及其字段布局不变。

`signal.glsl` 的通用信号 ABI 仍为 16 字节 `uvec4`：依次保存方向/亮度矩的两个 FP16 对、`CoCg` 对，以及 `(sigma, virtualDistance)` 对。空间链按估计器职责裁掉无消费者的计算，不改变这个通用布局，也不裁剪时域历史字段：

| 空间流 | 传递的有效统计量 | 固定置零字段 | 消费者 |
| --- | --- | --- | --- |
| proposal | `maxEntY`、估计量 sigma、`virtualDistance` | `CoCg` | 邻域权重、虚拟几何与下一级 proposal |
| independent current | `maxEntY`、`CoCg`、估计量 sigma | `virtualDistance` | 下一级 current 与最终时域 resolve |

两域从方差准备 50/65 就写入这些零值，后续空间级继续保持；proposal 累加器不携带色度，current 累加器不携带虚拟距离。末级 56/71 只发布 current。resolve 的公开距离仍由既有几何与历史契约重建，不依赖 current scratch 中已清零的距离字段。这里减少的是算术与活跃数据，并未再次缩小每条信号的存储字节数。

降噪的 `scratch_io.glsl` 复用已有 RGBA32F 辉光图像：A 对应 `bloomAtlas`，B 对应 `bloomBlur`。写入时对每个 uint 加 `0x00800000u` 再作位转换，读回后逆变换。信号字由两个有限 FP16 值组成，这个偏置使传输值保持为有限的规格化 FP32，避免次正规数被清零而丢失原始位。传输期间只允许 `imageStore` 写和 `texelFetch` 读，不能对这些浮点外观的值进行插值、颜色转换或算术。

| 图像使用阶段 | 生命周期与消费者 |
| --- | --- |
| composite50–58 | 50 初始化漫反射 scratch A，六级空间滤波交替使用 A/B，58 消费最终 A 并提交历史 |
| composite65–72 | 65 覆写 A，反射复用同一组图像和空间算法；72 消费最终 A 后，降噪对两张图像的使用结束 |
| composite90 / 91 | 将 `bloomAtlas` 的有效 LOD 矩形覆写为真实 FP32 辉光数据；区外旧值不能参与有效滤波贡献 |
| composite92 / 93 / 98 | 横向模糊写 `bloomBlur`，纵向模糊写 `bloomAtlas`，98 按原边界规则读取辉光；不再读取降噪 scratch |

宿主在相邻 pass 之间必须保证图像写入对后续纹理读取可见。上述复用不新增图像分配或复制，但要求两个降噪域及辉光保持该调度顺序。当前实现与 1080p 验证见 [基于 profile 的优化记录](bench1080_optimization.md)。

RT payload 固定为 16 个 uint。`payload_pack.glsl` 顶部列出完整槽位；0/1 在发射时暂存 ray cone，在 closest-hit 后转为纹理梯度。第 8 槽的低 16 位是 FP16 前段距离，高 16 位现在直接存储整型 flags，避免整数到 FP16 再回整数。所有 RT 入口共用该接口；它不跨帧保存。

`tmp_Payload` 会被每次追踪覆盖。需要跨阴影/折射射线使用的命中信息，应在发射前解码成局部值。分配器、历史、临时过滤平面不能仅因类型相同而互换。

着色器不能使用 `packed` 作为标识符；`audit_shader_pipeline.py` 会在剥离注释后扫描该保留字。

## 验证入口

```powershell
python -B tools/audit_shader_pipeline.py --label current
python -B tools/audit_shader_pipeline.py --label eon_off -D EON_ENABLED=0
python -B tools/audit_transport_math.py --gpu
python -B tools/benchmark_transport.py --baseline-ref 75023be947703564b2867678abbc576049a46ad0
```

编译器默认取 `E:/VulkanSDK/Bin/glslangValidator.exe`。数值审计需要 NumPy，GPU 算术审计和微基准还需要 ModernGL。产物全部写入 `temp/`。编译脚本模拟 Iris 对选项定义的替换，随后验证 SPIR-V；自动绑定位置只用于静态检查，不证明宿主运行时描述符匹配。

测试边界与性能证据见 [本轮优化记录](transport_optimization.md)。

## 非 PT 模块补充

| 模块 | 职责 |
| --- | --- |
| `post/denoiser/reflection/reprojection.glsl` | 反射原始及滤波历史的共同足迹重投影和分支合并 |
| `lib/lighting/denoiser/light_difference.glsl` | 光场距离端点预计算；空间循环复用中心状态 |
| `lib/lighting/denoiser/signal.glsl` | 通用 16 字节信号 ABI、完整检查与只供封闭空间链使用的有效性快捷判据 |
| `lib/lighting/denoiser/atrous_filter.glsl` | 统一 tap 权重与 sigma 传播；按 proposal/current 职责保留不同的累加字段 |
| `lib/lighting/denoiser/atrous_tap.glsl` | 两种采样排布共用的邻居校验、权重与仍有消费者的估计量更新 |
| `lib/lighting/denoiser/atrous_policy.glsl` | 共用的输出生命周期：末级保留提案输入和权重，移除无消费者的提案累计与图像写入 |
| `lib/lighting/denoiser/scratch_io.glsl` | 两张辉光图像中的独立当前帧 A/B 存取，以及防止次正规数丢失的可逆位传输 |
| `lib/lighting/denoiser/variance_prepare.glsl` | 输入清理、估计量方差准备与角色字段初始化；从空间链入口清零无消费者字段 |
| `lib/lighting/denoiser/variance_tile.glsl` | 共用方差瓦片装载、一次方向解码与空间矩重建 |
| `lib/buffers/radiance_cache/{storage,allocator,sampling}.glsl` | 缓存 ABI/查找、单写者分配、带校验的采样 |
| `post/resolve_refraction.glsl` | PSR 屏幕、缓存和环境端点解码 |
| `lib/debug/lighting_views.glsl` | 合成阶段诊断显示 |
| `lib/buffers/debug_buffer.glsl` | 仅产生当前 `DEBUG_VIEW` 使用的诊断字段；其他字段不保证初始化 |
| `lib/post_processing/gaussian_blur.glsl` | 横纵向共享的对称高斯核；适配器指定轴和图像 |
| `lib/post_processing/exposure_response.glsl` | 曝光标定、瞳孔模型、适应响应 |
| `lib/post_processing/tonemap_coefficients.glsl` | 显示映射固定系数，独立于求值代码 |

两类信号共用空间执行规则：小步长为 16×16，大步长为 8×8，均保留虚拟平面重建。粗糙度通过数据参与同一个算法。第 32 步仍用提案确定有效性和权重，但不写提案图像；只有过滤后方差调试视图 25/38 保留末级提案累计。resolve58/72 从 scratch A 读取最终独立当前帧估计量及其有效性。这个规则由共同的消费者关系决定，不按信号域特化。

小核将每轴偏移加一后按每 tap 两位存入整数常量，替代动态索引的偏移/权重数组。八个邻居仍按 `(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)` 累加，斜向/轴向权重仍为原来的 `0.44445` / `0.66667`，不改为会改变舍入的分数。

空间链的快捷有效性检查只允许用于“方差准备 → 空间交替读写”的封闭链：输入已清理，每级输出再次清理，sigma 只能是有限非负数、-1（无效）或 -2（有效但方差未知）。因此 `denoiserSpatialPreparedSignalWordsValid` 可直接排除 FP16 的 -1 编码，再使用可信解码。两个估计器仍共同参与有效性判断，几何检查仍保留。此判据不能识别任意损坏编码；history、reprojection、resolve 等边界继续使用完整检查。新增空间输入来源必须先建立同样的清理契约，不能直接复用快捷判据。

方差准备将重复方向解码移到共享瓦片装载阶段。自动曝光采用一个 128 线程工作组，仅线程 0 发布 FrameData。原始矩、估计量 sigma、持久历史字段及唯一历史提交点保持不变；调试缓冲只有当前视图消费的字段属于有效输出。

空间小核在步长 1/2/4 分别共享 18×18、20×20、24×24 的 FP32 主射线瓦片，复用投影、归一化和世界旋转结果。所有线程完成协作装载并经过屏障后才能退出。`geometry.glsl` 的统一射线接口由小核提供共享读取实现，大核保留直接重建；不改变两域的采样、统计或持久布局。历史性能变化及验证见[捕获 9 回归分析](bench9_regression_analysis.md)。

最新游戏反馈、联合 profile 分析和验证记录见 [捕获 11 联合复核](bench11_joint_optimization.md)；上一轮独立微基准与存储改动保留在 [1080p 优化记录](bench1080_optimization.md)，此前覆盖表保留在 [所有 pass 的整理与验证](all_pass_optimization.md)。
