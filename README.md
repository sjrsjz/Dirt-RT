# Dirt RT

A simple path tracing shader for vulkanite mod,which uses Nvidia GPU's RT cores to render

**WARNING: You need a LabPBR resourcepack to use this shader pack properly. IF you reload your resourcepack, you need to RESTART the game to make the shader work properly.**

**This pack needs this version of [Vulkanite](https://github.com/sjrsjz/vulkanite-modified/releases/tag/26.2-v0.0.6)**, you may build the latest version by yourself.

**Technical documentation:** [ZH](doc/tech.pdf) | [EN](doc/tech_en.pdf)

**Code guide:** [模块与管线](doc/shader_architecture.md) | [优化与验证记录](doc/transport_optimization.md)

**1080p profile:** [针对性优化与验证](doc/bench1080_optimization.md) | [游戏反馈后的联合诊断（捕获 11）](doc/bench11_joint_optimization.md)

**Historical baseline:** [捕获 9、提交历史与性能回归](doc/bench9_regression_analysis.md) | [同输入提交消融与 oct32 净收益](doc/bench_culprit_analysis.md)

**Sigma / ray analysis:** [未知统计、预计算射线与独立派发实测](doc/bench_sigma_ray_analysis.md)

# Screenshots

![1784211218150](image/README/1784211218150.png)

![1784211225727](image/README/1784211225727.png)

![1784211234505](image/README/1784211234505.png)

降噪、缓存、光栅和后处理的改动及验证：[所有 pass 优化记录](doc/all_pass_optimization.md)。
