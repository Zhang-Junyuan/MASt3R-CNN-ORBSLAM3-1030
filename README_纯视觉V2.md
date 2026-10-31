# 纯视觉 MASt3R 动态 SLAM V2

项目：`E:\MASt3R_Dynamic_ORB_SLAM3`。仅使用 RGB 图像和相机标定。已接通稠密预测、卷积动态过滤、稀疏观测、局部地图跟踪、关键帧管理、局部 BA、历史重定位、回环验证、全局 BA，以及稀疏/稠密地图导出。

## 启动

双击 `运行纯视觉全链路.cmd`，运行 120 帧主序列，再运行几何测试、真实特征关联丢失测试和受控回访测试，最后生成报告。双击 `查看纯视觉结果.cmd` 查看结果。

```powershell
cd E:\MASt3R_Dynamic_ORB_SLAM3
.\run_visual.ps1 -Verify
.\run_visual.ps1 -Mode both -Frames 120
.\run_visual.ps1 -Build -Mode dynamic -Frames 120
```

`-Verify` 要求 120 帧、Stride=3，并包含 dynamic 模式，以使受控测试有完整输入。默认运行不注入跟踪故障；故障注入和回访仅在明确命名的独立测试中执行。已有的 `run.ps1` 和动态建图入口也改为生成 V2 输出。

环境仍为 WSL Ubuntu-22.04 / RTX 4060 Laptop / 共享 Python 依赖环境 `/home/yuan/mast3r-slam-local/env/bin/python`。MASt3R、DUSt3R、CroCo 源码与 RoPE CUDA 扩展均从本项目加载；原有 SLAM 工程未修改。来源见 `SOURCES.json`，修改补丁见 `pure_visual_v2_integration.patch`。

## 实现结构

1. `MASt3RFrontend` 预测参考坐标系下的双视图点图、24 维描述子和置信度。初始 PnP 仅用于运动补偿。
2. 10 通道有序点图进入 3,817 参数 CNN；过滤高风险点，使用二维网格和三维体素生成稀疏观测。
3. 原生 `MASt3RSystem` 合并输入关联与最近 5 个关键帧的浮点投影匹配，估计 PnP 位姿并调用原生 `PoseOptimization`。
4. 关键帧写入原生 Atlas / Map / KeyFrame / MapPoint；更新观测与共视关系，撤销强动态证据点，周期性运行原生 `LocalBundleAdjustment`。
5. 历史检索使用归一化描述子均值签名和 L2 排序，候选关键帧再做局部 L2 / ratio / 双向匹配及 PnP 验证。它是紧凑工程检索器，不是经过独立训练的 VPR 网络。
6. 跟踪失败时，历史关联通过几何与位姿优化后恢复定位。正常跟踪时，周期性检索时间间隔足够远的关键帧；验证后融合几何一致重复点，建立共享观测与回环边，再调用原生全局 BA。
7. 保存关键帧局部相机点云，使用最终关键帧位姿重新融合，导出 4 cm 体素点云及至少两个关键帧支持的点云。轨迹也按最终参考关键帧位姿更新。

控制器串行调度上述模块，地图修改有唯一入口。没有启动原版 Tracking / LocalMapping / LoopClosing 三个线程，也没有使用 ORB 二进制词袋；保留的是原生地图数据结构与几何优化能力。

为支持 MapPoint::Replace 的浮点描述子代表选择，`ORBmatcher::DescriptorDistance` 新增浮点 L2 分支；学习特征跟踪使用自己的匹配阈值，不沿用原版 Hamming 阈值。原二进制分支仍保留。`configure_native.py` 可在重新下载源码后重新应用该修改。

## 几何验证条件

当前实现包括最少对应数、PnP 内点数与内点比例、图像空间覆盖以及正深度检查。正常跟踪时回环候选还要求与当前估计的相机中心差小于 0.75 m；近期关键帧被排除，回环设冷却间隔。它主要针对有界漂移下的重访，尚未实现大尺度 Sim3 校正和多地图合并。

动态候选过滤阈值 0.65。旧地图点撤销使用更高概率门槛 0.85，并同时要求运动补偿后的重投影、三维残差及匹配支持。首帧及无支持区域属于未知，不保证静态。

## 实际验证与边界

真实 RGB 主序列、真实特征故障重放、真实特征受控回访、合成已知几何测试分别记录，见 `纯视觉V2_验证报告.md` 和 `results/visual_v2/verification.json`。受控回访通过重放开头视图触发，不是自然闭环数据集性能证明；重放过程不重新运行 MASt3R。

动态 CNN 仍是合成初始模型，真实动态检测精度未知。运行成功、剔除数量、回环触发次数均不能单独证明定位精度提升。单目重投影 BA 仍有尺度漂移风险。未实现任意尺度回环、多地图合并、完整历史稠密动态清理，也没有实时性保证。当前展示的是已经执行并验证的纯视觉研究链路。

## 输出与框架图

- 主序列：`results/visual_v2/dynamic/frames.csv`、`trajectory.txt`、`sparse_map.ply`、`dense_filtered_candidate.ply`、`dense_static_multiview.ply`。
- 主序列回环事件：`results/visual_v2/dynamic/loop_events.csv`，没有事件时只有表头。
- 故障重定位：`results/visual_v2/relocalization_replay.json`。
- 受控回环与融合：`results/visual_v2/loop_replay.json`、`loop_replay/loop_events.csv`。
- 详细整体图：`figures/04_visual_v2_complete.png` / `.svg`。
- 纯视觉原始与新框架对照：`figures/05_visual_v2_comparison.png` / `.svg`。
- CNN 结构：`figures/02_cnn_detail.png` / `.svg`。

源码入口为 `run_pipeline.py`、`mast3r_frontend.py`、`dynamic_gate.py`、`native/mast3r_backend.cc`。V1 历史记录单独保留在 `README_V1_原型记录.md` 和原 results/baseline、results/dynamic 中。
