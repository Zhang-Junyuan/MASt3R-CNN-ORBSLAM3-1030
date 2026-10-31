# MASt3R 动态点过滤与 ORB-SLAM3 后端研究原型

独立工程：`E:\MASt3R_Dynamic_ORB_SLAM3`。本项目使用独立下载并校验的官方 MASt3R、DUSt3R、CroCo、ORB-SLAM3 源码。没有复制或修改用户已有的 ORB-SLAM3 工程。

这是实际调用 MASt3R 和原生 ORB-SLAM3 优化器的可运行研究原型。**不是已经证明有效的动态 SLAM，也不是具备原版全部功能的替代发行版。** 当前卷积动态点模型只经过合成数据初始训练，真实动态点准确率未知。实际运行结果见 `验证报告.md` 和 `results/report.html`。

## 一键运行

- 双击 `运行动态建图.cmd`：运行 60 帧 RGB 演示，输出过滤后的稠密与稀疏地图、轨迹。
- 双击 `运行对照验证.cmd`：依次运行同一 MASt3R 系统的“不启用过滤”和“启用过滤”两组。
- 双击 `查看结果.cmd`：打开本地报告，拖动查看实际三维点云。

PowerShell：

```powershell
cd E:\MASt3R_Dynamic_ORB_SLAM3
.\run.ps1 -Mode both -Frames 120 -Stride 3
.\run.ps1 -Build -Train -Mode dynamic
```

运行环境为 WSL Ubuntu-22.04，使用已安装的共享 Python 依赖环境 `/home/yuan/mast3r-slam-local/env/bin/python`。这只是解释器与依赖，不会导入旧工程的 MASt3R/SLAM 源码。MASt3R、DUSt3R、CroCo 源码路径明确指向本项目；RoPE CUDA 扩展也在本项目重新构建。更换电脑时需要重建环境，启动脚本中的解释器路径须对应修改。

## 实际实现的框架

```text
前一有效参考帧 RGB + 当前 RGB + 相机内参
    ↓ MASt3R（官方预训练权重）
两视图有序三维点图 + 几何置信度 + 24 维归一化描述子
    ↓ 网格候选点、浮点 L2 双向匹配
鲁棒 PnP 初始相机运动 → 重投影残差 / 对应三维点不一致性
    ↓ 10 通道有序点图特征
轻量深度可分离 CNN → 每个像素/三维点的动态风险概率
    ↓ 剔除高风险点、置信度与深度检查
二维网格分布 + 三维体素稀疏化 → 关键点、描述子、3D、地图 ID
    ↓ 原生 C++ MASt3RSystem 适配前端
地图点 PnP-RANSAC → ORB-SLAM3 PoseOptimization
    ↓
Atlas / Map / KeyFrame / MapPoint → ORB-SLAM3 LocalBundleAdjustment
    ↓ 优化后的关键帧位姿
重建关键帧稠密点世界坐标 → 体素融合 → 多视角支持点云
```

`architecture.svg` 是横向框架图。

这里的 MASt3RSystem 是新增前端，直接使用原生 ORB-SLAM3 地图与优化模块。它替代了原版 `Tracking + ORBextractor + 基于 ORB 的局部点生成` 路径，没有把 MASt3R 深度伪装成 RGB-D 输入后继续偷偷提取 ORB。原版的 `System::TrackRGBD()` / `TrackMonocular()` 不会自动切换到这条新路径；应使用本项目启动入口。

## 三维坐标和估计顺序

MASt3R 的第一张点图在参考相机坐标系，第二张输出 `pts3d_in_other_view` **也在参考相机坐标系**，不能当作当前相机坐标系。定义 `Tcw` 为世界到相机变换，则第二张点图的世界坐标为：

`X_world = inverse(Tcw_reference) × X_current_in_reference`

前端用已关联地图点的深度中位数比例锚定每对预测的尺度，不使用传感器深度或轨迹真值。初步相机运动只用于构建运动补偿残差；剔除动态点后，最终位姿由稀疏地图点的 PnP 和原生位姿优化估计。这样避免把初始运动补偿和最终位姿估计混为一谈。

原生后端局部 BA 会改变关键帧和地图点。导出稠密地图时用最终关键帧位姿重新变换保存的局部点云；轨迹也按参考关键帧更新。它不是把未修正的逐帧点云简单拼接。

初始化等待两张不同的 RGB 图像，用第一张图像在该图像对中的点图建立初始参考地图，不采用自配对图像伪造双视图。几何置信度按每帧分位数筛选并设最低值 1.01，稠密导出用更严格的中位数门槛。阈值是工程设置，仍需要真实数据上的标定与消融。

## 卷积动态点模型

`dynamic_gate.py` 中的 `DynamicPointCNN` 共 3,817 个参数，结构为 1×1 通道投影、三个深度可分离卷积块（膨胀率 1/2/4）、1×1 概率头。输入 10 个通道：RGB 3、归一化 XYZ 3、几何置信度 1、运动补偿重投影残差 1、两视图对应点三维差异 1、匹配支持度 1。

这是在保留像素邻接关系的 **有序点图** 上做二维卷积；不能直接把无序 PLY 顶点数组送给同一个卷积。输出是运动风险图，从任务定义上属于动态点分类/运动掩码，但不含独立的语义分割网络或目标检测器，也不计算光流场。

当前阈值 `p_dynamic > 0.65`。无时间对应支持的区域标记为未知并保留，第一帧不能判断真实运动。网络训练数据为程序生成的静态/动态残差块及错配噪声：它验证可训练和推理链路，**不能证明网络已经学会识别人、车或真实物体运动**。`checkpoints/training.json` 中近乎完美的合成分数不具有真实场景精度含义。

本机演示输入为 320×240，使用官方 512 预训练权重，并非在本项目重新训练 MASt3R。减小输入分辨率对匹配和三维精度的影响尚未做系统消融；`run_pipeline.py --size 512` 可用于后续分辨率对照，但该配置未计入本次验证结果。

## 稀疏化、匹配与地图

1. 在 8×8 图像网格中，以纹理梯度、点置信度、描述子置信度联合评分选点。不是 FAST/ORB 检测。
2. 使用 MASt3R 的 24 维描述子做 L2 最近邻、双向一致性和 ratio 过滤，不使用 Hamming 或二进制词袋。
3. 对通过动态过滤的候选点做 3.5 cm 三维体素去重；已经跟踪到的地图点保留关联，避免稀疏化破坏连续观测。
4. 原生后端用持久 MapPoint 的世界坐标估计当前相机位姿，进行原版 `Optimizer::PoseOptimization()`；关键帧关联进入真实地图结构，触发原版 `Optimizer::LocalBundleAdjustment()`。
5. 关键帧稠密候选点以 4 cm 体素融合，另外导出需要至少两个不同关键帧支持的点云。两次支持有助于减少瞬态点，但不是静态真值证明。

不对动态图像做黑色填充或纹理补洞；地图与位姿估计不使用虚构的像素观测。跟踪失败帧不插入地图、不写成有效轨迹，保留上一次有效参考帧进行后续匹配。

## 已适配和未适配的范围

|组件|当前状态|
|---|---|
|MASt3R 稠密预测、置信度、浮点描述子|实际加载官方权重运行|
|动态 CNN|已训练并接入；仅合成初始模型，需真实数据训练|
|过滤、网格/体素稀疏化、浮点匹配|已实现|
|ORB-SLAM3 地图结构、位姿优化、局部 BA|实际调用原生 C++ 实现|
|稠密融合、稀疏地图、轨迹与日志|已实现|
|原版 Tracking、LocalMapping、LoopClosing 线程|新路径不启动；由适配前端驱动地图和优化|
|原版 ORB 词袋回环和全局重定位|未适配浮点特征，当前关闭|
|IMU、多地图合并、双目、无标定相机|当前原型未实现/未验证|
|全网端到端反向传播|未实现；MASt3R 冻结、CNN 单独训练、几何优化分开|
|实时性|没有保证；MASt3R 主干远重于轻量 CNN|

因此它是“基于 ORB-SLAM3 后端重建学习前端”的研究实现，不应写成“无损替换 ORB 特征并保留全部原版功能”。当前 BA 使用单目重投影边，MASt3R 三维点作为初始化与配对尺度锚定，不是测距传感器约束；仍可能发生尺度漂移。对于重新匹配到的已建地图点，CNN 概率大于 0.85 且重投影、三维残差共同支持时，原生后端会调用 SetBadFlag 撤销该点及其观测关联。尚不包括完整的跨关键帧重复点融合、未重观测旧点的动态清理和稠密历史点的逐点回溯；首帧未知动态点仍可能残留，不能把导出文件名中的 static 理解成全部点已证明静态。

## 文件导航

- `mast3r_frontend.py`：图像预处理、内参缩放、MASt3R 推理、浮点匹配、几何补偿。
- `dynamic_gate.py` / `train_gate.py`：卷积模型和可复现合成训练。
- `run_pipeline.py`：完整串联流程、过滤、稀疏化、原生通信、稠密融合。
- `native/mast3r_backend.cc`：新增 MASt3RSystem，使用 ORB-SLAM3 Frame/Atlas/Map/KeyFrame/MapPoint/Optimizer。
- `third_party/`、`SOURCES.json`：官方源码与固定提交记录。
- `build.sh` / `build_curope.py`：本项目原生后端和 CUDA 扩展构建。
- `tests.py`：已知运动、错误匹配、卷积过滤和实际局部 BA 的验证。
- `results/*/frames.csv`：每帧跟踪、过滤、关键帧、BA、耗时记录。
- `results/*/trajectory.txt`：有效相机位姿，TUM 时间戳格式。
- `results/*/sparse_map.ply`：原生优化器的稀疏地图点。
- `results/*/dense_filtered_candidate.ply`：过滤后关键帧稠密候选融合。
- `results/*/dense_static_multiview.ply`：至少两个关键帧支持的融合点。
- `results/*/visualizations`：绿点是已关联特征，右图为动态概率；NPZ 保存数值。

YAML 数据包便于检查接口，但不是高性能生产通信。缓存中的 MASt3R 描述子保存为 FP16；为了精确对照，同一配对使用同一缓存表示。报告区分首次推理耗时与缓存，不能把缓存速度当作网络实时性能。

## 数据与来源

新项目内存放 MASt3R 官方预训练权重的本地副本（约 2.6 GB），SHA256 记录在 `checkpoints/SHA256.txt`。RGB 数据从此前独立下载的公开 TUM walking_xyz 数据集只读复制，完整 RGB 列表 859 张；RGB 数量与 RGB-D 关联帧数不同。没有复制 depth，`groundtruth.txt` 只用于事后误差评估。记录见 `data/tum_walking_xyz_rgb/PROVENANCE.json`。

官方源：[MASt3R](https://github.com/naver/mast3r)、[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)。各自源码 LICENSE 和 MASt3R 权重使用条件保留在 third_party 中；MASt3R 上游标注非商业许可，ORB-SLAM3 为 GPLv3。

后续研究优先项是使用有真实动态标签/可靠几何伪标签的数据训练 CNN，补充静态场景和移动相机场景的对照，衡量动态召回、静态误删、跟踪成功率、ATE/RPE、地图残留动态点与端到端耗时。不能仅凭剔除更多点或一次 ATE 下降宣称创新有效。
