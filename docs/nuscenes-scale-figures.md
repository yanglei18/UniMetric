# nuScenes 尺度恢复可视化

本地 `data/` 为空。2026-09-16 已通过配置中的 SSH 主机 `369u237m02.qicp.vip` 核实远程项目 `/home/yanglei/ws_yuner/Metric-Depth`：`outputs` 指向 `/data1/yanglei/outputs`，原始数据在 `data/official-data/nuScenes` 下可访问，已确认 scene-0003 的 JPG、PCD、YAML 文件。

远程 `/data1/yanglei/outputs/nuScenes` 包含 116 个 scene 目录，但其中 929 个模型输出目录全部为空。递归统计未发现 NPY、NPZ、PLY、PCD 预测数组；保留了 SAM 掩码、VLM 结果、九个模型评测报告和消融实验文件。报告引用的另一工作区 `/home/yanglei/ws_yiyang/Metric-Depth-new/outputs` 也指向相同输出根。

已下载九个模型报告至 `output/figures/nuscenes_scale/evidence/all_summaries.json`，保留来源路径。`scripts/plot_nuscenes_available_metrics.py` 从报告中生成真实三任务指标图及逐场景 CSV，对每个模型的恢复前后共同有效场景等权平均。八个模型各有 116 个场景，Scal3R 仅一个。指标图不等于预测深度、点云、轨迹的可视化；原始预测需要找回或重跑。`nuscenes_layout_preview` 仍仅为排版。

## 出图

```powershell
python scripts/plot_nuscenes_scale_triptych.py
python scripts/plot_nuscenes_scale_triptych.py --bundle path/to/scene.npz --title "nuScenes scene-XXXX | model"
```

输出为 PNG 与 PDF。无 bundle 时只生成明确标注的空排版；真实 bundle 输出 depth、reconstruction、ego_pose 三张图。

## NPZ 输入契约

全部数组为数值类型，不允许 pickle。`gt / before / after` 分别代表真实参考、恢复前、恢复后。

| Key | Shape | 含义 |
|---|---|---|
| rgb | H,W,3 | 同一帧 RGB，uint8 或 [0,1] float |
| depth_gt / depth_before / depth_after | H,W | camera-z 米制深度；无效值 NaN 或 <=0；GT 保持 LiDAR 稀疏性 |
| object_gt / object_before / object_after | N,3 | 同一物体 ROI 点云，相机坐标 x 右、y 下、z 前 |
| endpoints_gt / endpoints_before / endpoints_after | 2,3 | 同一物理端点对应的三维位置，单位米 |
| cloud_gt / cloud_before / cloud_after | N,3 | 融合点云，同一坐标系，z 向上，单位米 |
| trajectory_gt / trajectory_before / trajectory_after | T,3 | 时间同步的 ego 平移，同一坐标系，z 向上，单位米 |

点云不同分支可以有不同点数。轨迹必须等长，默认每 10 帧计算一个位移尺度比，GT 段位移 <=0.5m 的段不计算比值。ATE 是输入共同坐标系内的平移 RMSE，脚本不执行对齐。

## 数据准备与公平性

- 使用同一 scene、CAM_FRONT 和时间戳；不能分别挑恢复前后最有利的帧。
- 将 LiDAR 经过其采样时刻 ego pose 和相机曝光时刻 ego pose 变换到相机坐标，投影时保留 z-buffer 最近点；深度是 camera-z，不是 LiDAR range。
- 单帧深度与物体点云参考第一张示例：统一深度色条，标注同一组端点的 3D 欧氏距离。nuScenes 标注框尺寸不能直接当作任意可见点对的距离；若使用框长宽高，应明确标注为 annotation box extent。
- 当前工具局部点云为统一范围的 x-z 投影。最终可改为同一虚拟视角的 RGB 点云渲染，尺寸仍从三维端点计算。
- 融合重建必须明确使用预测 pose 还是 GT pose；GT pose 仅适合隔离 depth 尺度效果。对融合的动态物体进行过滤或运动补偿。
- 重建图与轨迹图必须保留尺度；可在输入准备时做一次全序列刚体 SE(3) 对齐，记录变换，不能做 Sim(3)、逐帧或逐段缩放对齐。
- ego pose 与 camera pose 应先按标定外参转换，不能混用。点云和轨迹共用相应坐标变换。
- 若原模型只输出任意尺度，应明确声明原尺度的单位约定；不能把任意单位伪称为米或用 GT 拟合尺度后称为恢复前结果。
- 保存 scene、frame、模型、scale variant、单位约定、对齐变换和点云过滤策略作为结果 provenance。仅有汇总指标无法生成这些真实可视化。

接入尚待提供的数据路径后，需按当前仓库 pointmap/*.npy、camera_params/*.json 以及帧 YAML/PCD 契约制作 bundle；本脚本不是已经完成的原始 nuScenes 数据适配器。
