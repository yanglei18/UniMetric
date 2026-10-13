# Metric-Depth 文档与记忆继承梳理

日期：2026-05-26

## 范围

本轮梳理覆盖当前仓库 `Metric-Depth/`、父级 `Metric-Depth-Project/` 下与当前代码强相关的 sibling repositories，以及当前代码里的配置、pipeline、模型适配器、评测和工具脚本。

本轮没有修改已有代码逻辑，也没有覆盖当前已有的 README 或配置文件。输出新增为：

- `plan/documentation-inventory.md`：文档盘点、继承上下文、风险与补档建议。
- `plan/metric-depth-codebase-map.md`：当前代码实际功能地图。
- `.codex/project-memory/metric-depth.md`：后续会话可快速继承的本地记忆快照。
- `README.md` 与 `docs/`：已在后续执行中补齐当前 pipeline、配置、semantic scale 和 evaluation 文档。

## 当前文档状态

当前 `Metric-Depth/` 内的 Markdown 文档只有 `README.md`。这份 README 主要记录早期数据格式：

- sequence 目录包含 `*.pcd`、`*.yaml`、`*_camera{0,1,2}.jpg`。
- YAML 中包含相机内外参、LiDAR pose、车辆/行人/骑行者对象信息。
- `dataset_tools/varify_dataset.py` 用于数据验证和可视化。

这份 README 对当前代码的覆盖已经明显不足。代码实际已经发展为一个多 foundation depth / 3D reconstruction 模型对比平台，包含 VLM/SAM 语义尺度锚点、长序列模型适配、metric-scale 标定、多尺度输出变体、LiDAR/pose 评测和跨 run 对比。

父级 `Metric-Depth-Project/` 没有检测到现成 `.codex/project-memory/registry.yaml` 或 Obsidian vault 绑定；因此本轮“继承记忆”采用本地归纳方式：从当前仓库代码、父级 sibling repos 的 README、现有配置树和当前工作区状态中恢复项目理解。

## 已继承的父级项目背景

父级 `Metric-Depth-Project/` 更像一个 foundation model integration workspace。当前 `Metric-Depth/` 是中心评测与标定仓库，旁边的 sibling repos 是外部模型/方法的本地来源或背景参考：

- `Depth-Anything-3/`：DA3 / DA3-Streaming，提供任意视角视觉几何、depth、confidence、pose、point map 能力。
- `VGGT-Long/`：将 VGGT 用 chunk / overlap / alignment / loop closure 推到长 RGB sequence。
- `Pi-Long/`：把 VGGT-Long 的长序列框架迁移到 Pi3/Pi3X，强调更稳定和更省显存。
- `Scal3R/`：大规模 3D reconstruction 的 test-time training / block inference 系统。
- `vggt-omega/`：VGGT-Omega 1B 模型，提供 camera/depth/point/cloud 输出，当前仓库已有 wrapper。
- `lingbot-map/`：streaming 3D reconstruction，支持 KV cache、long sequence、windowed/streaming mode，当前仓库已有 long wrapper。
- `STream3R/`：causal / streaming 3D reconstruction 背景仓库，当前 `Metric-Depth` 暂未看到直接 wrapper。

这些 sibling repos 的共同角色是：提供多种 depth / 3D foundation models 的候选输出，当前仓库负责统一输入输出、补 metric scale、对齐 LiDAR / pose 评测，并沉淀哪个模型和 scale strategy 更可靠。

## 当前代码实际定位

`Metric-Depth` 的核心问题不是“单纯运行一个深度模型”，而是：

> 在自动驾驶/移动机器人长序列场景中，将不同 foundation depth / 3D reconstruction 模型输出统一到可评测的 metric geometry，并用 semantic anchors、ground/plane cues、LiDAR sparse GT 和 pose metrics 比较不同尺度策略。

当前主流程由 `get_metric_scale.py` 调用 `metric_scale.processing.processing.pipeline.run()`。注意：当前 `ProcessingPipeline._process_sequence_task()` 中只有 semantic plane scale stage 处于启用状态，VLM、SAM、P3 image 准备和 depth inference stage 在代码里被注释掉。这意味着当前默认运行更像“基于已有中间产物重新做 scale / metrics”，不是端到端全流程。

## 代码主干地图

- `get_metric_scale.py`
  - `MainRunner.run()`：打印配置、运行 pipeline、随后做可视化和 metric calculation。
  - `MainRunner.process_visualization_and_metrics()`：按 sequence 运行 `ResultVisualizer` 和 `MetricCalculator`。
  - `cleanup_post_metric_outputs()`：清理 depth output 目录中除 `relative_scale` 外的后处理产物。

- `metric_scale/config/config.py`
  - YAML 配置加载、CLI overrides、模板路径解析。
  - 配置被分成 `runtime`、`parallel`、`vision`、`depth`、`semantic_scale`、`models`、`paths`。
  - 支持 sequence difficulty profiles，根据已有 `dataset_summary.json` 调整 semantic-scale 参数。

- `metric_scale/processing/processing.py`
  - Stage 1：VLM semantic anchor mining。
  - Stage 2：复制 VLM 关联图像到 `p3_images/`。
  - Stage 3：SAM segmentation from VLM bboxes。
  - Stage 4：depth / 3D foundation model inference。
  - Stage 5：semantic plane scale estimation，生成 `consistent_scale`、多个 `optimized_scale_*`、`selected_scale`、`absolute_scale`。
  - 支持多 GPU sequence/subsequence scheduling。

- `metric_scale/models/models.py`
  - `ModelManager` 和 `ModelFactory` 管理各类模型 wrapper。
  - 已支持：`pi3_long`、`vggt_long`、`vggt_omega_long`、`da3_long`、`dust3r_long`、`must3r_long`、`scal3r_long`、`lingbot_map_long`、`moge`。

- `metric_scale/models/semantic_scale_estimator.py`
  - semantic scale 求解器核心。
  - 从 VLM anchors / semantic planes / masks / pointmap / confidence 计算 scale intervals。
  - 支持 robust consensus、weighted BI、IFPA、consensus prior、class allowlist、anchor confidence / geometry filters。

- `metric_scale/models/vlm_semantic_scale_anchor.py`
  - VLM 提取 dynamic height anchors、static traffic-facility anchors 和 scene planes。
  - prompts 从 `metric_scale/prompts/{dataset}.py` 或 default fallback 加载。
  - bbox 使用 normalized `[0,1000]`，再转 pixel bbox。

- `metric_scale/metrics/metrics.py`
  - LiDAR projection sparse depth/point metrics。
  - Sparse reconstruction proxy。
  - TUM-style pose ATE / RPE。
  - Distance-bin metrics、dataset/sequence reports、sequence difficulty grouping。
  - 文件非常大，已经超过本项目小文件原则，后续适合拆分。

- `dataset_tools/`
  - 数据集转换、验证、清理、sequence filtering、camera ego / BEV / sparse reconstruction visualization。

## 配置树现状

旧的单文件 dataset configs 已被删除，新结构改为 `configs/{dataset}/{model}.yaml`：

- datasets：`argoverse`、`dair_v2x_i`、`kitti`、`nuscenes`、`v2x_radar_i`、`waymo`。
- model configs：`pi3_long`、`vggt_long`、`vggt_omega_long`、`da3_long`、`dust3r_long`、`must3r_long`、`scal3r_long`、`lingbot_map_long`，部分 dataset 还有 `default` 或 `pi3x_long`。
- 当前 `config.py` 默认指向 `configs/nuscenes/lingbot_map_long.yaml`。
- 大多数长序列配置使用 `frame_count: 1000` 和 `subsequence_size: 1000`，旧 default 则保留 `60` 帧测试风格。
- `output_resolution` 在不同模型之间有 `original` / `native` 差异，评测时必须关注 sampling mode。

## 重要风险

- 多个 YAML 配置曾存在硬编码 API key。已迁移为 `vision.api_key_env: "DOUBAO_API_KEY"`，但旧凭据已经暴露过，仍建议立即 rotate。
- 当前 `ProcessingPipeline._process_sequence_task()` 只启用 scale stage，端到端运行前需要确认是否重新打开 VLM/SAM/depth stages。
- 当前工作区有大量未提交改动和未跟踪外部目录，后续任何清理/提交都需要先分清用户改动与生成产物。
- `metric_scale/metrics/metrics.py` 超过 3000 行，和 AGENTS.md 的 small file principle 冲突。
- 当前仓库没有 `pyproject.toml` / `requirements.txt`，依赖实际来自外部环境和 sibling repos；复现门槛高。
- `README.md` 已更新，但仍需要后续和实际端到端运行结果持续校准。

## 建议补档顺序

1. Rotate 旧 API key，并确认远端/本地服务已经撤销旧凭据。
2. 为 config loading 和 `vision.api_key_env` 增加轻量测试。
3. 为 semantic-scale solver 补最小单元测试，尤其是 scale interval aggregation 和 selected-scale gating。
4. 将 `metric_scale/metrics/metrics.py` 拆成 depth、point、reconstruction、pose、reporting 子模块。
5. 添加 repo-local dependency manifest，降低复现实验环境成本。
