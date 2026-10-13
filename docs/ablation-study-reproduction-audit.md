# ablation_study 复现审计

审计日期：2026-09-23。源工程：`/home/yanglei/ws_yiyang/ablation_study`。范围仅为 VGGT-Omega、nuScenes、FusionPortable；本次只读审计，没有修改原工程、安装环境或重新运行全量实验。

## 结论

目前不是可在新服务器一键安装并从原始数据运行的独立项目。主工程没有 requirements / environment.yml / pyproject / Dockerfile / install.sh；vendor 下其他模型的安装文件不能当成这个项目的依赖清单。

现有 launcher 使用 Conda base 做消融评测、pi-long 做 UniMetric 尺度恢复，不能据此称整套工程使用 QWen。裸 `python` 的脚本取决于调用时的环境。QWen 是环境名称，也不等于一定加载 Qwen 模型。

## 实际流程与外部依赖

1. `scripts/run_relative_scale.py`：VGGT-Omega 推理，写出 `relative_scale`。读取 `vendor/vggt-omega/source`，该 vendor 目录实际是项目外软链接。
2. `scripts/run_unimetric.py`：硬编码导入 `/home/yanglei/ws_yiyang/Metric-Depth-new` 的 pipeline，读取 `/data1/yanglei/outputs` 的 VLM/SAM 缓存，再运行尺度求解。该脚本本身不从图像生成 VLM/SAM。
3. `scripts/run_ablation.py`：读取 relative_scale 与 semantic_scale_debug，生成各尺度变体并评测 AbsRel、delta1、CD、ATE。输出目录包含逐序列结果、dataset summary 和 `ablation_table.txt`。
4. `final_8/*.sh`：继续历史 run_id、读取既有 Oracle 结果目录的补跑脚本，不适合作为干净安装的入口。

只运行 VGGT-Omega 的六类结果时，显式使用：

```text
none,metric_model,midpoint_mean,midpoint_median,table_lookup,oracle
```

默认不传 `--variants` 会额外要求 MoGe-2、MoGe-3、UniDepth-v1/v2 标定对照。这四行不在本次模型范围；不必打包它们的权重、baseline_model、MoGe/UniDepth/FlexGEMM 源码。

## 现有环境版本（现场读取）

| 包 | base（评测 launcher） | pi-long（UniMetric launcher） |
|---|---|---|
| Python | 3.11.3 | 3.10.18 |
| torch | 2.6.0+cu124 | 2.5.1+cu118 |
| torchvision | 0.21.0+cu124 | 0.20.1+cu118 |
| numpy | 1.24.4 | 1.26.4 |
| scipy | 1.10.1 | 1.15.3 |
| opencv-python | 4.11.0.86 | 4.11.0.86 |
| PyYAML | 6.0.2 | 6.0.3 |
| Pillow | 9.5.0 | 12.0.0 |
| einops | 0.8.2 | 0.8.1 |
| transformers | 4.49.0 | 5.0.0 |

这是版本审计，不是经过干净环境安装测试的完整 lockfile。正式复现包应导出已验证环境的完整清单及可编辑安装源码位置，在独立环境验证后再提供安装脚本；不要把整个 base 的所有无关包机械塞进 requirements。

本次在源服务器对 base、pi-long、QWen 分别做了 VGGT-Omega 模型模块、工程 wrapper 和消融 scale_variants 的导入测试，三者均通过。QWen 测得 torch 2.6.0+cu124。因此 QWen 可作为后续统一环境的候选，但这不能证明历史实验全部运行于 QWen，也不能替代 GPU 推理和全流程测试。

源服务器 Omega `source` 仓库提交为 `39a0cb8af88554f15ddcb5354cd52bde588fa014`；应同时保留本地修改（如果存在），不能仅凭 commit 就假定源码完全相同。

## 权重

当前配置指定 `/data1/yanglei/FoundationModel/vggt-omega/checkpoints/vggt_omega_1b_512.pt`，现场文件存在，大小 4,576,706,117 字节。

官方源码：https://github.com/facebookresearch/vggt-omega

官方权重：https://huggingface.co/facebook/VGGT-Omega

官方说明要求先申请权重访问权限。获批后可使用 `hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints`；也可迁移服务器上已有的同一文件，并记录 SHA-256。为了重现现有结果，应保留 512 checkpoint、匹配源码和现有预处理，不替换成新增的 416-Reproduction 权重。

若完全重跑 VLM/SAM，还需 SAM2 源码、匹配配置及 SAM2 checkpoint，并明确 VLM API 后端/模型版本与凭据。若只复用既有尺度调试 JSON 进行消融评测，则不需要运行 VLM/SAM，也不需要下载 Qwen 权重。

## 缓存覆盖与打包边界

在原共享输出与本工程输出中联合检查：nuScenes 116 个序列、FusionPortable 5 个序列，均找到非空 relative pointmap、`shared_scale_*.json`、VLM results 和 SAM 目录。此检查只确认存在性，不代表已逐帧验证全部文件完整性。旧 PLAN.md 中“只有三个序列具备预测”的描述已过时。

有三种不同的复现层次：

- **重算结果表**：打包本工程源码、原始数据、relative_scale（pointmap/conf/camera_params 等完整结构）、semantic_scale_debug、现有参考结果与配置。此路线最容易对照现有数字，但没有重新做模型推理或语义提取。
- **重跑 Omega 与尺度恢复**：除数据和源码外，需 Omega 源码/权重、VLM/SAM 缓存，以及 `Metric-Depth-new` 的尺度求解代码和依赖。重新生成 relative_scale 后，应使用对应预测重新计算尺度，不要直接把旧预测的尺度套到新预测上。
- **完全从图像重跑**：还需补齐 VLM → SAM → Omega → 尺度恢复 → 评测的独立入口。当前 ablation_study 本身没有闭合这条链路。VLM 版本或非确定性输出变化可能导致结果与旧表不同，应保留原语义缓存作对照。

建议复现包拆成：

1. `ablation-code`：主代码、两个 YAML、真正的 Omega 源码、必要的尺度求解模块；记录源码版本/文件哈希。
2. `omega-checkpoint`：现有 512 权重及 SHA-256。
3. `semantic-cache`：仅这两个数据集的 VLM/SAM 缓存、调试 JSON；打包实际文件，不能只打包指回旧服务器的软链接。
4. `reference-results`：`result.txt`、对应各序列结果和数据集汇总、配置、序列/帧清单及环境清单。

数据 ZIP 已单独准备，不需要重复打包其他数据集。若只执行第一种路线，增加 relative_scale 数据包；若重新推理，可不搬大型预测数组。

## 需要补齐的可移植改造

- 将 data/source_output_root/Omega 源码和权重路径、`run_unimetric.py` 的 MD_ROOT/SOURCE_ROOT、shell 的 ROOT/解释器路径参数化。
- 把必要外部源码纳入包，处理 vendor 和 data 的软链接；不要整体 dereference 所有输出链接而意外复制整个共享盘。
- 写 `environment`/锁定依赖、`install.sh`、`download_weights.sh`（受限权重访问失败时明确说明）、`check_assets.py`、单序列 smoke 脚本和两数据集运行脚本。
- 在安装阶段下载权重；离线运行前确认所有资源齐全。现有 launcher 设置了 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE。
- 预检查逐帧 JPG/PCD/YAML、pointmap/conf/camera_params 和缓存帧号，避免旧输出优先命中或缺帧混用。
- 固定现有 chunk=60、overlap=20、resize_only、alignment=false、评测阈值 100m、相机/位姿和评测配置，不随意修改来凑跑通。
- 单序列冒烟通过后再运行全集；核对 nuScenes 116 序列/4627 帧、FusionPortable 5 序列/3847 帧和同一聚合口径。

路径和依赖修正后的示意命令（当前尚非开箱可运行承诺）：

```bash
python scripts/run_relative_scale.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278
python scripts/run_unimetric.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278
python scripts/run_ablation.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278 --variants none,metric_model,midpoint_mean,midpoint_median,table_lookup,oracle
```

FusionPortable 可先用 handheld_room00 验证；全集步骤要对所有序列运行尺度恢复，然后分别评测两个数据集。避免直接运行依赖历史 run_id 的 final_8 脚本。

## 对照结果与限制

原 `result.txt` 中 UniMetric full：nuScenes 为 0.1670 / 0.8391 / 1.0028 / 4.9629，FusionPortable 为 0.4178 / 0.5224 / 1.6907 / 14.8457（依次 AbsRel/delta1/CD/ATE）。这些是现有报告值，本次未重新计算。

原报告说明 nuScenes 的 scene-0060、scene-0177、scene-0557、scene-0778 和 FusionPortable 的 ugv_parking00 无有效尺度区间，相关方法回退 scale=1.0；复现时应保留此策略并显式报告，不能静默跳过这些序列。
