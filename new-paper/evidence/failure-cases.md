# Negative Results & Failure Cases — 失败实验与踩坑记录

> 这些结果不进入 `paper.md` 主线，但对复现者和后续智能体最有价值：
> 它们标记了已走过且无效的路线，请勿重复。
> 来源：`docs/eventvpr_exploration_journey.md` 附录 A.4/A.5、`docs/nyc_vpr_sacf_scfm_ablation.md`。

## N1 — SACF 训练在 batch size 18 下 OOM

- **Hypothesis**: SACF-only 可以直接沿用基线的 `train_batch_size=18`。
- **Result**: RTX A4000（15.6 GiB）上 OOM；SACF 前向保留额外激活。
- **Decision**: SACF 训练 batch 降至 12（inference 16）。代价：与基线 batch 18 不完全同配置，是 C1 的一个 protocol 瑕疵。
- **Artifacts**: `docs/eventvpr_exploration_journey.md` 实验二。

## N2 — SACF + SCFM 组合不增反降

- **Hypothesis**: SACF 与 SCFM 功能互补，叠加应进一步提升。
- **Result**: NYC 联合训练 Event test R@1 从 93.4（SACF only）降至 92.8；Event val 降 1.0。
- **Possible explanation**: 通道-空间注意力与频域增强功能重叠、参数冗余；小数据上过拟合；channel attention 抑制高频细节。
- **Decision**: 推荐配置为 SACF only（对应主张 C3）。
- **Artifacts**: `configs/experiment/nyc_vpr_sacf_scfm_joint.yaml`，E05–E07。

## N3 — FastKAN / RBF-KAN gate 无优势

- **Hypothesis**: 用 FastKAN gate 替代 Conv gate 可提升 SE-RSACF。
- **Result**: sersacf-fastkan-gate R@1 = 94.33，与 conv-gate（93.95）相比无稳定优势，低于双 prior 版本（94.64）。
- **Decision**: 不采用。详见 `docs/KAN_Complete_Guide_v2.md` 背景与附录 A.4。

## N4 — Multi-dilation 邻域聚合无效

- **Hypothesis**: dilation [1,3] 的多尺度邻域聚合可捕获更多上下文。
- **Result**: sersacf-dilated13 R@1 = 93.87，与 baseline 持平；事件帧稀疏性使 dilated 邻域与标准邻域几乎无差异。
- **Decision**: 不采用。

## N5 — ES-SACF 不能单独开启

- **Hypothesis**: Event-Stability 门控可独立使用。
- **Result**: 缺少 EATS/event_cues 时门控引入噪声，性能不稳定。
- **Decision**: ES-SACF 必须与 event cue 提取配套（配方 B）。

## N6 — 去掉 auxiliary regularization 性能下降

- **Hypothesis**: SE-RSACF 的残差/噪声正则可有可无。
- **Result**: sersacf-no-aux-reg R@1 = 93.95，明显低于双 prior 全开（94.64）。
- **Decision**: 保留正则；这是一个“减法实验”证据，方向与 N2 相反——并非所有组件都可省。

## N7 — Adam → AdamW

- **Observation**: 早期使用 Adam 时基线与 SACF 训练稳定性较差；改 AdamW（lr=1e-4, wd=1e-3）后改善。
- **Decision**: 全项目固定 AdamW；不要回退。

## N8 — trick 堆叠（sersacf-full）不如精选组合

- **Result**: sersacf-full（所有 trick 堆叠）R@1 = 94.33，低于双 prior + 多层混合的 94.79。
- **Lesson**: 组件增益不可加；每个新 trick 必须单独消融。

## N9 — `eprbench_v2` 50-epoch 复跑批次全部失败（17 runs）

- **Observation**: `logs/mlflow.db` 中 exp4（2026-06-20/21，9 runs）与 exp2 尾部
  （06-22，8 runs）的 `baselineW/sacfW/scfmW/sacf-scfmW` 复跑全部 FAILED。
  两种模式：①启动即失败（duration≈0，sacfW / sacf-scfmW 多次）；
  ②训练 24–49 epochs 后、写出 test 指标前失败（val R@1 已达 94.1–95.0）。
- **Result**: 该批次不产出任何可用 test 结论；mlflow 未记录失败原因，
  对应 Hydra 日志已清理，根因不可考。
- **Lesson**: 长时间批量复跑必须保留 Hydra 日志；训练结束到 test 之间
  （推断为测试阶段显存/数据加载）是第二个故障点，复跑前先单跑 1-epoch 冒烟。

## N10 — SPM 模块显存代价高且效果最差

- **Hypothesis**: 显著性原型模块（spm）可作为频域之外的增强路径。
- **Result**: 10-epoch 筛选中 batch 12 / 6 均启动即失败，降到 batch 4 才跑通；
  完成 run 的 R@1 = 92.87，为全组最低（baseline 94.02）。
- **Decision**: 放弃 spm 路线。来源：`experiments/mlflow_runs.tsv`（eb4c3574 等）、
  `evidence/mlflow-analysis.md` 第 4 节。
