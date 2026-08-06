---
title: "SGVPR 复现与 SACF 空间注意力模块消融：事件流视觉地点识别的可执行研究包"
authors:
  - name: "EPRBench_lightning 项目组"
    role: "reproduction, ablation, writing"
version: "0.2.0"
date: "2026-08-06"
status: "working-draft"
research_question: >
  在事件流视觉地点识别（event-based VPR）中，频域-空域自适应融合模块（SACF）
  是否能在同数据集与跨数据集场景下稳定提升 SGVPR 的召回率？它与 SCFM 模块
  是否互补？
primary_claim: >
  SACF 单独启用即可在 EventVPR 同数据集（R@1 +0.6）与 EventVPR→NYC 跨数据集
  （R@1 +2.9）场景下带来一致增益；而 SCFM 与 SACF 叠加在 NYC 联合训练中产生
  负向干扰（Event test R@1 -0.6）。
tasks:
  - event-based visual place recognition
  - cross-dataset generalization
  - multi-modal (event + RGB) joint training
datasets:
  - EPRBench / EventVPR (65K event frames, 13K LLM scene descriptions)
  - NYC-Event-VPR (Event / RGB, VG split converted to EPRBench format)
metrics:
  - Recall@1
  - Recall@5
  - Recall@10
  - Recall@100
  - trainable parameter count
code: "../src"
configs: "../configs/experiment"
license: "CC-BY-4.0"
code_license: "见仓库根目录 LICENSE"
---

# SGVPR 复现与 SACF 模块消融研究包

> 本文件是整个研究成果的**入口文件**，不是全部成果。
> 主张的完整机器可读版本见 `claims.yaml`；证据映射见 `evidence/evidence-map.md`；
> 实验原始记录见 `experiments/results.tsv`；智能体操作说明见 `agent.md`。

## Claims（主张清单）

### C1 — SACF 在 EventVPR 同数据集上独立有效

SACF-only + text（R@1 = 94.2）优于有文本基线（R@1 = 93.6），R@10 同步提升（97.2 vs 96.6）。
MLflow 复核（M01/M02）：AdamW 批次 sacfW R@1 = 95.10 vs baselineW 94.33，+0.77。

- 支持：E01、E02、M01、M02
- 边界：单卡单种子训练；baseline 4 次独立重复 std ≈ 0.36，SACF 增益（+0.6~+0.8）超出噪声带，但未做显著性检验。
- 可信度：中高

### C2 — SACF 提升跨数据集泛化

EventVPR 训练的模型直接迁移到 NYC Event：SACF-only（R@1 = 76.2）全面优于
baseline（73.3），R@1 +2.9，且 R@5/10/100 全部领先。

- 支持：E03、E04
- 边界：推理时关闭了文本分支；仅验证了 EventVPR→NYC 单向迁移。
- 可信度：中高

### C3 — SCFM 与 SACF 叠加产生负向干扰

NYC Event+RGB 联合训练中，SACF only（Event test R@1 = 93.4）优于
SACF+SCFM（92.8）与 SCFM only（93.1）。

- 支持：E05、E06、E07
- 边界：NYC 数据规模有限（~1800 places）；更大规模数据上是否仍冲突未验证。
  MLflow 复核：EventVPR 上 SACF+SCFM 与 SACF only 持平（95.10 vs 95.10，M02/M04），
  负向干扰为 NYC 小数据特有。
- 可信度：中

### C4 — 文本分支在跨数据集推理时可安全关闭

EventVPR 上有文本训练的模型，在 NYC Event 推理时关闭文本分支仍保持竞争力，
说明视觉分支已学到足够表征。

- 支持：E03、E04（与表内同配置开文本结果的历史观察一致）
- 边界：未做开/关文本的系统对照实验，属间接证据。
- 可信度：低-中

### C5 — SACF 插入层存在 sweet spot（layer_idx = 6）

历史 sweep（H01–H07）显示 block 6 兼顾空间分辨率与高层语义，增益最大（+0.69）。
MLflow 复核：layer 6 经 sacf6W / sacfW 两次独立运行确认（95.02 / 95.10，M02），
且 7 个层位全部 ≥ 同批 baseline。

- 支持：H01–H07、M02
- 边界：各层增益差距不大（0.23–0.77），但 layer 6 是唯一双重复现的最高层位。
- 可信度：中高

## 结果总表（由 experiments/results.tsv 汇总）

### EventVPR 同数据集

| 实验 | 模型 | R@1 | R@5 | R@10 | R@100 |
|---|---|:---:|:---:|:---:|:---:|
| E01 | Baseline + text | 93.6 | 95.6 | 96.6 | 99.2 |
| E02 | SACF-only + text | **94.2** | 95.4 | **97.2** | 99.2 |

### 跨数据集（EventVPR → NYC Event，推理关闭文本）

| 实验 | 模型 | R@1 | R@5 | R@10 | R@100 |
|---|---|:---:|:---:|:---:|:---:|
| E03 | Baseline + text | 73.3 | 87.5 | 91.2 | 97.6 |
| E04 | SACF-only + text | **76.2** | **88.9** | **91.7** | **98.1** |

### NYC Event+RGB 联合训练（visual-only）

| 实验 | 配置 | Event test R@1 | RGB test R@1 | 可训练参数 |
|---|---|:---:|:---:|:---:|
| E05 | SACF only | **93.4** | **99.3** | 28.1 M |
| E06 | SCFM only | 93.1 | 98.9 | 22.2 M |
| E07 | SACF + SCFM | 92.8 | 99.2 | 29.3 M |
| E08 | 单模态 visual-only（参考） | 93.5 | 98.5 | — |

### MLflow 消融网格精选（EventVPR 同数据集，文本开启，AdamW，50ep 上限）

来源 `logs/mlflow.db` 复核；完整 116 行见 `experiments/mlflow_runs.tsv`，判读见
`evidence/mlflow-analysis.md`。同批 baseline R@1 = 94.18–94.33，重复实验 std ≈ 0.36。

| 实验 | 配置 | R@1 | R@5 | R@10 | R@100 |
|---|---|:---:|:---:|:---:|:---:|
| M01 | Baseline（AdamW） | 94.18 | 96.32 | 97.01 | 99.16 |
| M02 | SACF only（layer 6） | **95.10** | 96.25 | **97.32** | 99.08 |
| M04 | SACF + SCFM | **95.10** | 96.09 | 97.16 | 99.16 |
| M05 | EATS + ES-SACF | 94.79 | 95.79 | 96.78 | **99.31** |
| H09 | SE-RSACF layers468-mix | 94.79 | 95.63 | 96.78 | 99.16 |
| M09 | core + text dropout | 94.48 | 95.94 | 96.78 | **99.39** |

完整记录（含历史 sweep H 系列与 M 系列全部行）见 `experiments/results.tsv`。

## 方法概述

基于 SGVPR（Semantic-Guided VPR）的 Lightning + Hydra 复现：DINOv2 ViT-B/14
冻结 backbone + 可训练 adapter，多尺度 patch 聚合为 14 个 token 后经 2 层
TransformerEncoder；OpenCLIP 文本编码器提供场景描述条件；MultiSimilarityLoss
+ 图文对比损失。SACF 在指定 block（默认第 6 层）做频域-空域自适应融合；
SCFM 为通道-空间特征调制模块。详见 `../docs/baseline_sacf_pipeline.md` 与
`../src/models/components/mods/`。

## 讨论与适用边界

- 所有 EventVPR 结果为**单卡（RTX A4000, 15.6 GiB）、单种子**；MLflow 复核给出噪声标尺：baseline 4 次独立重复 std ≈ 0.36，R@1 差异 <0.5 的结论（C3 等）应视为方向性证据而非统计显著结论。
- 跨数据集实验（C2）的增益幅度（+2.9）大于同数据集（+0.6~+0.8），与“注意力选择机制提升域外鲁棒性”的假设一致，但仅单向、单对数据集验证。
- 历史 SE-RSACF 探索（附录 H 系列）提示双 prior + 多层混合仍有上限空间，但 trick 堆叠（FastKAN gate、multi-dilation）无持续收益，见 `evidence/failure-cases.md`。
- 10-epoch 快速筛选（exp3）只能用于淘汰（spm、temporal-fusion、bim、fbgw 负向），不能用于正面排名；scfm / lgfp / eats-orig-sacfm 值得 50-epoch 复验。

## 开放问题（供后续研究 / 智能体接手）

1. ~~在 EventVPR 上训练 SACF+SCFM 组合~~（已由 M02/M04 回答：EventVPR 上持平，负向干扰为 NYC 特有）；剩余问题：NYC 上为何冲突，需更大规模数据验证。
2. 对 C1、C3 做多种子（≥3）重复并报告均值 ± 方差（已有 baseline 重复 std ≈ 0.36 作为先验）。
3. 无文本训练消融，量化文本分支对事件 VPR 的真实增益（与 C4 直接相关）。
4. SACF 变体（`original` / `compact` / `se_rsacf`）在当前基线上的系统对比。

## 引用

本研究包基于 EPRBench 官方工作复现，引用见 `sources/bibliography.bib` 与根目录 `README.md`。
