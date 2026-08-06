# Changelog — new-paper 研究包

本文件记录研究包本身的版本历史（语义化版本）。规则：

- 新证据改变主张置信度 → minor version；
- 新主张或主张被推翻 → major version；
- 仅修订措辞/补充边界 → patch version。

## [0.2.0] - 2026-08-06

### Added

- 读取并分析 `logs/mlflow.db`（116 runs，2026-06-03 ~ 06-22）：
  - `experiments/mlflow_runs.tsv`：全部 run 的结构化导出（run_id、git commit、
    配置摘要、时长、val/test 指标）；
  - `evidence/mlflow-analysis.md`：分组详细判读（噪声估计、主消融网格、
    SE-RSACF 变体、10-epoch 筛选、失败批次）；
  - `experiments/results.tsv` 新增 M01–M15（MLflow 来源的关键配置）及复现命令。
- `evidence/failure-cases.md` 新增 N9（v2 复跑批次 17 runs 全失败）、N10（spm 双重负面）。

### Changed

- C1 置信度 medium → medium-high：baseline 4 次重复 std ≈ 0.36，sacfW 增益 +0.77
  超出噪声带，三条独立证据链同向。
- C5 置信度 medium → medium-high：layer 6 获 sacf6W / sacfW 双重复现（95.02/95.10）。
- C3 补充边界：负向干扰为 NYC 小数据特有，EventVPR 上 SCFM 为中性冗余（M02/M04）。

## [0.1.0] - 2026-08-06

### Added

- 初始化研究包：`paper.md`、`claims.yaml`（C1–C5）、`agent.md`。
- `experiments/results.tsv`：E01–E08（当前 30-epoch 实验）+ H01–H10（历史 sweep），
  数据来源 `docs/eventvpr_exploration_journey.md` 与 `docs/nyc_vpr_sacf_scfm_ablation.md`。
- `evidence/evidence-map.md` 与 `evidence/failure-cases.md`（N1–N8）。
- `sources/bibliography.bib` 与首篇 source note（EPRBench）。
