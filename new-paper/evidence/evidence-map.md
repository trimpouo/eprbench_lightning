# Evidence Map — 主张 → 证据映射

> 每个主张的完整机器可读定义见 `../claims.yaml`；实验编号对应
> `../experiments/results.tsv` 与 `../experiments/README.md`。
> 统计状态说明：本项目所有结果均为**单卡单种子**训练，未做显著性检验；
> 下文 "single-run" 表示该结论基于单次运行的数值差。

## Claim C1 — SACF 在 EventVPR 同数据集上独立有效

Statement: SACF-only + text 在 EventVPR test 上 R@1 = 94.2，优于 baseline + text 的 93.6。

Evidence:

- E01: Baseline + text → R@1/5/10/100 = 93.6/95.6/96.6/99.2
- E02: SACF-only + text → 94.2/95.4/97.2/99.2（R@1 +0.6，R@10 +0.6，R@5 -0.2）

Counter-evidence:

- E02 的 R@5（95.4）略低于 E01（95.6），增益并非全指标一致。

Statistical status:

- single-run；增益 0.6 个百分点，与预期单种子波动（±0.5）同量级 → 方向性证据。
- 旁证：历史 sweep（H06 等）与附录配方 A 在另一基线上也观察到 +0.7 左右的独立增益。
- **MLflow 复核（2026-08-06）**：`logs/mlflow.db` 中 baseline 4 次独立重复 std ≈ 0.36；
  AdamW 批次 sacfW R@1 = 95.10 vs baselineW 94.33（M01/M02），**+0.77 超出噪声带**，
  且与 H06（+0.69）、E02（+0.6）三条独立证据链同向 → C1 置信度上调为 medium-high。
  详见 `mlflow-analysis.md` 第 1、2.1 节。

Source: `docs/eventvpr_exploration_journey.md` 实验一、二。

## Claim C2 — SACF 提升跨数据集泛化

Statement: EventVPR → NYC Event 迁移，SACF-only R@1 = 76.2，比 baseline（73.3）高 2.9。

Evidence:

- E03: Baseline → NYC Event：73.3/87.5/91.2/97.6
- E04: SACF-only → NYC Event：76.2/88.9/91.7/98.1（全部指标领先）

Counter-evidence:

- 无直接反例；但绝对 R@1 从 ~94 降至 ~76，域 gap 依然显著，SACF 并未解决迁移问题本身。

Statistical status:

- single-run；+2.9 的幅度明显大于预期波动，方向性结论较稳。
- 仅单向（EventVPR→NYC）、单对数据集。

Source: `docs/eventvpr_exploration_journey.md` 实验三。

## Claim C3 — SCFM 与 SACF 叠加产生负向干扰

Statement: NYC Event+RGB 联合训练中 SACF only（Event test R@1 = 93.4）优于 SACF+SCFM（92.8）。

Evidence:

- E05: SACF only → Event 93.4 / RGB 99.3
- E06: SCFM only → Event 93.1 / RGB 98.9
- E07: SACF+SCFM → Event 92.8 / RGB 99.2
- 验证集同向：SACF only Event val R@1 = 58.4 vs SACF+SCFM 57.4（-1.0）
- 旁证：历史附录 A.3（`docs/eventvpr_exploration_journey.md`）在 EventVPR 上
  也观察到“加入 SCFM 不冲突但无进一步提升”。

Counter-evidence:

- RGB test 上 SACF+SCFM（99.2）与 SACF only（99.3）几乎无差；负向干扰集中在 Event 模态。
- 联合训练使 RGB R@1 相对单模态基线提升（98.5 → 99.2），联合训练本身对 RGB 有益。
- **MLflow 复核（2026-08-06）**：EventVPR 上 sacf-scfmW（95.10）与 sacfW（95.10）持平
  （M02/M04），即负向干扰是 NYC 小数据场景特有；在 EventVPR 上 SCFM 表现为中性冗余。

Statistical status:

- single-run；Event 上 -0.6 ~ -1.0 的下降方向在 val 与 test 上一致，较可信；
  但无法排除数据规模（~1800 places）导致的偶然性。

Source: `docs/nyc_vpr_sacf_scfm_ablation.md` 第 4–6 节。

## Claim C4 — 文本分支在跨数据集推理时可安全关闭

Statement: 有文本训练的模型迁移 NYC Event 时关闭文本分支仍保持竞争力。

Evidence:

- E03、E04 均在 `data.use_text=false model.use_text=false` 下取得上表结果。

Counter-evidence / 缺口:

- 未做“同一 checkpoint 开文本 vs 关文本”的对照；该主张目前是间接证据。

Statistical status: inconclusive（方向性观察）。

Source: `docs/eventvpr_exploration_journey.md` 实验结论第 3 条。

## Claim C5 — SACF 插入层 sweet spot 为 layer_idx = 6

Statement: 历史 sweep 中 layer_idx=6 增益最大（R@1 +0.69 vs 当时 baseline 94.33）。

Evidence:

- H01–H07（layer_idx = 1..7）：R@1 分别为 +0.23/+0.38/+0.38/+0.38/+0.23/+0.69/+0.46。
- **MLflow 复核（2026-08-06）**：H01–H07 即 `logs/mlflow.db` exp1 的 sacf1W–sacf7W；
  layer 6 另有独立复跑 sacfW（M02，R@1 = 95.10），与 sacf6W（95.02）构成
  **双重复现的最高点**；且 7 个层位全部 ≥ 同批 baseline（94.18–94.33）。

Counter-evidence:

- 各层增益差距不大（0.23–0.69），sweet spot 结论对噪声敏感；
  但 layer 6 是唯一被两次独立运行确认的最高层位。

Statistical status: single-run sweep；历史基线与当前不同，仅作方向参考。

Source: `docs/eventvpr_exploration_journey.md` 附录 A.2。
