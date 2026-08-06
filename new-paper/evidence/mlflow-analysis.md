# MLflow 实验数据库分析 — `logs/mlflow.db`

> 数据来源：`logs/mlflow.db`（SQLite，MLflow backend store）。
> 全量 run 级导出见 `../experiments/mlflow_runs.tsv`（116 行，含 run_id、git commit、
> 配置摘要、时长、val/test 指标）。本文是综合判读层；数值以 TSV 为准。
> 分析时间：2026-08-06。

## 0. 数据库概览

| MLflow experiment | 含义 | runs | FINISHED | FAILED | RUNNING |
|---|---|---|---|---|---|
| 0 `Default` | MLflow UI 冒烟测试 | 2 | 2 | 0 | 0 |
| 1 `eprbench` | 主消融网格（EATS/ES-SACF/ER-SCFM/core）+ SACF 层位 sweep | 44 | 43 | 1 | 0 |
| 2 `eprbench_sacfablations` | SE-RSACF 变体消融 + v2 复跑（失败） | 26 | 18 | 8 | 0 |
| 3 `eprbench_mods_ablation` | 10-epoch 快速筛选（频域/新模块） | 35 | 27 | 6 | 2 |
| 4 `eprbench_v2` | 50-epoch AdamW 复跑批次（全部失败） | 9 | 0 | 9 | 0 |

- 时间跨度：2026-06-03 ~ 2026-06-22；所有 run 来自 `src/train.py`，用户 `featurize`。
- 统一训练协议（除注明外）：EventVPR → EventVPR，文本分支开启（`use_text=True`），
  `lambda_contrast=0.15`，`max_epochs=50`，early stopping patience=20，
  实际训练 22–50 epochs（单卡，单次时长 0.8–1.9 h）。
- 优化器分两批：06-03 批次为 **Adam**，06-04 起全部切换为 **AdamW**
  （run 名后缀 `W`，见 failure-cases.md N7）。
- 指标：`test/R1` 等与 `val/R1_best` 完全一致（val/test 同 split，项目既定协议）。
- 2 个 `RUNNING` 状态 run（exp3 的 smoke-test 与 eats-only）是进程中断留下的僵尸记录，
  无 test 指标，分析中剔除。

## 1. 噪声水平估计（重要先验）

exp1 中 baseline 配置被独立重复训练了 3 次（均训满 26+ epochs）：

| run | 优化器 | epochs | R@1 |
|---|---|---|---|
| baseline (06-03) | Adam | 27 | 93.49 |
| baseline (06-03 晚) | Adam | 26 | 94.25 |
| baseline (06-04) | AdamW | 32 | 94.18 |

另有 AdamW 的 `baselineW`（41 ep，R@1 = 94.33）。合并 4 次独立重复：
**mean = 94.06，std ≈ 0.36，range = 0.84**。

结论：**单种子 R@1 差异 < ~0.5 个百分点时不能视为有效增益**。
下文所有判读均以此为标尺——本项目绝大多数消融差异都在噪声带内，
只有超出噪声带的差异才标注为"稳健"。

## 2. exp1 `eprbench`：主消融网格（44 runs）

### 2.1 SACF / SCFM 基础消融（AdamW 批次）

| run | 配置 | ep | R@1 | R@5 | R@10 | R@100 | ΔR@1 vs baseline |
|---|---|---|---|---|---|---|---|
| baselineW | 全关 | 41 | 94.33 | 96.02 | 96.70 | 99.08 | — |
| sacfW | SACF(original, layer6) | 37 | **95.10** | 96.25 | 97.32 | 99.08 | **+0.77** |
| scfmW | SCFM(original) | 27 | 94.33 | 95.79 | 96.78 | 99.00 | 0.00 |
| sacf-scfmW | SACF+SCFM | 39 | **95.10** | 96.09 | 97.16 | 99.16 | +0.77 |

- **SACF 的 +0.77 超出噪声带（std 0.36），是全部 116 个 run 中最稳健的单模块增益**，
  与 H 系历史 sweep（+0.69）和 E01→E02（+0.6）三个独立证据链方向一致 → 支撑 C1/C5。
- SCFM 单独无增益；SACF+SCFM 不损失但也不再提升（增益完全由 SACF 贡献），
  与 NYC 联合训练中的负向干扰（C3）相比，EventVPR 上表现为"中性冗余"。

### 2.2 EATS / ES-SACF / ER-SCFM 网格（AdamW 批次，06-04/05）

| run | ep | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|---|
| baseline (06-04) | 32 | 94.18 | 96.32 | 97.01 | 99.16 |
| eats-only | 30 | 94.33 | 95.63 | 97.01 | 99.16 |
| essacf-only | 28 | 93.56 | 95.33 | 96.70 | 99.16 |
| erscfm-only | 27 | 93.95 | 95.56 | 96.40 | 99.08 |
| eats-essacf | 36 | 94.79 | 95.79 | 96.78 | 99.31 |
| eats-erscfm | 49 | 93.79 | 95.33 | 96.70 | 99.16 |
| essacf-erscfm | 32 | 94.25 | 95.71 | 96.70 | 99.08 |
| eats-essacf-erscfm | 43 | 94.41 | 95.79 | 96.70 | 99.16 |
| orig-sacfm | 28 | 94.33 | 95.71 | 96.86 | 99.23 |
| eats-orig-sacfm | 27 | 94.64 | 95.63 | **97.24** | 99.16 |

判读：

- `eats-essacf`（94.79，+0.61）与 `eats-orig-sacfm`（94.64，+0.46）处于噪声带上沿，
  方向性证据：EATS 与 SACF 组合可能有益，但单次运行不足以定论。
- `essacf-only`（93.56）反而低于基线 → 再次确认 N5：ES 门控不能脱离 EATS 单独开启。
- 全组合 `eats-essacf-erscfm`（94.41）不如精选两模块组合 → 与 N8 同向：组件不可加。

### 2.3 core 附加技术（temporal consistency / text dropout）

| run | 附加项 | ep | R@1 | R@100 |
|---|---|---|---|---|
| eats-essacf-erscfm（参照） | — | 43 | 94.41 | 99.16 |
| core-temporal | +use_temporal_consistency | 29 | 94.18 | 99.16 |
| core-textdrop | +text_dropout 0.15 / word_dropout 0.10 | 32 | 94.48 | **99.39** |
| core-temporal-textdrop | 两者 | 34 | 94.71 | 99.08 |

- temporal consistency 无增益（94.18 ≤ 94.41）。
- text dropout 单独 +0.07（噪声内），但与 temporal 组合达 94.71；R@100 单项最高
  （99.39，全库最高）出现在 core-textdrop。整体属噪声带内的弱信号，不足以立主张。

### 2.4 SACF 层位 sweep（sacf1W–sacf7W + sacfW，AdamW）

| layer_idx | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|
| 1 | 94.56 | 96.02 | 97.39 | 99.00 |
| 2 | 94.71 | 96.09 | 96.86 | 99.00 |
| 3 | 94.71 | 95.86 | 96.86 | 99.31 |
| 4 | 94.71 | 96.17 | 97.16 | 99.31 |
| 5 | 94.56 | 95.48 | 96.70 | 99.00 |
| **6** | **95.02**（sacf6W）/ **95.10**（sacfW 复跑） | 96.02/96.25 | 96.78/97.32 | 99.23/99.08 |
| 7 | 94.79 | 96.02 | 97.16 | 99.31 |

- 7 个层位**全部 ≥ baseline**（94.18–94.33），说明 SACF 本身稳健、层位是二阶调参。
- layer 6 两次独立运行 95.02 / 95.10，是唯一的双重复现最高点 → **C5 的 sweet spot
  结论从"单次 sweep"升级为"双重复现"，证据强度上升**。

## 3. exp2 `eprbench_sacfablations`：SE-RSACF 变体消融（18 有效 runs）

统一协议：variant=se_rsacf，layer 6（除注明），AdamW，50ep 上限。
当批 baseline R@1 = 93.87。

| run | 关键差异 | R@1 | Δ vs baseline |
|---|---|---|---|
| sacf-original-l6 | 普通 SACF 参照 | 94.18 | +0.31 |
| essacf-l6 | event_stability 变体 | 94.56 | +0.69 |
| sersacf-no-sem-no-evt | 双 prior 全关 | 94.56 | +0.69 |
| sersacf-sem-only | 仅语义 prior | 94.33 | +0.46 |
| sersacf-evt-only | 仅事件 prior | 94.48 | +0.61 |
| sersacf-sem-evt | 双 prior | **94.64** | **+0.77** |
| sersacf-conv-gate | Conv gate | 93.95 | +0.08 |
| sersacf-fastkan-gate | FastKAN gate | 94.33 | +0.46 |
| sersacf-fastkan-hidden32 | KAN hidden=32 | 93.95 | +0.08 |
| sersacf-layer4 | 插 layer 4 | 94.18 | +0.31 |
| sersacf-layer8 | 插 layer 8 | 94.41 | +0.54 |
| sersacf-layers468-mix | 多层混合+可学习权重 | **94.79** | **+0.92** |
| sersacf-layers468-textgate | 多层+文本门控 | 94.56 | +0.69 |
| sersacf-dilated13 | dilation [1,3] | 93.87 | 0.00 |
| sersacf-adaptive-topk | 自适应文本 topk | 94.33 | +0.46 |
| sersacf-no-aux-reg | 去辅助正则 | 93.95 | +0.08 |
| sersacf-full | 全部 trick 堆叠 | 94.33 | +0.46 |

判读（与 failure-cases.md N3/N4/N6/N8 互为印证）：

- **sem+evt 双 prior 是 SE-RSACF 的有效核心**（94.64）；单独任一 prior 都更弱。
- **layers468-mix 全场最高（94.79，+0.92）**，是唯一接近"超噪声"的变体。
- gate 结构（Conv/FastKAN/KAN-32）均无优势甚至有害；dilation 无效；去正则显著回落
  （94.64→93.95）→ 辅助正则是必要组件。
- 全部 trick 堆叠（sersacf-full 94.33）不如精选组合：组件增益不可加。

## 4. exp3 `eprbench_mods_ablation`：10-epoch 快速筛选（27 有效 runs）

协议注意：**max_epochs=10，batch 12**（与其余组 50ep/batch 18 不同），
数值不可与 exp1/exp2 直接比较，仅用于同组内排序。当组 baseline R@1 = 94.02。

| run | 模块 | R@1 | 组内 Δ |
|---|---|---|---|
| scfm-original | SCFM | **94.48** | +0.46 |
| eats-orig-sacfm | EATS+SACF+SCFM(original) | **94.48** | +0.46 |
| lgfp | 局部-全局频域处理 | 94.41 | +0.39 |
| sersacf-full | SE-RSACF 全堆叠 | 94.33 | +0.31 |
| eats-essacf-erscfm | 全组合 | 94.33 | +0.31 |
| adaptive-topk | 自适应 topk | 94.33 | +0.31 |
| igr / cross-spectral | 频域增强 | 94.25 | +0.23 |
| freq-enhance | 频率增强损失 | 93.87 | -0.15 |
| fbgw / bim | 频域模块 | 93.87 / 93.56 | 负 |
| temporal-fusion | 时序融合 | 93.33 | -0.69 |
| spm | 显著性原型模块（batch 4 才跑通） | **92.87** | **-1.15** |

判读：

- 10-epoch 下全部差异（-1.15 ~ +0.46）基本处于短训噪声范围，**只能用于淘汰，
  不能用于排名定论**：spm、temporal-fusion、bim、fbgw 可优先放弃；
  scfm/lgfp/eats-orig-sacfm 值得进入 50-epoch 复验。
- spm 另两次以 batch 12/6 启动即 FAILED，batch 4 才完成 → 显存代价高且效果最差，
  双重负面（已记入 failure-cases）。
- exp3 前置的 fixed-test（1-epoch）与 smoke run 验证了 MLflow 记录链路本身，
  无分析价值但说明实验者做过管线验证，数据可信度加分。

## 5. exp4 `eprbench_v2` + exp2 尾部：失败的 50-epoch 复跑批次（17 runs 全失败）

- exp4（06-20/21，9 runs）与 exp2 尾部（06-22，8 runs）均为
  `baselineW / sacfW / scfmW / sacf-scfmW` 四配置的 50-epoch AdamW 复跑，**全部 FAILED**。
- 两种失败模式：
  1. **启动即失败**（duration ≈ 0，无 epoch 指标）：sacfW、sacf-scfmW 多次出现，
     疑似配置/代码回归；
  2. **训练完成后失败**（训满 24–49 epochs、有 val 指标但无 test 指标）：
     如 sacfW 49ep val R@1 = 94.10、sacf-scfmW 40ep val = 95.02、scfmW 24ep val = 94.71，
     疑似测试阶段崩溃（显存或数据加载）。
- 由于没有 run 写出 test 指标，**该批次不产出任何可用结论**；
  但 val 指标侧面提示复跑结果与 exp1 同配置一致（94.1–95.0）。
- mlflow 未记录失败原因文本；若要定位需查对应时段 Hydra 日志（已清理，不可考）。
  已记入 failure-cases.md N9。

## 6. 对现有主张的影响汇总

| 主张 | 本次新证据 | 影响 |
|---|---|---|
| C1（SACF 独立有效） | sacfW +0.77 超噪声带；与 H06、E02 三链同向 | **置信度可由 medium 上调至 medium-high** |
| C2（跨数据集泛化） | 无（mlflow 中无 NYC eval run） | 不变 |
| C3（SACF+SCFM 负干扰） | EventVPR 上 sacf-scfmW 与 sacfW 持平（95.10），即负干扰是 NYC 小数据特有 | 补充边界：EventVPR 上为中性冗余 |
| C4（文本可关） | 无新证据 | 不变 |
| C5（layer 6 sweet spot） | layer 6 双重复现 95.02/95.10 | **证据升级：单次 sweep → 双重复现** |

新产生的负面/边界记录：spm（N10）、v2 批次失败（N9）见 failure-cases.md。

## 7. 复现与追溯

- 单 run 追溯：`mlflow_runs.tsv` 的 `run_id`（前 8 位）+ `git`（commit 短哈希）
  可定位 MLflow UI 中的完整参数；本库共涉及 5 个 git commit。
- 批量复跑：`scripts/ablation_scfm_sacf.sh`（exp1 网格）、
  `scripts/se_rsacf_ablation.sh`（exp2）、`scripts/sacf_sweep.sh`（层位 sweep）。
- 读取数据库的最小代码：

```python
import sqlite3
con = sqlite3.connect("logs/mlflow.db")
for row in con.execute(
    "SELECT r.name, m.key, m.value FROM runs r "
    "JOIN latest_metrics m ON r.run_uuid = m.run_uuid "
    "WHERE m.key LIKE 'test/%' AND r.status = 'FINISHED'"
):
    print(row)
```
