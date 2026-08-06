# Instructions for Research Agents — EPRBench_lightning 研究包

> 本文件面向复现、审查和继续推进本研究的智能体（及新协作者）。
> 它规定什么是证据、允许做什么、如何复现、审查时问什么问题。
> 对应 Karpathy `autoresearch` 中 `program.md` 的角色。

## Objective（目标）

验证或挑战 `claims.yaml` 中的主张 C1–C5，并在统一指标（Recall@1/5/10/100）下
推进 `paper.md` 末尾列出的开放问题。

## Ground truth（证据边界）

- **只有** `new-paper/experiments/results.tsv`、`logs/` 下的训练日志、以及
  `docs/` 中的结果表可被视为经验证据。
- `docs/` 与 `paper.md` 讨论部分的假设性解释（如“可能原因”）**不是**已验证结论。
- `README.md` / `AGENTS.md` 中的快速参考表是二手汇总，引用数字时应回到
  `results.tsv` 或原始日志核对。

## Do not（禁止事项）

- 不得编造缺失的实验结果；没有对应 `results.tsv` 行的数字不得写入主张。
- 不得把讨论中的假设当作已验证主张。
- 不得在无警告的情况下比较不同 protocol 下的结果（例如：有文本 vs 无文本、
  batch 18 vs batch 12、历史 sweep 基线 vs 当前基线）。
- 不得修改固定评估代码以“凑”结果（`src/eval.py`、`src/models/eprbench_module.py`
  中的 R@N 计算逻辑）；发现评估 bug 应单独报告，而不是顺手改掉后继续比较。
- 不得在未启用 `model.net.mods.sacf.enabled=true` 的情况下评估 SACF checkpoint
  并把结果当作 SACF 的真实性能。

## Environment（环境）

- GPU：NVIDIA RTX A4000，15.6 GiB（所有既有结果均在此环境产生）。
- 数据：`data/EventVPR/`（必需）、`data/NYC-Event-VPR-converted/`（跨数据集与联合训练）。
- 预训练权重：`model/dinov2_vitb14_pretrain.pth`。
- 依赖：`requirements.txt` 或 `environment.yaml`。
- **稳定设置**：`data.num_workers=2`（默认 6 在本机可能卡死）。

## Reproduction（最小复现路径）

1. 创建环境并准备数据与 DINOv2 权重（见根目录 `README.md` Quick Start）。
2. 复现基线 E01：`python src/train.py data=eprbench model=eprbench trainer=gpu data.use_text=true model.use_text=true model.lambda_contrast=0.25 data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 trainer.max_epochs=30`
3. 复现关键实验 E02（SACF）：`python src/train.py experiment=eprbench_sacf_only trainer=gpu data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 trainer.max_epochs=30`
4. 复现跨数据集 E04：`python src/eval.py ckpt_path=<E02 的 epoch_029.ckpt> data=nyc_vpr_event model=eprbench trainer=gpu data.use_text=false model.use_text=false model.net.mods.sacf.enabled=true data.infer_batch_size=32 data.num_workers=2`
5. 将新结果与 `results.tsv` 对应行比较；单种子波动预期在 ±0.5 R@1 以内，
   超出该范围必须先排查 protocol 差异，再考虑结论更新。

每条实验的完整命令见 `experiments/README.md`。

## Workflow for new experiments（新实验循环）

1. 在 `claims.yaml` 或开放问题中选定一个待验证对象。
2. 新建 `configs/experiment/` 配置或用命令行 override；保持评估 protocol 不变。
3. 训练/评估后把结果追加到 `experiments/results.tsv`（新编号），并更新
   `experiments/README.md` 的复现命令。
4. 按统一指标判断：达到预期 → 更新对应主张的证据与置信度；
   未达到 → 记入 `evidence/failure-cases.md`，**不要**悄悄删除。
5. 在 `CHANGELOG.md` 记录版本变更。

## Review questions（审查清单）

- 每个主要主张是否都有 `results.tsv` 中的直接证据行？
- 单种子结果是否被表述成了统计结论？（当前 C1、C3 的增益幅度 ≤1%，需谨慎措辞。）
- 基线与实验组的 protocol 是否可比（batch size、文本开关、epoch 数）？
- SACF checkpoint 的评估是否显式启用了 SACF？
- 是否有失败实验被遗漏在 `failure-cases.md` 之外？
- `paper.md` 是否遗漏了已知限制（单种子、单向迁移、数据规模）？
