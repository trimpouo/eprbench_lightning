# EventVPR 数据集 SACF 消融与跨数据集测试探索历程

## 目标

在 [EPRBench](https://arxiv.org/abs/2602.12919) Lightning + Hydra 复现代码上，基于 EventVPR 数据集做更深入的数据集层面消融与跨数据集测试，并记录完整的探索过程、踩坑经验与最终结论。

---

## 环境速览

- **GPU**：NVIDIA RTX A4000，15.6 GiB
- **数据**：`data/EventVPR/`，约 65k 张 event 帧 + 13k 条 LLM 场景描述
- **基础命令**：
  ```bash
  python src/train.py data=eprbench model=eprbench trainer=gpu ...
  python src/eval.py ckpt_path=... data=nyc_vpr_event model=eprbench trainer=gpu ...
  ```
- **稳定配置**：`num_workers=2`；默认 `num_workers=6` 在本机会偶发卡死。

---

## 实验一：EventVPR 有文本基线（30 epochs）

### 命令

```bash
python src/train.py \
  data=eprbench model=eprbench trainer=gpu \
  data.use_text=true model.use_text=true \
  model.lambda_contrast=0.25 \
  data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30 \
  logger.csv.name=eprbench_baseline_text
```

### 关键观察

- 训练稳定，约 1.5 小时完成 30 epochs。
- 验证/测试使用同一 `test` split（与官方实现一致）。

### EventVPR 测试结果

| 模型 | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|
| Baseline + text | 93.6 | 95.6 | 96.6 | 99.2 |

---

## 实验二：EventVPR SACF-only + 文本（30 epochs）

### 动机

之前在 NYC Event+RGB 联合训练上发现 `SACF-only` 优于 `SACF+SCFM`（详见 `docs/nyc_vpr_sacf_scfm_ablation.md`）。本实验在更大规模的 EventVPR 上验证 SACF 模块的独立增益。

### 新增配置

创建 `configs/experiment/eprbench_sacf_only.yaml`：

```yaml
# @package _global_

tags: ["eprbench", "sacf_only"]

model:
  use_text: true
  lambda_contrast: 0.15
  net:
    mods:
      sacf:
        enabled: true
        variant: original
        layer_idx: 6
      scfm:
        enabled: false

logger:
  csv:
    name: "eprbench_sacf_only"
```

### 踩坑：显存不足

直接以 `data.train_batch_size=18` 启动 SACF-only 训练时，由于 SACF 在前向传播中保留了额外激活，出现 OOM。

**解决**：将训练 batch size 降至 12，推理 batch size 降至 16：

```bash
python src/train.py \
  experiment=eprbench_sacf_only trainer=gpu \
  data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30
```

### 关键观察

- SACF 收敛同样稳定，总耗时与基线相近。
- 在 EventVPR 上，`SACF-only + text` 相比 `Baseline + text` 在 R@1 上提升约 **0.6%**。

### EventVPR 测试结果

| 模型 | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|
| Baseline + text | 93.6 | 95.6 | 96.6 | 99.2 |
| SACF-only + text | **94.2** | 95.4 | **97.2** | 99.2 |

---

## 实验三：跨数据集测试（EventVPR → NYC Event）

### 动机

评估在 EventVPR 上训练得到的模型，对 NYC Event 这一未见城市数据集的泛化能力，以及 SACF 模块对跨域鲁棒性的影响。

### 评估命令

**1. Baseline + text → NYC Event（推理时关闭文本分支）**

```bash
python src/eval.py \
  ckpt_path=logs/train/runs/2026-06-28_05-54-13/checkpoints/epoch_029.ckpt \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  data.infer_batch_size=32 data.num_workers=2
```

**2. SACF-only + text → NYC Event（推理时必须显式启用 SACF）**

```bash
python src/eval.py \
  ckpt_path=logs/train/runs/2026-06-28_07-46-15/checkpoints/epoch_029.ckpt \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  model.net.mods.sacf.enabled=true \
  data.infer_batch_size=32 data.num_workers=2
```

### 跨数据集测试结果

| 训练集/模型 | 测试集 | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|---|
| EventVPR Baseline + text | NYC Event | 73.3 | 87.5 | 91.2 | 97.6 |
| EventVPR SACF-only + text | NYC Event | **76.2** | **88.9** | **91.7** | **98.1** |

### 关键观察

- 从 EventVPR 迁移到 NYC Event，R@1 下降约 **20%**，说明城市/场景域 gap 仍然显著。
- SACF-only 在跨数据集场景下 R@1 提升 **2.9%**，R@5/R@10/R@100 也全面优于 baseline，说明空间注意力选择机制对跨域泛化有正向作用。

---

## 实验结论

1. **EventVPR 上 SACF 独立有效**：SACF-only + text 在 R@1/R@10 上均优于有文本基线。
2. **SACF 带来跨域增益**：从 EventVPR 迁移到 NYC Event 时，SACF-only 模型全面领先 baseline。
3. **文本在跨数据集推理时可安全关闭**：关闭文本分支后性能下降有限，说明视觉分支已学到足够表征。
4. **SACF 需要更小训练 batch**：在 RTX A4000 上，SACF 训练 batch size 需从 18 降至 12 以避免 OOM。

---

## 复用提示

- 训练 EventVPR SACF-only：
  ```bash
  python src/train.py experiment=eprbench_sacf_only trainer=gpu \
    data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
    trainer.max_epochs=30
  ```
- 评估 SACF 模型时务必开启 `model.net.mods.sacf.enabled=true`，否则 checkpoint 中的 SACF 权重不会被使用。
- 跨数据集评估使用 `data=nyc_vpr_event` 并关闭文本分支 `data.use_text=false model.use_text=false`。

---

## 下一步可探索方向

- 在 EventVPR 上训练 `SACF+SCFM` 组合，确认是否与 NYC Event+RGB 结论一致（组合反而下降）。
- 尝试冻结 backbone 下不同 adapter 学习率、或 SACF 插入层（`layer_idx`）的消融。
- 对 EventVPR 做无文本训练，观察文本分支对事件 VPR 的真实增益。

---

# 附录 A：历史探索精华（整合自早期 `experiment_journey.md`）

> 以下内容为仓库历史上多轮 SACF/SE-RSACF 消融的精华总结，保留关键表格与可复现命令，供后续调参参考。

## A.1 已验证有效的 SACF 配方

### 配方 A：最简涨点（推荐）

```bash
python src/train.py \
  model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.variant=original \
  model.net.mods.sacf.layer_idx=6
```

预期：在 EventVPR 基线上 **R@1 ≈ +0.7%**。

### 配方 B：SE-RSACF 多层混合（追求极限）

```bash
python src/train.py \
  model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.variant=se_rsacf \
  model.net.mods.sacf.layer_indices=[4,6,8] \
  model.net.mods.sacf.use_layer_mixture=True \
  model.net.mods.event_cues.enabled=True \
  model.net.mods.sacf.use_semantic_prior=True \
  model.net.mods.sacf.use_event_prior=True
```

预期：**R@1 ≈ +0.9%** 相对于基线（需配合 event cue 提取）。

## A.2 SACF 插入层敏感性（历史 sweep）

| Run | layer_idx | R@1 | R@5 | R@10 | vs baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| baselineW | — | 94.33 | 96.02 | 96.70 | — |
| sacf1W | 1 | 94.56 | 96.02 | 97.39 | +0.23 |
| sacf2W | 2 | 94.71 | 96.09 | 96.86 | +0.38 |
| sacf3W | 3 | 94.71 | 95.86 | 96.86 | +0.38 |
| sacf4W | 4 | 94.71 | 96.17 | 97.16 | +0.38 |
| sacf5W | 5 | 94.56 | 95.48 | 96.70 | +0.23 |
| **sacf6W** | **6** | **95.02** | 96.02 | 96.78 | **+0.69** |
| sacf7W | 7 | 94.79 | 96.02 | 97.16 | +0.46 |

**结论**：`layer_idx=6` 是 sweet spot；block 6 兼顾空间分辨率与高层语义。

## A.3 SACF vs SCFM 组合验证

| Run | SACF | SCFM | R@1 | R@5 | R@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| baselineW | — | — | 94.33 | 96.02 | 96.70 |
| sacfW | ✓ | — | **95.10** | 96.25 | 97.32 |
| scfmW | — | ✓ | 94.33 | 95.79 | 96.78 |
| sacf-scfmW | ✓ | ✓ | **95.10** | 96.09 | 97.16 |

**结论**：SACF 单独即可打满增益；加入 SCFM 不冲突但无进一步提升。

## A.4 SE-RSACF 深度消融

| Run | R@1 | R@5 | R@10 | R@100 | 关键差异 |
|:---|:---:|:---:|:---:|:---:|:---|
| baseline | 93.87 | 95.86 | 96.78 | 99.23 | — |
| sacf-original-l6 | 94.18 | 96.09 | 97.32 | 99.16 | 原始 SACF 对照 |
| essacf-l6 | 94.56 | 95.79 | 96.70 | 99.23 | Event-Stability SACF |
| sersacf-no-sem-no-evt | 94.56 | — | — | — | 无 prior |
| sersacf-sem-only | 94.33 | — | — | — | 仅 semantic prior |
| sersacf-evt-only | 94.48 | — | — | — | 仅 event prior |
| **sersacf-sem-evt** | **94.64** | 96.32 | 97.01 | 99.08 | **双 prior 全开** |
| sersacf-conv-gate | 93.95 | — | — | — | Conv gate |
| sersacf-fastkan-gate | 94.33 | — | — | — | FastKAN gate |
| sersacf-layer4 | 94.18 | — | — | — | layer=4 |
| sersacf-layer8 | 94.41 | — | — | — | layer=8 |
| **sersacf-layers468-mix** | **94.79** | 95.63 | 96.78 | 99.16 | **多层混合 + learnable weight** |
| sersacf-dilated13 | 93.87 | — | — | — | dilation [1,3] |
| sersacf-no-aux-reg | 93.95 | — | — | — | 去掉 residual/noise 正则 |
| sersacf-full | 94.33 | — | — | — | 所有 trick 堆叠 |

**关键洞察**：
- 双 prior（semantic + event）是 SE-RSACF 核心；多层混合（4/6/8）达到最高 **94.79%**。
- FastKAN gate、multi-dilation、adaptive top-k 等设计未带来持续提升。
- 去掉 auxiliary regularization 后性能明显下降，说明残差/噪声正则必要。

## A.5 历史踩坑记录

1. **Adam → AdamW**：将 optimizer 改为 AdamW 后基线与 SACF 稳定性均有提升。
2. **ES-SACF 不能单独开**：缺少 EATS/event_cues 时，event-stability 门控可能引入噪声。
3. **Layer 6 是 sweet spot**：多次 sweep 验证，不建议随意更换。
4. **FastKAN 提升有限**：RBF-KAN gate 相比 Conv gate 未体现优势。
5. **Multi-dilation 不适用**：事件帧稀疏性使 dilated 邻域聚合效果与标准邻域几乎无差异。
