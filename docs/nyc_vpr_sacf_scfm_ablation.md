# NYC-Event-VPR SACF / SCFM 联合训练消融实验

> 本文档记录在 NYC-Event-VPR 的 Event + RGB 联合训练设置下，对 **SACF (Spectral-Spatial Adaptive Fusion)** 与 **SCFM (Spatial-Channel Feature Modulator)** 进行的消融实验。

---

## 1. 实验动机

在之前的联合训练实验中发现：
- `SACF + SCFM` 联合开启时，Event test R@1 = 92.8，RGB test R@1 = 99.2。
- 单模态 visual-only 基线 Event test R@1 = 93.5，RGB test R@1 = 98.5。

这引出了两个问题：
1. SACF 与 SCFM 各自的独立贡献是什么？
2. 两者同时启用是否真的必要？

为此，在完全相同的 joint Event+RGB 数据设置下，分别训练：
- **SACF only**：开启 SACF，关闭 SCFM
- **SCFM only**：关闭 SACF，开启 SCFM

并与已完成的 **SACF + SCFM** 联合配置进行对比。

---

## 2. 实验设置

- 数据：`NYCVPRJointDataModule`（Event + RGB VG 训练/验证，VPR-Bench 测试）
- 模型：SGVPR + adapter，visual-only（`use_text=false`）
- 训练：30 epochs，validation 每 5 epochs
- Checkpoint：基于 Event val R@1 选择
- 优化器：AdamW lr=1e-4，weight_decay=1e-3，StepLR gamma=0.5 / 3 epochs
- 关键模块开关：

| 配置 | SACF | SCFM | 可训练参数量 |
|:---|:---:|:---:|:---:|
| SACF only | ✅ | ❌ | 28.1 M |
| SCFM only | ❌ | ✅ | 22.2 M |
| SACF + SCFM | ✅ | ✅ | 29.3 M |

---

## 3. 训练命令

```bash
# SACF only
python src/train.py \
  data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sac_only_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5

# SCFM only
python src/train.py \
  data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_scfm_only_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5
```

---

## 4. 验证集结果

| 配置 | Event val R@1 | Event val R@5 | Event val R@10 | Event val R@100 | RGB val R@1 | RGB val R@5 | RGB val R@10 | RGB val R@100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SACF only** | **58.4** | **75.1** | **79.4** | **91.2** | **79.0** | **90.6** | **93.5** | **98.2** |
| SCFM only | 57.8 | 74.7 | 79.5 | 91.1 | 78.6 | 90.2 | 93.1 | 98.0 |
| SACF + SCFM | 57.4 | 74.3 | 78.8 | 91.6 | 78.6 | 90.6 | 93.3 | 98.4 |

- **SACF only** 在 Event val 上取得了最高 R@1（58.4），比 SACF+SCFM 高 1.0 个百分点。
- SCFM only 的 Event val R@1（57.8）优于 SACF+SCFM（57.4），说明 SCFM 对 SACF 存在一定的干扰。
- RGB val 上三个配置差异较小（78.6 ~ 79.0）。

---

## 5. 官方测试集结果

| 配置 | Event test R@1 | Event test R@5 | Event test R@10 | Event test R@100 | RGB test R@1 | RGB test R@5 | RGB test R@10 | RGB test R@100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SACF only** | **93.4** | **98.0** | **98.7** | **99.7** | **99.3** | **99.8** | **99.9** | **99.9** |
| SCFM only | 93.1 | 97.7 | 98.6 | 99.7 | 98.9 | 99.5 | 99.8 | 99.9 |
| SACF + SCFM | 92.8 | 97.7 | 98.5 | 99.7 | 99.2 | 99.8 | 99.9 | 99.9 |
| 单模态 visual-only* | 93.5 | 97.8 | 98.6 | 99.7 | 98.5 | 99.9 | 99.9 | 99.9 |

\* 单模态 visual-only 为在 Event 或 RGB 上单独训练、无 SACF/SCFM 的基线，供参考。

---

## 6. 关键发现

### 6.1 SACF 是主要增益来源

- **SACF only 全面优于 SCFM only**：
  - Event test：93.4 vs 93.1（+0.3）
  - RGB test：99.3 vs 98.9（+0.4）
  - Event val：58.4 vs 57.8（+0.6）

- SACF 通过频域-空域自适应融合增强了 patch 特征的判别性，对 Event 边缘图和 RGB 图像均有效。

### 6.2 SCFM 对 SACF 有轻微负向干扰

| 指标 | SACF only | SACF + SCFM | 变化 |
|:---|:---:|:---:|:---:|
| Event val R@1 | 58.4 | 57.4 | **-1.0** |
| Event test R@1 | 93.4 | 92.8 | **-0.6** |
| RGB test R@1 | 99.3 | 99.2 | -0.1 |

- 在 SACF 已经较强的情况下，叠加 SCFM 没有带来提升，反而在 Event 上下降约 0.6~1.0 个百分点。
- 可能原因：
  1. SCFM 的通道-空间注意力与 SACF 的频域增强存在功能重叠，导致参数冗余。
  2. 两个模块串联增加了优化难度，在数据规模有限（~1800 places）时容易过拟合。
  3. SCFM 的 channel attention 可能抑制了 SACF 增强后的高频细节。

### 6.3 与单模态 visual-only 的对比

- **Event**：SACF only（93.4）接近单模态 visual-only（93.5），SACF+SCFM（92.8）略低。
- **RGB**：所有联合配置（98.9 ~ 99.3）均高于单模态 visual-only（98.5），说明联合训练本身对 RGB 是有益的，但模块选择以 SACF only 最佳。

---

## 7. 结论

1. **SACF 是有效模块**：在联合训练中，单独启用 SACF 即可达到最佳性能。
2. **SCFM 单独效果一般**：SCFM only 弱于 SACF only，与 `docs/eventvpr_exploration_journey.md` 附录 A 中“SCFM 单独几乎不涨点”的历史观察一致。
3. **SACF + SCFM 不是最优组合**：两者叠加在 Event 上产生负向干扰，RGB 上也无提升。
4. **推荐配置**：对于 NYC-Event-VPR 的联合训练，**仅启用 SACF** 是性价比最高的选择（28.1M 可训练参数， vs 29.3M for SACF+SCFM）。

---

## 8. 后续建议

- 尝试 SACF 的不同插入位置（`layer_idx`）和变体（`original` / `compact`），寻找更优设置。
- 对 SCFM 进行轻量化改造，或仅在 SACF 之前/之后作为辅助模块，避免功能重叠。
- 在更大规模数据集上验证 SACF+SCFM 是否仍然冲突，以判断是否为数据量不足导致。
- 记录 SACF 的残差损失与噪声损失权重，观察其对最终 recall 的影响。

---

## 附录 A：SACF+SCFM 联合训练历程（整合）

> 本节将原 `docs/nyc_vpr_sacf_scfm_joint_journey.md` 的核心内容合并到此，避免多份 NYC 文档分散。

### A.1 实现改动

- **Joint DataModule**：`src/data/nyc_vpr_joint_datamodule.py` 同时加载 Event 与 RGB 的 VG 数据，训练时用 `ConcatDataset` 混合，并通过 `rgb_place_offset=1_000_000` 避免 place ID 冲突。
- **多 dataloader 评估**：`src/models/eprbench_module.py` 的 `validation_step` / `test_step` 增加 `dataloader_idx`；第一个 dataloader 保留 `test/R@*` 命名，第二个使用 `test_rgb/R@*`。
- **配置**：`configs/data/nyc_vpr_joint.yaml`、`configs/experiment/nyc_vpr_sacf_scfm_joint.yaml`。

### A.2 训练命令

```bash
python src/train.py \
  data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sacf_scfm_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5
```

### A.3 联合 SACF+SCFM 结果

| 模态 | 最佳 epoch | R@1 | R@5 | R@10 | R@100 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Event val** | 24 | 57.4 | 74.3 | 78.8 | 91.6 |
| **Event test** | 24 | 92.8 | 97.7 | 98.5 | 99.7 |
| **RGB test** | 24/29 | 99.2 | 99.8 | 99.9 | 99.9 |

### A.4 与单模态 visual-only 基线对比

| 方法 | Event R@1/5/10/100 | RGB R@1/5/10/100 |
|:---|:---:|:---:|
| Visual-only（单模态） | 93.5 / 97.8 / 98.6 / 99.7 | 98.5 / 99.9 / 99.9 / 99.9 |
| SACF+SCFM（联合训练） | 92.8 / 97.7 / 98.5 / 99.7 | 99.2 / 99.8 / 99.9 / 99.9 |

**观察**：联合训练使 RGB test R@1 从 98.5 提升到 99.2，但 Event test 略降 0.7%；结合本文 6.2 节可知，该下降主要源于 SCFM 对 SACF 的轻微干扰。

---

## 附录 B：相关文件

| 文件 | 说明 |
|:---|:---|
| `configs/experiment/nyc_vpr_sac_only_joint.yaml` | SACF only 实验配置 |
| `configs/experiment/nyc_vpr_scfm_only_joint.yaml` | SCFM only 实验配置 |
| `configs/experiment/nyc_vpr_sacf_scfm_joint.yaml` | SACF+SCFM 联合训练配置 |
| `docs/eventvpr_exploration_journey.md` | EventVPR 上的 SACF/SE-RSACF 深度探索 |

---

*文档生成时间：2026-06-28*
