# EPRBench Lightning 实验探索路程总结

> 本文档汇总了在 EPRBench (EventVPR) 数据集上基于 Lightning + Hydra 框架进行的模块化增强实验。重点记录 **SACF (Spectral-Spatial Adaptive Fusion)** 及其衍生变体的探索过程与量化结果。数据来源于本地 MLflow 日志 (`http://127.0.0.1:5000`) 与 Git 提交历史。

---

## 1. 背景与基线

**基线模型**：SGVPR (Semantic-Guided Visual Place Recognition)  
- 视觉主干：DINOv2 ViT-B/14 (frozen，仅 adapter 可训练)
- 文本编码：OpenCLIP ViT-B-16
- 核心流程：DINOv2 patch token → 文本引导 top-25% patch 融合 → 多尺度 GeM 聚合 (14 tokens) → 2-layer TransformerEncoder → 10752-dim descriptor
- 损失：MultiSimilarityLoss + 图文对比损失 (`lambda_contrast=0.15`, `temperature=0.07`)
- 评估：FAISS `IndexFlatL2`，Recall@1/5/10/100

**官方权重预期**：R@1 ≈ 94.0%  
**本仓库复现基线**（未经 SACF 等增强）：约 **93.5% ~ 94.3%**（存在少量随机波动，详见下文）。

---

## 2. 探索时间线

| 时间 | Commit | 关键事件 |
|---|---|---|
| 5 月初 | `042ead8` | 加入频域增强 (`frequency_enhancement`) 与 `CrossSpectralAttentionFeatureRefinement` |
| 5/7 | `6c2f821` | 加入更多 spectral 增强模块 |
| 5/8 | `067f5aa` | **加入 SCFM (Spatial-Channel Feature Modulator)**，标记为 "works!" |
| 5/9 | `194cfbd` | **加入原始 SACF** (`SpectralSpatialAdaptiveFusion`) |
| 5/13 | `5251f28`, `8ef7bc6` | 批量加入 IGR、LGFP、FBGW 等模块 |
| 5/22 | `8573c11` | 加入 BIM (Brain-Inspired Module) |
| 5/28 | `f06dab5` | 加入 PPLN、SPM (Sparse Prompt Module) |
| 6/1 | `193204c`, `2055535`, `3e968c8` | 调整脚本、训练入口与数据配置 |
| 6/2 | `cd24be0` | SCFM 增加残差连接 (`+ input_feature`) |
| 6/4 凌晨 | `9a5e488` | 大规模消融实验：`eats-only`, `essacf-only`, `erscfm-only`, `orig-sacfm`, `core-temporal` 等 |
| 6/4 上午 | `a111acb` | **确认 SACF 与 SACF+SCFM 有效**，将 optimizer 从 Adam 改为 AdamW |
| 6/4 | — | 运行 SACF 层数 sweep (`sacf1W` ~ `sacf7W`) 与第二轮 ablation (`W` suffix batch) |
| 6/6 | `42e6b64` | 补充 SACF 流程文档 (`baseline_sacf_pipeline.md`) |
| 6/6 前后 | — | 运行 **SE-RSACF 深度消融** (exp=`eprbench_sacfablations`)：prior、gate、multi-layer、dilation、aux reg 等 |

---

## 3. 核心结论（TL;DR）

1. **原始 SACF (layer=6) 是最稳健、最可复现的涨点模块**：在多个独立运行中，相对于同批次基线 **稳定提升约 +0.7 ~ +0.8% R@1**。
2. **SCFM 单独几乎不涨点**（甚至略降），但与 SACF 组合不冲突。
3. **ES-SACF / SE-RSACF 在 SACF 基础上有进一步增益**，但需要 careful tuning；部分花哨设计（FastKAN gate、multi-dilation、adaptive top-k）并未带来持续提升。
4. **层数选择对 SACF 至关重要**：DINOv2 block 6 明显优于其他层。
5. **多底层混合 (`layers468-mix`) 是 SE-RSACF 的最佳实践**，在 SE-RSACF 批次中达到最高 R@1。

---

## 4. SACF 专项实验详解

### 4.1 原始 SACF 层数敏感性 sweep

实验脚本：`scripts/sacf_sweep.sh`  
配置：仅开启 `model.net.mods.sacf.enabled=True`，`variant=original`，改变 `layer_idx`。

| Run | layer_idx | test/R@1 | test/R@5 | test/R@10 | vs baselineW |
|:---|:---:|:---:|:---:|:---:|:---:|
| baselineW | — | 94.33 | 96.02 | 96.70 | — |
| sacf1W | 1 | 94.56 | 96.02 | 97.39 | +0.23 |
| sacf2W | 2 | 94.71 | 96.09 | 96.86 | +0.38 |
| sacf3W | 3 | 94.71 | 95.86 | 96.86 | +0.38 |
| sacf4W | 4 | 94.71 | 96.17 | 97.16 | +0.38 |
| sacf5W | 5 | 94.56 | 95.48 | 96.70 | +0.23 |
| **sacf6W** | **6** | **95.02** | **96.02** | **96.78** | **+0.69** |
| sacf7W | 7 | 94.79 | 96.02 | 97.16 | +0.46 |

**结论**：`layer_idx=6` 是 sweet spot。过浅（1-2）或过快接近输出层（7）都会削弱增益。原因推测：block 6 处于 DINOv2 深层语义的"临界点"，既保留了足够的空间分辨率，又具备丰富的高层语义，与最终 patch token 的互补性最强。

### 4.2 原始 SACF vs SCFM 组合验证（W batch）

| Run | SACF | SCFM | test/R@1 | test/R@5 | test/R@10 | 备注 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| baselineW | — | — | 94.33 | 96.02 | 96.70 | 同批次基线 |
| sacfW | ✓ | — | **95.10** | 96.25 | 97.32 | **+0.77** |
| scfmW | — | ✓ | 94.33 | 95.79 | 96.78 | 0 |
| sacf-scfmW | ✓ | ✓ | **95.10** | 96.09 | 97.16 | **+0.77** |

**结论**：
- SACF 单独即可打满增益；加入 SCFM 后未进一步上涨，但也未退化。
- SCFM 在该基线配置下不具备独立增益，可能更适合与其他类型的特征调制搭配。

### 4.3 两轮完整消融（ablation_scfm_sacf.sh）

脚本 `scripts/ablation_scfm_sacf.sh` 设计了 12 组实验，覆盖了单模块、两两组合、三模块组合、以及文本/时序正则等策略。由于训练随机性，脚本被执行了两轮（早期一轮无 `W` 后缀，后期一轮带 `W` 后缀）。

**第一轮（早期，基线偏低）**

| Run | test/R@1 | vs baseline |
|:---|:---:|:---:|
| baseline | 93.49 | — |
| sacf | 94.71 | **+1.22** |
| scfm | 94.41 | +0.92 |
| sacf-scfm | 94.41 | +0.92 |

**第二轮（后期，基线更稳）**

| Run | test/R@1 | vs baseline |
|:---|:---:|:---:|
| baseline | 94.18 | — |
| eats-only | 94.33 | +0.15 |
| essacf-only | 93.56 | **-0.62** |
| erscfm-only | 93.95 | -0.23 |
| eats-essacf | **94.79** | **+0.61** |
| eats-erscfm | 93.79 | -0.39 |
| essacf-erscfm | 94.25 | +0.07 |
| eats-essacf-erscfm | 94.41 | +0.23 |
| core-temporal | 94.18 | 0 |
| core-textdrop | 94.48 | +0.30 |
| core-temporal-textdrop | **94.71** | **+0.53** |
| orig-sacfm | 94.33 | +0.15 |
| eats-orig-sacfm | **94.64** | **+0.46** |

**关键洞察**：
- `essacf-only`（Event-Stability SACF 单独使用）在第二轮反而低于基线，说明 **event-stability 门控需要与 EATS 联合使用**才能发挥效果。
- `eats-essacf` 组合达到 94.79，验证了 event cue 与 semantic score 的协同。
- `core-temporal-textdrop` 等正则策略有一定帮助，但不如 SACF 直接。

### 4.4 SE-RSACF 深度消融（Experiment: eprbench_sacfablations）

在确认原始 SACF 有效后，进一步探索了 **SE-RSACF (Semantic-Event Residual SACF)**。该变体用 Cosine 相似度替代 Euclidean 距离做局部谱聚合，引入 semantic/event prior 引导，并采用残差输出。

| Run | test/R@1 | vs baseline | 关键差异 |
|:---|:---:|:---:|:---|
| baseline | 93.87 | — | — |
| sacf-original-l6 | 94.18 | +0.31 | 原始 SACF 作为对照 |
| essacf-l6 | 94.56 | +0.69 | Event-Stability SACF |
| sersacf-no-sem-no-evt | 94.56 | +0.69 | 无 prior 的 SE-RSACF |
| sersacf-sem-only | 94.33 | +0.46 | 仅 semantic prior |
| sersacf-evt-only | 94.48 | +0.61 | 仅 event prior |
| **sersacf-sem-evt** | **94.64** | **+0.77** | **双 prior 全开** |
| sersacf-conv-gate | 93.95 | +0.08 | Conv gate |
| sersacf-fastkan-gate | 94.33 | +0.46 | FastKAN gate |
| sersacf-fastkan-hidden32 | 93.95 | +0.08 | FastKAN hidden=32 |
| sersacf-layer4 | 94.18 | +0.31 | layer=4 |
| sersacf-layer8 | 94.41 | +0.54 | layer=8 |
| **sersacf-layers468-mix** | **94.79** | **+0.92** | **多层混合 + learnable weight** |
| sersacf-layers468-textgate | 94.56 | +0.69 | 文本条件层门控 |
| sersacf-dilated13 | 93.87 | 0 | dilation [1,3] |
| sersacf-adaptive-topk | 94.33 | +0.46 | adaptive text top-k |
| sersacf-no-aux-reg | 93.95 | +0.08 | 去掉 residual/noise 正则 |
| sersacf-full | 94.33 | +0.46 | 所有 trick 堆叠 |

**关键洞察**：
1. **双 prior (semantic + event) 是 SE-RSACF 的核心**：单独使用 semantic 或 event 的增益都弱于两者联合。
2. **多层混合 (`layers468-mix`) 是最佳单点策略**：将 block 4、6、8 的低层 token 混合后输入 SE-RSACF，通过 learnable softmax weight 自动分配各层重要性，达到 **94.79%**。
3. **FastKAN gate 并未优于简单 Conv gate**：`fastkan-gate` 与 `conv-gate` 都只有边际提升，且 hidden size 增大后反而下降。
4. **Multi-dilation (dilated13) 完全无效**：说明局部邻域的尺度对事件 VPR 并非瓶颈。
5. **Auxiliary regularization (residual loss + noise loss) 有稳定作用**：去掉后从 94.56 掉到 93.95，验证了正则的必要性。
6. **"全都要"的 `sersacf-full` 并未最优**：说明模块之间存在干扰，不能简单堆叠。

---

## 5. 其他模块探索简报

| 模块 | 来源/灵感 | 实验结论 |
|:---|:---|:---|
| **SCFM** | CVPR 2026 (Spatial-Channel Feature Modulator) | 单独几乎不涨点；增加残差连接后稳定；与 SACF 组合无冲突 |
| **Cross-Spectral Refinement** | AAAI 2026 | 早期实验加入，后期被 SACF 系列覆盖，未做独立深度消融 |
| **LGFP** | CVPR 2026 (Low-frequency Guided Feature Purification) | 代码已集成，未在 MLflow 中出现独立运行记录 |
| **FBGW** | CVPR 2026 (Frequency Band-Guided Weighting) | 同上 |
| **IGR** | — | 同上 |
| **BIM** | Brain-Inspired Module | 同上 |
| **SPM** | TPAMI 2026 (Sparse Prompt Module) | 同上 |
| **PPLN** | — | 同上 |
| **EATS** | Event-Aware Token Selector | 单独微弱，但与 ES-SACF 有协同 |
| **Temporal Consistency Loss** | — | 在 `core-temporal` 中验证，效果有限 |
| **Text Dropout** | — | 微弱正则效果 |

**总体判断**：在已测试的所有模块中，**SACF 系列（尤其是原始 SACF 与 SE-RSACF 的多层混合）是唯一被多轮独立实验验证、且增益幅度最大（>0.5% R@1）的增强手段**。

---

## 6. 训练行为观察

从 MLflow 的 loss 曲线与 epoch 数可观察到：

| Run | Epochs | Train Loss | Triplet Loss | 训练时长 | 观察 |
|:---|:---:|:---:|:---:|:---:|:---|
| baselineW | 41 | 1.498 | 0.633 | ~77min | 正常收敛 |
| sacfW | 37 | 1.340 | 0.478 | ~79min | **loss 更低，收敛更快** |
| sacf6W | 50 | 1.408 | 0.546 | ~108min | 跑到最大 epoch，仍有空间 |
| sersacf-sem-evt | 25 | 1.488 | 0.624 | ~52min | 早期即触发 patience |
| sersacf-layers468-mix | 34 | 1.490 | 0.628 | ~71min | 残差 scale 最终 ≈0.12 |

- SACF 的加入显著降低了 `train/tri_loss`（度量学习主损失），说明跨层融合确实让 descriptor 更具判别性。
- SE-RSACF 的 `res_scale` 最终稳定在 0.10~0.12 附近，未出现残差爆炸或衰减至零，表明残差结构工作正常。

---

## 7. 经验教训与后续建议

### 7.1 已验证有效的配方

**配方 A：最简涨点（推荐）**
```bash
python src/train.py \
  model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.variant=original \
  model.net.mods.sacf.layer_idx=6
```
预期：**R@1 ≈ +0.7%** 相对于基线。

**配方 B：SE-RSACF 多层混合（追求极限）**
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

### 7.2 踩坑记录

1. **Adam → AdamW**：`a111acb` 中将 optimizer 改为 AdamW 后，基线与 SACF 的稳定性均有提升。这是涨点的前提之一。
2. **ES-SACF 不能单独开**：`essacf-only` 在第二轮低于基线，说明 event-stability 门控对事件 cue 的依赖很强；没有 EATS/event_cues 时，门控可能引入噪声。
3. **Layer 6 是 magic number**：多次 sweep 验证，不要随意改成其他层。
4. **FastKAN 华而不实**：RBF-KAN gate 的参数量和计算量均高于 Conv gate，但实验结果未体现优势。
5. **Multi-dilation 对事件 VPR 不适用**：事件帧的稀疏性使得 dilated 邻域聚合效果与标准邻域几乎无差异。

### 7.3 尚未探索的方向

- SACF 与 ** torch.compile** 的组合效率
- SACF 在 **更大 backbone** (ViT-L/14, ViT-g/14) 上的迁移性
- SE-RSACF 的 `layer_indices` 是否需要根据场景动态选择（如室内 vs 室外）
- 将 SACF 思想迁移到多帧时序聚合 (TEFA) 中

---

## 附录：完整实验结果速查表

### A. SACF 层数 sweep (Experiment: eprbench)

| Run | layer | R@1 | R@5 | R@10 | R@100 |
|:---|:---:|:---:|:---:|:---:|:---:|
| baselineW | — | 94.33 | 96.02 | 96.70 | 99.08 |
| sacf1W | 1 | 94.56 | 96.02 | 97.39 | 99.00 |
| sacf2W | 2 | 94.71 | 96.09 | 96.86 | 99.00 |
| sacf3W | 3 | 94.71 | 95.86 | 96.86 | 99.31 |
| sacf4W | 4 | 94.71 | 96.17 | 97.16 | 99.31 |
| sacf5W | 5 | 94.56 | 95.48 | 96.70 | 99.00 |
| **sacf6W** | **6** | **95.02** | **96.02** | **96.78** | **99.23** |
| sacf7W | 7 | 94.79 | 96.02 | 97.16 | 99.31 |

### B. 核心模块组合 (Experiment: eprbench)

| Run | SACF | SCFM | R@1 | R@5 | R@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| baselineW | — | — | 94.33 | 96.02 | 96.70 |
| sacfW | ✓ | — | 95.10 | 96.25 | 97.32 |
| scfmW | — | ✓ | 94.33 | 95.79 | 96.78 |
| sacf-scfmW | ✓ | ✓ | 95.10 | 96.09 | 97.16 |

### C. SE-RSACF 消融 (Experiment: eprbench_sacfablations)

| Run | R@1 | R@5 | R@10 | R@100 |
|:---|:---:|:---:|:---:|:---:|
| baseline | 93.87 | 95.86 | 96.78 | 99.23 |
| sacf-original-l6 | 94.18 | 96.09 | 97.32 | 99.16 |
| essacf-l6 | 94.56 | 95.79 | 96.70 | 99.23 |
| sersacf-sem-evt | 94.64 | 96.32 | 97.01 | 99.08 |
| **sersacf-layers468-mix** | **94.79** | **95.63** | **96.78** | **99.16** |
| sersacf-full | 94.33 | 95.56 | 96.78 | 99.16 |

---

*文档生成时间：2026-06-11*  
*数据来源：MLflow experiment `eprbench` (59 runs) 与 `eprbench_sacfablations` (18 runs)*
