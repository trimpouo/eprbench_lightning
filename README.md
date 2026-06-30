# EPRBench (Lightning + Hydra)

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This is a **PyTorch Lightning + Hydra** reimplementation of the official **[EPRBench](https://arxiv.org/abs/2602.12919)** repository.

We propose **SGVPR** (Semantic-Guided Visual Place Recognition), a novel multi-modal fusion paradigm that leverages LLM-generated scene descriptions to guide spatially attentive token selection and cross-modal feature fusion.

---

## Abstract

Event stream-based Visual Place Recognition (VPR) is an emerging research direction that offers a compelling solution to the instability of conventional visible-light cameras under challenging conditions such as low illumination, overexposure, and high-speed motion. Recognizing the current scarcity of dedicated datasets in this domain, we introduce **EPRBench**, a high-quality benchmark specifically designed for event stream-based VPR. EPRBench comprises 10K event sequences and 65K event frames, collected using both handheld and vehicle-mounted setups to comprehensively capture real-world challenges across diverse viewpoints, weather conditions, and lighting scenarios.

To support semantic-aware and language-integrated VPR research, we provide LLM-generated scene descriptions, subsequently refined through human annotation. Furthermore, we propose a novel multi-modal fusion paradigm for VPR: leveraging LLMs to generate textual scene descriptions from raw event streams, which then guide spatially attentive token selection, cross-modal feature fusion, and multi-scale representation learning.

---

## Recent Updates

- **EventVPR SACF-only ablation**: 30-epoch training with text reaches **R@1 = 94.2%**, outperforming the text baseline (93.6%).
- **Cross-dataset testing**: EventVPR SACF-only transfers to NYC Event with **R@1 = 76.2%** (+2.9% over baseline transfer).
- **NYC-Event-VPR support**: added Event/RGB single-modality and joint-modality DataModules, plus conversion scripts and experiment configs.
- **Project cleanup**: removed template configs (MNIST/Optuna), stale caches, duplicate module files, and outdated docs; merged historical SACF notes into `docs/eventvpr_exploration_journey.md`.

See `AGENTS.md` for the full agent-facing quick-reference and `docs/` for detailed experiment notes.

---

## Quick Start

以下步骤用于快速恢复可训练/可测试的工作状态。

### 1. 环境配置

```bash
# 推荐 Python 3.10
git clone <repo-url>
cd eprbench_lightning

# pip 安装
pip install -r requirements.txt

# 或 conda 安装
conda env create -f environment.yaml
conda activate myenv
```

### 2. 下载预训练权重

```bash
# DINOv2 ViT-B/14（必需）
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
  -O model/dinov2_vitb14_pretrain.pth

# 官方 SGVPR 原始 state-dict（已随仓库提供为 model/SGVPR.pth，用于 notebook / visualize_server）
```

### 3. 下载 EventVPR 数据集

EPRBench (EventVPR) 数据集可从 **[百度网盘](https://pan.baidu.com/s/15DzGOliRku07opgHjE8FXw?pwd=y9ii)** 下载。

解压后按如下结构放置：

```
data/EventVPR/
├── train/
│   ├── database/
│   └── queries/
├── val/
│   ├── database/
│   └── queries/
├── test/
│   ├── database/
│   └── queries/
└── scene_descriptions/  # LLM-generated text descriptions (.txt files)
```

### 4.（可选）下载 NYC-Event-VPR 数据集

NYC-Event-VPR 官方发布在：

- Hugging Face: https://huggingface.co/datasets/ai4ce/NYC-Event-VPR
- GitHub: https://github.com/ai4ce/NYC-Event-VPR

下载 **NYC-Event-VPR_VG** 格式（兼容 Visual Geo-localization 框架）后，使用本仓库脚本转换为 EPRBench 风格：

```bash
python scripts/convert_nyc_vg_to_eprbench.py \
  --src data/NYC-Event-VPR-VG \
  --dst data/NYC-Event-VPR-converted \
  --modalities Event RGB \
  --threshold 25.0
```

转换后的目录可直接用于 `data=nyc_vpr_event`、`data=nyc_vpr_rgb` 或 `data=nyc_vpr_joint`。

### 5. 快速验证

```bash
# 训练 EventVPR baseline（单卡 GPU，30 epochs）
python src/train.py data=eprbench model=eprbench trainer=gpu \
  data.use_text=true model.use_text=true model.lambda_contrast=0.25 \
  data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30

# 训练 EventVPR SACF-only（batch 需降至 12 避免 OOM）
python src/train.py experiment=eprbench_sacf_only trainer=gpu \
  data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30

# 评估训练好的 Lightning checkpoint
python src/eval.py \
  ckpt_path=logs/train/runs/<run>/checkpoints/epoch_029.ckpt \
  data=eprbench model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  data.infer_batch_size=32 data.num_workers=2
```

> **提示**：
> - 本机稳定设置为 `data.num_workers=2`，默认 `6` 可能卡死。
> - SACF 模型评估时必须加 `model.net.mods.sacf.enabled=true`。
> - `model/SGVPR.pth` 是官方原始权重，不是 Lightning checkpoint；Hydra 评估请使用训练生成的 `*.ckpt`。

---

## Training

### EventVPR

Train the SGVPR model on the EPRBench (EventVPR) dataset:

```bash
# Baseline with text
python src/train.py data=eprbench model=eprbench trainer=gpu \
  data.use_text=true model.use_text=true model.lambda_contrast=0.25 \
  data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30

# SACF-only with text (use batch 12 to avoid OOM on 16 GiB GPUs)
python src/train.py experiment=eprbench_sacf_only trainer=gpu \
  data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30
```

### NYC-Event-VPR Event + RGB Joint Training

```bash
python src/train.py \
  data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sacf_scfm_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5
```

Key hyperparameters (aligned with the original implementation):
- `pretrained_foundation: true` — loads DINOv2 weights
- `lambda_contrast: 0.15` — contrastive learning weight (γ in the paper)
- `temperature: 0.07` — temperature for CLIP-style loss
- `num_workers=2` is the stable setting on this machine; the default `6` can hang.

> **Implementation note:** The token similarity matrix in `network.py` uses `-1e4` for masked fill instead of the original `-1e9`. This is an intentional compromise to avoid `inf` overflows when training with `precision=16-mixed`. If you need exact numerical alignment with the original fp32 training, switch to the fp32 trainer (`trainer=fp32`).

---

## Evaluation

Evaluate a trained **Lightning** checkpoint on the test set:

```bash
# EventVPR test
python src/eval.py \
  ckpt_path=logs/train/runs/<run>/checkpoints/epoch_029.ckpt \
  data=eprbench model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  data.infer_batch_size=32 data.num_workers=2

# Cross-dataset: EventVPR model on NYC Event test
python src/eval.py \
  ckpt_path=logs/train/runs/<run>/checkpoints/epoch_029.ckpt \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  model.net.mods.sacf.enabled=true \
  data.infer_batch_size=32 data.num_workers=2

# Or use the convenience script (edit CKPT_PATH first)
bash eval.sh
```

> **Note:** If the checkpoint was trained with SACF, you must add `model.net.mods.sacf.enabled=true` at eval time, otherwise the architecture mismatch will produce wrong results.

### Latest Results

| Experiment | Test set | R@1 | R@5 | R@10 | R@100 |
|---|---|:---:|:---:|:---:|:---:|
| EventVPR baseline + text | EventVPR | 93.6 | 95.6 | 96.6 | 99.2 |
| EventVPR SACF-only + text | EventVPR | 94.2 | 95.4 | 97.2 | 99.2 |
| EventVPR baseline → NYC | NYC Event | 73.3 | 87.5 | 91.2 | 97.6 |
| EventVPR SACF-only → NYC | NYC Event | 76.2 | 88.9 | 91.7 | 98.1 |
| NYC SACF+SCFM joint | NYC Event | 92.8 | 97.7 | 98.5 | 99.7 |
| NYC SACF+SCFM joint | NYC RGB | 99.2 | 99.8 | 99.9 | 99.9 |

Expected test metrics (official weights):
| R@1 | R@5 | R@10 | R@100 |
|-----|-----|------|-------|
| 94.0 | 96.0 | 96.9 | 99.2 |

---

## Project Structure

This project is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

```
.
├── configs/              # Hydra configuration files
│   ├── data/             # Data module configs
│   │   ├── eprbench.yaml
│   │   ├── nyc_vpr_event.yaml
│   │   ├── nyc_vpr_rgb.yaml
│   │   └── nyc_vpr_joint.yaml
│   ├── experiment/       # Experiment-specific overrides
│   │   ├── eprbench_sacf_only.yaml
│   │   ├── nyc_vpr_visual_only.yaml
│   │   ├── nyc_vpr_sac_only_joint.yaml
│   │   ├── nyc_vpr_scfm_only_joint.yaml
│   │   └── nyc_vpr_sacf_scfm_joint.yaml
│   ├── model/            # Model configs
│   ├── trainer/          # Trainer configs
│   └── callbacks/        # Callback configs
├── src/
│   ├── data/             # Data modules & datasets
│   │   ├── eprbench_datamodule.py
│   │   ├── nyc_vpr_datamodule.py
│   │   ├── nyc_vpr_joint_datamodule.py
│   │   └── components/
│   │       ├── eventvpr_dataset.py
│   │       └── nyc_vpr_bench_dataset.py
│   ├── models/           # Lightning modules & networks
│   │   ├── eprbench_module.py
│   │   └── components/
│   │       ├── network.py
│   │       └── mods/     # SACF, SCFM, etc.
│   ├── utils/            # Utility functions
│   ├── train.py          # Training entry point
│   ├── eval.py           # Evaluation entry point
│   └── visualize_server.py
├── scripts/              # Utility & ablation scripts
├── model/                # Pre-trained checkpoints
│   ├── dinov2_vitb14_pretrain.pth
│   └── SGVPR.pth
├── data/                 # Dataset directory
├── logs/                 # Training logs
└── docs/                 # Experiment notes
    ├── eventvpr_exploration_journey.md
    ├── nyc_vpr_sacf_scfm_ablation.md
    └── baseline_sacf_pipeline.md
```

---

## Citation

If you find this repo useful for your research, please consider citing the original paper:

```bibtex
@misc{wang2026eprbench,
      title={EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition},
      author={Xiao Wang and Xingxing Xiong and Jinfeng Gao and Xufeng Lou and Bo Jiang and Si-bao Chen and Yaowei Wang and Yonghong Tian},
      year={2026},
      eprint={2602.12919},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.12919},
}
```

## Acknowledgements

Parts of this repo are inspired by [CricaVPR](https://github.com/Lu-Feng/CricaVPR) and [DINOv2](https://github.com/facebookresearch/dinov2).
The Lightning + Hydra template is from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
