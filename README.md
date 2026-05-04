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

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (tested with RTX 3090 24G)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset Preparation

You can download the **EPRBench** dataset from **[Baidu Netdisk](https://pan.baidu.com/s/15DzGOliRku07opgHjE8FXw?pwd=y9ii)**.

Organize the dataset as follows:

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

### Pre-trained Models

Download the DINOv2 (ViT-B/14) pre-trained weights and place it under `model/`:

```bash
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -O model/dinov2_vitb14_pretrain.pth
```

The official SGVPR checkpoint (`model/SGVPR_lightning.ckpt`) is also provided for direct evaluation.

---

## Training

Train the SGVPR model on the EPRBench (EventVPR) dataset:

```bash
# Using Hydra configs
python src/train.py data=eprbench model=eprbench trainer=gpu

# Or use the convenience script
bash train.sh
```

Key hyperparameters (aligned with the original implementation):
- `pretrained_foundation: true` — loads DINOv2 weights
- `lambda_contrast: 0.25` — contrastive learning weight
- `temperature: 0.07` — temperature for CLIP-style loss

---

## Evaluation

Evaluate a trained checkpoint on the test set:

```bash
# Evaluate official checkpoint
python src/eval.py ckpt_path=model/SGVPR_lightning.ckpt trainer=gpu

# Or use the convenience script
bash eval.sh
```

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
│   ├── model/            # Model configs
│   ├── trainer/          # Trainer configs
│   └── callbacks/        # Callback configs
├── src/
│   ├── data/             # Data modules & datasets
│   ├── models/           # Lightning modules & networks
│   ├── utils/            # Utility functions
│   ├── train.py          # Training entry point
│   └── eval.py           # Evaluation entry point
├── model/                # Pre-trained checkpoints
├── data/                 # Dataset directory
└── logs/                 # Training logs
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
