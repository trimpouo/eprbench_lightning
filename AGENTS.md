# AGENTS.md — EPRBench (Lightning + Hydra)

> This file is intended for AI coding agents. It describes the project structure, conventions, and workflows so you can be productive without prior context.

---

## Project Overview

This is a **PyTorch Lightning + Hydra** reimplementation of the official **[EPRBench](https://arxiv.org/abs/2602.12919)** repository. It implements **SGVPR** (Semantic-Guided Visual Place Recognition), a multi-modal fusion paradigm that leverages LLM-generated scene descriptions to guide spatially attentive token selection and cross-modal feature fusion for event stream-based Visual Place Recognition.

The project is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

### Key Technologies
- **Python** 3.8+ (tested on 3.8, 3.9, 3.10; conda env specifies 3.10)
- **PyTorch** 2.0+ and **PyTorch Lightning** 2.0+
- **Hydra** 1.3 for hierarchical YAML configuration
- **DINOv2** ViT-B/14 as the frozen visual backbone (with trainable adapters)
- **OpenCLIP** for text encoding
- **FAISS** for nearest-neighbor retrieval during evaluation
- **pytorch-metric-learning** for MultiSimilarityLoss and online mining

---

## Repository Structure

```
.
├── configs/                  # Hydra configuration files (YAML)
│   ├── data/                 # DataModule configs
│   │   ├── eprbench.yaml
│   │   ├── nyc_vpr_event.yaml
│   │   ├── nyc_vpr_rgb.yaml
│   │   └── nyc_vpr_joint.yaml
│   ├── model/                # Model / LightningModule configs
│   ├── trainer/              # Trainer configs (cpu, gpu, ddp, ...)
│   ├── callbacks/            # Callback configs (checkpoint, early stopping, ...)
│   ├── logger/               # Logger configs (csv, tensorboard, wandb, ...)
│   ├── paths/                # Path resolution config
│   ├── hydra/                # Hydra runtime config (output dirs, logging)
│   ├── extras/               # Extra utilities config
│   ├── debug/                # Debugging presets
│   └── experiment/           # Experiment-specific overrides
│       ├── eprbench_sacf_only.yaml
│       ├── nyc_vpr_visual_only.yaml
│       ├── nyc_vpr_sac_only_joint.yaml
│       ├── nyc_vpr_scfm_only_joint.yaml
│       └── nyc_vpr_sacf_scfm_joint.yaml
├── src/                      # Source code
│   ├── data/
│   │   ├── eprbench_datamodule.py       # LightningDataModule
│   │   ├── nyc_vpr_datamodule.py        # NYC Event/RGB single-modality DataModule
│   │   ├── nyc_vpr_joint_datamodule.py  # NYC Event + RGB joint DataModule
│   │   └── components/
│   │       ├── eventvpr_dataset.py      # EventVPRDataset
│   │       ├── nyc_vpr_bench_dataset.py # NYC-VPR-Bench dataset
│   │       └── datasets_ws.py           # Dataset helper
│   ├── models/
│   │   ├── eprbench_module.py           # LightningModule (train/val/test loop)
│   │   └── components/
│   │       ├── network.py               # SGVPRNet (backbone + fusion + aggregation)
│   │       ├── loss.py                  # Loss function (MultiSimilarity + contrastive)
│   │       ├── mods/                    # Optional enhancement modules (SACF, SCFM, ...)
│   │       └── backbone/
│   │           ├── text_encoder_clip.py # OpenCLIP text encoder
│   │           ├── vision_transformer.py
│   │           └── dinov2/              # DINOv2 blocks
│   ├── utils/                # Utilities (logging, instantiators, rich helpers)
│   ├── train.py              # Training entry point
│   ├── eval.py               # Evaluation entry point
│   └── visualize_server.py   # Visual inference server (uses model/SGVPR.pth)
├── tests/                    # pytest suite
│   ├── conftest.py           # Hydra config fixtures
│   ├── helpers/              # Test helpers (RunIf, package_available)
│   ├── test_train.py
│   ├── test_eval.py
│   ├── test_datamodules.py
│   └── test_configs.py
├── model/                    # Pre-trained checkpoints
│   ├── dinov2_vitb14_pretrain.pth
│   └── SGVPR.pth             # official/original SGVPR weights (dict format)
├── data/                     # Dataset directory (gitignored content)
│   ├── EventVPR/
│   └── NYC-Event-VPR-converted/  # optional, for NYC experiments
├── logs/                     # Hydra output / training logs
├── docs/                     # Experiment notes and method docs
│   ├── eventvpr_exploration_journey.md
│   ├── nyc_vpr_sacf_scfm_ablation.md
│   └── baseline_sacf_pipeline.md
├── scripts/                  # Utility scripts
│   ├── convert_nyc_vg_to_eprbench.py
│   ├── extract_nyc_results.py
│   ├── ablation_scfm_sacf.sh
│   ├── sacf_sweep.sh
│   ├── schedule.sh
│   ├── se_rsacf_ablation.sh
│   └── eval_schedule_runs.sh
├── requirements.txt
├── environment.yaml          # Conda environment spec
├── pyproject.toml            # pytest and coverage settings
├── setup.py                  # Package setup (install_requires: lightning, hydra-core)
├── Makefile                  # Convenience commands
├── train.sh                  # Quick training script
├── eval.sh                   # Quick evaluation script
└── .pre-commit-config.yaml   # Pre-commit hooks
```

---

## Build and Test Commands

### Environment Setup
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yaml
conda activate myenv
```

### Training
```bash
# Hydra-based training (recommended)
python src/train.py data=eprbench model=eprbench trainer=gpu

# Convenience script
bash train.sh
```

### Evaluation
```bash
# Evaluate a Lightning training checkpoint (replace with your actual checkpoint path)
python src/eval.py ckpt_path=logs/train/runs/.../checkpoints/last.ckpt data=eprbench model=eprbench trainer=gpu

# Convenience script (expects you to set the checkpoint path inside)
bash eval.sh
```

### Testing
```bash
# Run only fast tests (exclude @pytest.mark.slow)
pytest -k "not slow"

# Run all tests
pytest

# Using Makefile
make test        # fast tests
make test-full   # all tests
```

### Code Quality
```bash
# Run all pre-commit hooks on all files
pre-commit run -a

# Using Makefile
make format
```

### Cleaning
```bash
make clean       # Remove pycache, pytest cache, coverage, etc.
make clean-logs  # Remove logs/**
```

---

## Current Progress & How to Resume

This section records the latest working state so you can quickly return to it.

### Datasets

- **EventVPR** (`data/EventVPR/`) — primary benchmark; supports text branch.
  - Download: [Baidu Netdisk](https://pan.baidu.com/s/15DzGOliRku07opgHjE8FXw?pwd=y9ii)
- **NYC-Event-VPR** (`data/NYC-Event-VPR-converted/`) — optional; supports Event, RGB, and Event+RGB joint training.
  - Download the `NYC-Event-VPR_VG` split from [Hugging Face](https://huggingface.co/datasets/ai4ce/NYC-Event-VPR) or [GitHub](https://github.com/ai4ce/NYC-Event-VPR), then run `scripts/convert_nyc_vg_to_eprbench.py`.

### Reproduced Checkpoints

EventVPR 30-epoch checkpoints were produced by the commands below. The latest run directories are under `logs/train/runs/`.

| Experiment | Config | Key result |
|---|---|---|
| EventVPR baseline + text | `model=eprbench` + text | R@1/5/10/100 = 93.6/95.6/96.6/99.2 |
| EventVPR SACF-only + text | `experiment=eprbench_sacf_only` | R@1/5/10/100 = 94.2/95.4/97.2/99.2 |
| Cross-dataset EventVPR → NYC Event | `data=nyc_vpr_event` | baseline 73.3/87.5/91.2/97.6; SACF 76.2/88.9/91.7/98.1 |

### One-line Training / Eval Recipes

```bash
# 1. EventVPR baseline with text (30 epochs, batch 18)
python src/train.py data=eprbench model=eprbench trainer=gpu \
  data.use_text=true model.use_text=true model.lambda_contrast=0.25 \
  data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30

# 2. EventVPR SACF-only with text (30 epochs, batch 12 due to OOM)
python src/train.py experiment=eprbench_sacf_only trainer=gpu \
  data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30

# 3. Evaluate a trained EventVPR checkpoint (enable SACF if the model was trained with it)
python src/eval.py \
  ckpt_path=logs/train/runs/<run>/checkpoints/epoch_029.ckpt \
  data=eprbench model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  data.infer_batch_size=32 data.num_workers=2

# 4. Cross-dataset eval: EventVPR → NYC Event
python src/eval.py \
  ckpt_path=logs/train/runs/<run>/checkpoints/epoch_029.ckpt \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  model.net.mods.sacf.enabled=true \
  data.infer_batch_size=32 data.num_workers=2

# 5. NYC Event + RGB joint training with SACF+SCFM (visual-only)
python src/train.py \
  data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sacf_scfm_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5

# 6. NYC joint ablations
python src/train.py data=nyc_vpr_joint trainer=gpu \
  experiment=nyc_vpr_sac_only_joint trainer.max_epochs=30
python src/train.py data=nyc_vpr_joint trainer=gpu \
  experiment=nyc_vpr_scfm_only_joint trainer.max_epochs=30
```

### Important Gotchas

- **`num_workers=2`** is the stable setting on this machine; default `6` may hang.
- **SACF checkpoints must be evaluated with SACF enabled**: add `model.net.mods.sacf.enabled=true` to the eval command.
- **Text is disabled at cross-dataset eval**: set `data.use_text=false model.use_text=false`.
- **`model/SGVPR.pth`** is the original official state-dict, not a Lightning checkpoint. Use it with `notebooks/visual_inference.ipynb` or `src/visualize_server.py`; for Hydra `src/eval.py` use a training checkpoint (`*.ckpt`).
- **Logs are cleaned**: only `logs/.gitkeep` remains; previous 48 GB of Hydra outputs were removed.

### Latest Experiment Docs

- `docs/eventvpr_exploration_journey.md` — EventVPR SACF ablation & cross-dataset testing, plus historical SE-RSACF recipes.
- `docs/nyc_vpr_sacf_scfm_ablation.md` — NYC Event+RGB joint training and SACF/SCFM ablation.
- `docs/baseline_sacf_pipeline.md` — Baseline + SACF workflow explanation.

---

## Code Style Guidelines

The project enforces strict code quality via **pre-commit** hooks. Do not bypass these hooks.

- **Formatter**: [Black](https://github.com/psf/black) with `--line-length 99`
- **Import sorting**: [isort](https://pycqa.github.io/isort/) with `--profile black`
- **Linter**: [flake8](https://flake8.pycqa.org) with ignores `E203,E402,E501,F401,F841,RST2,RST301`
- **Docstrings**: [docformatter](https://github.com/PyCQA/docformatter) (sphinx style, wrap at 99)
- **Docstring coverage**: [interrogate](https://interrogate.readthedocs.io) (fail-under 80%)
- **Security**: [bandit](https://bandit.readthedocs.io) (skip `B101`)
- **YAML**: prettier
- **Shell**: shellcheck
- **Markdown**: mdformat
- **Spelling**: codespell

### Python Version Target
- `pyupgrade` enforces Python 3.8+ syntax.

### Important Conventions
- Use type hints where practical.
- Follow the existing Google/Spinx-style docstrings (`:param`, `:return:`).
- Do not commit large files; `check-added-large-files` is enabled.
- Avoid trailing whitespace and missing EOF newlines.

---

## Testing Instructions

### Test Organization
- **Fast tests** (no `slow` marker): run on CPU, single step/epoch, no GPU required.
- **Slow tests** (`@pytest.mark.slow`): train 1+ epochs, DDP simulation, resume logic.
- **GPU tests** (`@RunIf(min_gpus=1)`): skipped automatically if no CUDA GPUs are available.

### Key Fixtures (`tests/conftest.py`)
- `cfg_train_global` / `cfg_eval_global`: package-scoped Hydra configs with safe defaults (CPU, 1 epoch, no pre-trained foundation model, no loggers).
- `cfg_train` / `cfg_eval`: function-scoped copies that redirect `output_dir` and `log_dir` to `tmp_path`, and clear `GlobalHydra` after each test.

### Writing New Tests
- Use the `cfg_train` or `cfg_eval` fixtures to get a ready-to-use Hydra `DictConfig`.
- Mark long-running tests with `@pytest.mark.slow`.
- Use `@RunIf(min_gpus=1)` for GPU-dependent tests.
- When modifying source paths, update `pytest --cov src` in CI (`.github/workflows/test.yml`) if the folder name changes.

### Running Tests with Coverage
```bash
pytest --cov src
```

---

## Architecture and Runtime Details

### Configuration System (Hydra)
- Entry configs: `configs/train.yaml` and `configs/eval.yaml`.
- They compose sub-configs via `defaults:` lists (e.g., `data: eprbench`, `model: eprbench`, `trainer: gpu`).
- Override any value from the command line: `python src/train.py model.lambda_contrast=0.25`.
- Output directories are dynamically generated by Hydra under `logs/<task_name>/runs/<timestamp>/`.
- `rootutils` sets `PROJECT_ROOT` and adds the repo root to `PYTHONPATH` so imports like `from src import utils` work without installing the package.

### Data Pipeline
- **DataModule**: `EPRBenchDataModule` (`src/data/eprbench_datamodule.py`)
- **Dataset**: `EventVPRDataset` (`src/data/components/eventvpr_dataset.py`)
- Dataset layout expected under `data/EventVPR/`:
  ```
  EventVPR/
  ├── train/
  │   ├── database/
  │   └── queries/
  ├── val/          # (optional; original code uses test for validation)
  ├── test/
  │   ├── database/
  │   └── queries/
  └── scene_descriptions/   # .txt files with LLM-generated descriptions
  ```
- Training samples `img_per_place` images per place (default 5).
- Evaluation returns all database and query images; retrieval uses FAISS.

### Model Architecture
- **LightningModule**: `EPRBenchLitModule` (`src/models/eprbench_module.py`)
- **Network**: `SGVPRNet` (`src/models/components/network.py`)
  - Backbone: DINOv2 ViT-B/14 (patch 14, img 518). Loaded from `model/dinov2_vitb14_pretrain.pth`.
  - Backbone is **frozen** except for parameters with `"adapter"` in their name.
  - Adapter weights are initialized to near-zero (constant 0.0 for D_fc2, 1e-5 for conv).
  - Aggregation: L2Norm → GeM → Flatten.
  - Multi-scale patch aggregation into 14 tokens, then a 2-layer TransformerEncoder.
  - Text conditioning via OpenCLIP (`ViT-B-16`, `laion2b_s34b_b88k`).
  - Cross-modal fusion: similarity-based token selection, top-25% patch retention, feature enhancement.
- **Loss**: `MultiSimilarityLoss` + optional image-to-text / text-to-image contrastive loss (weighted by `lambda_contrast`, temperature `0.07`).

### Evaluation Metrics
- Recall@N (R@1, R@5, R@10, R@100) computed with FAISS `IndexFlatL2`.
- Validation and test use the same split (`test`) to mirror the original implementation.

### Training Notes
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-3`).
- Scheduler: StepLR (`gamma=0.5`, step size = 3 epochs in batches).
- Key hyperparameters from official weights:
  - `pretrained_foundation: true`
  - `lambda_contrast: 0.15`
  - `temperature: 0.07`
- `torch.compile` can be enabled via config (`compile: true`) for speed.
- On this machine, use `data.num_workers=2` for stable data loading (the default `6` can hang).
- SACF-only training needs a smaller batch size than the baseline (e.g. `data.train_batch_size=12`) to avoid OOM on a 16 GiB GPU.

---

## Security Considerations

- **Pre-commit hooks** include `detect-private-key` and `bandit` (Python security linter).
- Do not commit credentials, API keys, or checkpoint files unrelated to the public release.
- The `.env` file pattern is supported via `rootutils` / `python-dotenv` conventions, but `.env` itself is gitignored. Use `.env.example` as a template.
- `data/` and `logs/` are excluded from linting and should not contain sensitive metadata.

---

## CI/CD

GitHub Actions workflows (`.github/workflows/`):
- **Tests** (`test.yml`): runs on Ubuntu/macOS/Windows with Python 3.8/3.9/3.10; uploads coverage to Codecov.
- **Code Quality** (`code-quality-main.yaml`, `code-quality-pr.yaml`): runs pre-commit hooks.
- **Release Drafter** (`release-drafter.yml`): auto-generates release notes.

---

## Quick Reference for Agents

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Train EventVPR baseline | `python src/train.py data=eprbench model=eprbench trainer=gpu` |
| Train EventVPR SACF-only | `python src/train.py experiment=eprbench_sacf_only trainer=gpu data.train_batch_size=12 data.num_workers=2` |
| Eval a checkpoint | `python src/eval.py ckpt_path=... data=eprbench model=eprbench trainer=gpu` |
| Cross-dataset eval | `python src/eval.py ckpt_path=... data=nyc_vpr_event model=eprbench trainer=gpu model.net.mods.sacf.enabled=true data.use_text=false model.use_text=false` |
| Train NYC joint | `python src/train.py data=nyc_vpr_joint experiment=nyc_vpr_sacf_scfm_joint trainer=gpu` |
| Fast tests | `pytest -k "not slow"` |
| All tests | `pytest` |
| Format / lint | `pre-commit run -a` |
| Clean artifacts | `make clean` |
| Clean logs | `make clean-logs` |

When modifying code, always:
1. Respect the **99-character line length** (Black).
2. Add or update **docstrings** (interrogate enforces 80% coverage).
3. Run **fast tests** before committing.
4. Update this `AGENTS.md` if you change build steps, entry points, or folder structure.
