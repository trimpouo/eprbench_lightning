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
│   ├── model/                # Model / LightningModule configs
│   ├── trainer/              # Trainer configs (cpu, gpu, ddp, ...)
│   ├── callbacks/            # Callback configs (checkpoint, early stopping, ...)
│   ├── logger/               # Logger configs (csv, tensorboard, wandb, ...)
│   ├── paths/                # Path resolution config
│   ├── hydra/                # Hydra runtime config (output dirs, logging)
│   ├── extras/               # Extra utilities config
│   ├── debug/                # Debugging presets
│   ├── experiment/           # Experiment-specific overrides
│   └── hparams_search/       # Hyperparameter search configs (Optuna)
├── src/                      # Source code
│   ├── data/
│   │   ├── eprbench_datamodule.py       # LightningDataModule
│   │   └── components/
│   │       ├── eventvpr_dataset.py      # EventVPRDataset
│   │       └── datasets_ws.py           # Dataset helper
│   ├── models/
│   │   ├── eprbench_module.py           # LightningModule (train/val/test loop)
│   │   └── components/
│   │       ├── network.py               # SGVPRNet (backbone + fusion + aggregation)
│   │       ├── loss.py                  # Loss function (MultiSimilarity + contrastive)
│   │       └── backbone/
│   │           ├── text_encoder_clip.py # OpenCLIP text encoder
│   │           ├── vision_transformer.py
│   │           └── dinov2/              # DINOv2 blocks
│   ├── utils/                # Utilities (logging, instantiators, rich helpers)
│   ├── train.py              # Training entry point
│   └── eval.py               # Evaluation entry point
├── tests/                    # pytest suite
│   ├── conftest.py           # Hydra config fixtures
│   ├── helpers/              # Test helpers (RunIf, package_available)
│   ├── test_train.py
│   ├── test_eval.py
│   ├── test_datamodules.py
│   ├── test_configs.py
│   └── test_sweeps.py
├── model/                    # Pre-trained checkpoints
│   ├── dinov2_vitb14_pretrain.pth
│   └── SGVPR_lightning.ckpt
├── data/                     # Dataset directory (gitignored content)
│   └── EventVPR/
├── logs/                     # Hydra output / training logs
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
# Evaluate a checkpoint
python src/eval.py ckpt_path=model/SGVPR_lightning.ckpt data=eprbench model=eprbench trainer=gpu

# Convenience script
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
- Optimizer: Adam (`lr=1e-4`, `weight_decay=1e-3`).
- Scheduler: StepLR (`gamma=0.5`, step size = 3 epochs in batches).
- Key hyperparameters from official weights:
  - `pretrained_foundation: true`
  - `lambda_contrast: 0.25`
  - `temperature: 0.07`
- `torch.compile` can be enabled via config (`compile: true`) for speed.

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
| Train | `python src/train.py data=eprbench model=eprbench trainer=gpu` |
| Eval | `python src/eval.py ckpt_path=... data=eprbench model=eprbench trainer=gpu` |
| Fast tests | `pytest -k "not slow"` |
| All tests | `pytest` |
| Format / lint | `pre-commit run -a` |
| Clean artifacts | `make clean` |

When modifying code, always:
1. Respect the **99-character line length** (Black).
2. Add or update **docstrings** (interrogate enforces 80% coverage).
3. Run **fast tests** before committing.
4. Update this `AGENTS.md` if you change build steps, entry points, or folder structure.
