#!/bin/bash
# Ablation study for new event-specific modules
# Run from root folder with: bash scripts/ablation_new_modules.sh
#
# Recommended: run inside tmux/screen so it continues after disconnect.
#   tmux new -s ablation
#   bash scripts/ablation_new_modules.sh

set -e

# Common logger config
LOG="logger=mlflow"
EPOCHS=""  # leave empty to use config default; set e.g. trainer.max_epochs=5 if needed

# ---------------------------------------------------------------------------
# 0. Baseline (original SG-VPR, all new switches OFF)
# ---------------------------------------------------------------------------
echo "=== [0/12] Baseline ==="
python src/train.py $LOG logger.mlflow.run_name=baseline $EPOCHS

# ---------------------------------------------------------------------------
# 1. Single new modules
# ---------------------------------------------------------------------------

# 1a. Event-Aware Token Selector (EATS) only
echo "=== [1/12] EATS only ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    $LOG logger.mlflow.run_name=eats-only $EPOCHS

# 1b. Event-Stability SACF (ES-SACF) only
echo "=== [2/12] ES-SACF only ==="
python src/train.py \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    $LOG logger.mlflow.run_name=essacf-only $EPOCHS

# 1c. Event-Reliable SCFM (ER-SCFM) only
echo "=== [3/12] ER-SCFM only ==="
python src/train.py \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    $LOG logger.mlflow.run_name=erscfm-only $EPOCHS

# ---------------------------------------------------------------------------
# 2. Pairwise combinations
# ---------------------------------------------------------------------------

# 2a. EATS + ES-SACF
echo "=== [4/12] EATS + ES-SACF ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    $LOG logger.mlflow.run_name=eats-essacf $EPOCHS

# 2b. EATS + ER-SCFM
echo "=== [5/12] EATS + ER-SCFM ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    $LOG logger.mlflow.run_name=eats-erscfm $EPOCHS

# 2c. ES-SACF + ER-SCFM
echo "=== [6/12] ES-SACF + ER-SCFM ==="
python src/train.py \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    $LOG logger.mlflow.run_name=essacf-erscfm $EPOCHS

# ---------------------------------------------------------------------------
# 3. Triple combination (core event-specific design)
# ---------------------------------------------------------------------------

echo "=== [7/12] EATS + ES-SACF + ER-SCFM ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    $LOG logger.mlflow.run_name=eats-essacf-erscfm $EPOCHS

# ---------------------------------------------------------------------------
# 4. Add Temporal Consistency Loss (descriptor-level, no network change)
# ---------------------------------------------------------------------------

echo "=== [8/12] Core + Temporal Consistency Loss ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    model.use_temporal_consistency=True \
    model.temporal_consistency_weight=0.05 \
    $LOG logger.mlflow.run_name=core-temporal $EPOCHS

# ---------------------------------------------------------------------------
# 5. Add Text Dropout for robustness
# ---------------------------------------------------------------------------

echo "=== [9/12] Core + Text Dropout ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    model.text_dropout_prob=0.15 \
    model.word_dropout_prob=0.10 \
    $LOG logger.mlflow.run_name=core-textdrop $EPOCHS

# ---------------------------------------------------------------------------
# 6. Full stack: core + temporal + text dropout
# ---------------------------------------------------------------------------

echo "=== [10/12] Core + Temporal + Text Dropout ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.sacf.use_event_stability=True \
    model.net.mods.scfm.enabled=True \
    model.net.mods.scfm.use_event_reliable=True \
    model.use_temporal_consistency=True \
    model.temporal_consistency_weight=0.05 \
    model.text_dropout_prob=0.15 \
    model.word_dropout_prob=0.10 \
    $LOG logger.mlflow.run_name=core-temporal-textdrop $EPOCHS

# ---------------------------------------------------------------------------
# 7. Legacy comparison: original SACF + SCFM (no event cues)
# ---------------------------------------------------------------------------

echo "=== [11/12] Original SACF + SCFM ==="
python src/train.py \
    model.net.mods.sacf.enabled=True \
    model.net.mods.scfm.enabled=True \
    $LOG logger.mlflow.run_name=orig-sacfm $EPOCHS

# ---------------------------------------------------------------------------
# 8. Best candidate vs. original SACF+SCFM with EATS added
# ---------------------------------------------------------------------------

echo "=== [12/12] Original SACF+SCFM + EATS ==="
python src/train.py \
    model.net.mods.event_cues.enabled=True \
    model.net.mods.event_aware_token_selector.enabled=True \
    model.net.mods.sacf.enabled=True \
    model.net.mods.scfm.enabled=True \
    $LOG logger.mlflow.run_name=eats-orig-sacfm $EPOCHS

echo "=== All ablation runs finished ==="
