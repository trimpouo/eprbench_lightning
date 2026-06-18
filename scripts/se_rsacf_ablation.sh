#!/bin/bash
# Ablation study for SE-RSACF additions.
# Run from the project root with:
#   bash scripts/se_rsacf_ablation.sh
#
# Optional environment variables:
#   EPOCHS="trainer.max_epochs=5" bash scripts/se_rsacf_ablation.sh
#   LOGGER="logger=tensorboard" RUN_NAME_KEY="logger.tensorboard.name" \
#       bash scripts/se_rsacf_ablation.sh

set -euo pipefail

LOGGER=${LOGGER:-"logger=mlflow"}
EPOCHS=${EPOCHS:-""}
RUN_NAME_KEY=${RUN_NAME_KEY:-"logger.mlflow.run_name"}

COMMON=(
    "$LOGGER"
    $EPOCHS
)

SE_RSACF_BASE=(
    "model.net.mods.sacf.enabled=True"
    "model.net.mods.sacf.variant=se_rsacf"
    "model.net.mods.sacf.layer_idx=6"
    "model.net.mods.sacf.local_window_size=3"
    "model.net.mods.sacf.local_dilations=[1]"
    "model.net.mods.sacf.tau=0.10"
    "model.net.mods.sacf.lambda_sem=0.30"
    "model.net.mods.sacf.lambda_evt=0.20"
    "model.net.mods.sacf.residual_scale_init=0.10"
    "model.net.mods.sacf.zero_init_out_proj=True"
    "model.sacf_residual_loss_weight=0.001"
    "model.sacf_noise_loss_weight=0.001"
)

run_train() {
    local run_name=$1
    shift

    echo "=== ${run_name} ==="
    python src/train.py \
        "${COMMON[@]}" \
        "${RUN_NAME_KEY}=${run_name}" \
        "$@"
}

# ---------------------------------------------------------------------------
# 0. Baselines and legacy comparisons
# ---------------------------------------------------------------------------
run_train "baseline" \
    "model.net.mods.sacf.enabled=False"

run_train "sacf-original-l6" \
    "model.net.mods.sacf.enabled=True" \
    "model.net.mods.sacf.variant=original" \
    "model.net.mods.sacf.layer_idx=6"

run_train "essacf-l6" \
    "model.net.mods.sacf.enabled=True" \
    "model.net.mods.sacf.variant=event_stability" \
    "model.net.mods.sacf.layer_idx=6" \
    "model.net.mods.event_cues.enabled=True"

# ---------------------------------------------------------------------------
# 1. SE-RSACF prior ablations
# ---------------------------------------------------------------------------
run_train "sersacf-no-sem-no-evt" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.sacf.use_semantic_prior=False" \
    "model.net.mods.sacf.use_event_prior=False"

run_train "sersacf-sem-only" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.sacf.use_semantic_prior=True" \
    "model.net.mods.sacf.use_event_prior=False"

run_train "sersacf-evt-only" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.sacf.use_semantic_prior=False" \
    "model.net.mods.sacf.use_event_prior=True" \
    "model.net.mods.event_cues.enabled=True"

run_train "sersacf-sem-evt" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.sacf.use_semantic_prior=True" \
    "model.net.mods.sacf.use_event_prior=True" \
    "model.net.mods.event_cues.enabled=True"

# ---------------------------------------------------------------------------
# 2. Gate ablations
# ---------------------------------------------------------------------------
run_train "sersacf-conv-gate" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.gate_type=conv"

run_train "sersacf-fastkan-gate" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.gate_type=fastkan" \
    "model.net.mods.sacf.gate_hidden=16" \
    "model.net.mods.sacf.gate_num_grids=8"

run_train "sersacf-fastkan-hidden32" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.gate_type=fastkan" \
    "model.net.mods.sacf.gate_hidden=32" \
    "model.net.mods.sacf.gate_num_grids=8"

# ---------------------------------------------------------------------------
# 3. Low-layer and multi-scale ablations
# ---------------------------------------------------------------------------
run_train "sersacf-layer4" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.layer_idx=4"

run_train "sersacf-layer8" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.layer_idx=8"

run_train "sersacf-layers468-mix" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.layer_indices=[4,6,8]" \
    "model.net.mods.sacf.use_layer_mixture=True"

run_train "sersacf-layers468-textgate" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.layer_indices=[4,6,8]" \
    "model.net.mods.sacf.use_layer_mixture=True" \
    "model.net.mods.sacf.use_text_layer_gate=True"

run_train "sersacf-dilated13" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.local_dilations=[1,3]"

# ---------------------------------------------------------------------------
# 4. Top-k and regularization ablations
# ---------------------------------------------------------------------------
run_train "sersacf-adaptive-topk" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.adaptive_text_topk.enabled=True" \
    "model.net.mods.adaptive_text_topk.rho_min=0.20" \
    "model.net.mods.adaptive_text_topk.rho_max=0.30" \
    "model.net.mods.adaptive_text_topk.rho_base=0.25"

run_train "sersacf-no-aux-reg" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.sacf_residual_loss_weight=0.0" \
    "model.sacf_noise_loss_weight=0.0"

run_train "sersacf-full" \
    "${SE_RSACF_BASE[@]}" \
    "model.net.mods.event_cues.enabled=True" \
    "model.net.mods.sacf.gate_type=fastkan" \
    "model.net.mods.sacf.gate_hidden=16" \
    "model.net.mods.sacf.gate_num_grids=8" \
    "model.net.mods.sacf.layer_indices=[4,6,8]" \
    "model.net.mods.sacf.use_layer_mixture=True" \
    "model.net.mods.sacf.use_text_layer_gate=True" \
    "model.net.mods.sacf.local_dilations=[1,3]" \
    "model.net.mods.adaptive_text_topk.enabled=True"

echo "=== All SE-RSACF ablation runs finished ==="
