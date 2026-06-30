#!/bin/bash
# Evaluate the four checkpoints produced by scripts/schedule.sh

export HF_HUB_OFFLINE=1
export http_proxy=
export https_proxy=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR=/home/featurize/eprbench_lightning

echo "=== baselineW ==="
python "$ROOT_DIR/src/eval.py" \
    ckpt_path="$ROOT_DIR/logs/train/runs/2026-06-22_05-17-52/checkpoints/last.ckpt" \
    data=eprbench model=eprbench trainer=gpu \
    data.infer_batch_size=8 data.pin_memory=false

echo "=== sacfW ==="
python "$ROOT_DIR/src/eval.py" \
    ckpt_path="$ROOT_DIR/logs/train/runs/2026-06-22_06-47-14/checkpoints/last.ckpt" \
    data=eprbench model=eprbench trainer=gpu \
    data.infer_batch_size=8 data.pin_memory=false \
    model.net.mods.sacf.enabled=True

echo "=== scfmW ==="
python "$ROOT_DIR/src/eval.py" \
    ckpt_path="$ROOT_DIR/logs/train/runs/2026-06-22_09-19-38/checkpoints/last.ckpt" \
    data=eprbench model=eprbench trainer=gpu \
    data.infer_batch_size=8 data.pin_memory=false \
    model.net.mods.scfm.enabled=True

echo "=== sacf-scfmW ==="
python "$ROOT_DIR/src/eval.py" \
    ckpt_path="$ROOT_DIR/logs/train/runs/2026-06-22_10-57-12/checkpoints/last.ckpt" \
    data=eprbench model=eprbench trainer=gpu \
    data.infer_batch_size=8 data.pin_memory=false \
    model.net.mods.sacf.enabled=True model.net.mods.scfm.enabled=True
