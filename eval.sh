#!/bin/bash
# Evaluate a trained Lightning checkpoint.
# Replace CKPT_PATH with the path to your actual .ckpt file before running.

CKPT_PATH="logs/train/runs/<your_run>/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    ckpt_path="${CKPT_PATH}" \
    data=eprbench \
    model=eprbench \
    trainer=gpu
