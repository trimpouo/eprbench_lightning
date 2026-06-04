#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py trainer.max_epochs=1 logger=tensorboard logger.tensorboard.name=test
python src/train.py logger=mlflow logger.mlflow.run_name=baseline
python src/train.py model.net.mods.sacf.enabled=True logger=mlflow logger.mlflow.run_name=sacf
python src/train.py model.net.mods.scfm.enabled=True logger=mlflow logger.mlflow.run_name=scfm
python src/train.py model.net.mods.sacf.enabled=True model.net.mods.scfm.enabled=True logger=mlflow logger.mlflow.run_name=sacf-scfm
# python src/train.py trainer.max_epochs=3 model.net.mods.sacf.enabled=True model.net.mods.scfm.enabled=True logger=tensorboard logger.tensorboard.name=sacf-scfm
# python src/train.py trainer.max_epochs=3 model.net.mods.sacf.enabled=True model.net.mods.scfm.enabled=True logger=mlflow logger.mlflow.run_name=sacf-scfm