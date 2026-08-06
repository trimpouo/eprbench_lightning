# Experiments — 复现命令索引

> `results.tsv` 中每一行（Ex / Hx）都能在这里找到对应复现命令。
> 公共前提：`data.num_workers=2`；GPU 为 RTX A4000 15.6 GiB；
> SACF checkpoint 评估时必须加 `model.net.mods.sacf.enabled=true`。
> 带 `r` 后缀的行（E05r 等）是同一联合训练 checkpoint 在 RGB 测试侧的结果，无需单独训练。

## E01 — EventVPR Baseline + text

```bash
python src/train.py \
  data=eprbench model=eprbench trainer=gpu \
  data.use_text=true model.use_text=true model.lambda_contrast=0.25 \
  data.train_batch_size=18 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30
```

## E02 — EventVPR SACF-only + text

```bash
python src/train.py \
  experiment=eprbench_sacf_only trainer=gpu \
  data.train_batch_size=12 data.infer_batch_size=16 data.num_workers=2 \
  trainer.max_epochs=30
```

注意：batch 18 会 OOM（见 `../evidence/failure-cases.md` N1）。

## E03 — 跨数据集：E01 → NYC Event

```bash
python src/eval.py \
  ckpt_path=<E01 的 epoch_029.ckpt> \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  data.infer_batch_size=32 data.num_workers=2
```

## E04 — 跨数据集：E02 → NYC Event

```bash
python src/eval.py \
  ckpt_path=<E02 的 epoch_029.ckpt> \
  data=nyc_vpr_event model=eprbench trainer=gpu \
  data.use_text=false model.use_text=false \
  model.net.mods.sacf.enabled=true \
  data.infer_batch_size=32 data.num_workers=2
```

## E05 / E06 / E07 — NYC 联合训练消融

```bash
# E05 SACF only
python src/train.py data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sac_only_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5

# E06 SCFM only
python src/train.py data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_scfm_only_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5

# E07 SACF + SCFM（结果为 discard，见 N2）
python src/train.py data=nyc_vpr_joint model=eprbench trainer=gpu \
  experiment=nyc_vpr_sacf_scfm_joint \
  trainer.max_epochs=30 trainer.check_val_every_n_epoch=5
```

## E08 — NYC 单模态 visual-only 参考基线

```bash
python src/train.py data=nyc_vpr_event model=eprbench trainer=gpu \
  experiment=nyc_vpr_visual_only trainer.max_epochs=30
# RGB 侧同理：data=nyc_vpr_rgb
```

## H01–H07 — SACF 插入层 sweep（历史）

```bash
python src/train.py model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.variant=original \
  model.net.mods.sacf.layer_idx=<1..7>
```

完整脚本见 `scripts/sacf_sweep.sh`。**注意**：该 sweep 基于历史基线（R@1 = 94.33），
绝对数值不可与 E01/E02 直接比较。

## H08–H10 — SE-RSACF 消融（历史）

命令模板见 `docs/eventvpr_exploration_journey.md` 附录 A.1 配方 B 与
`scripts/se_rsacf_ablation.sh`。

## M01–M15 — MLflow 消融网格（来源 `logs/mlflow.db`）

> M 系列的原始记录全部在 `mlflow_runs.tsv`（116 行全量导出，含 run_id 与 git commit）；
> 详细判读见 `../evidence/mlflow-analysis.md`。
> 统一协议：EventVPR，文本开启，`lambda_contrast=0.15`，`trainer.max_epochs=50`，
> `data.num_workers=2`，AdamW（lr=1e-4, wd=1e-3）。
> exp1 网格的完整批量脚本：`scripts/ablation_scfm_sacf.sh`（注意该脚本默认优化器
> 已随代码库切换为 AdamW，对应 run 名后缀 `W`）。

```bash
# M01 baseline（全部模块关闭）
python src/train.py logger=mlflow logger.mlflow.run_name=baseline

# M02 SACF only, layer 6（sacfW）
python src/train.py model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.layer_idx=6 \
  logger=mlflow logger.mlflow.run_name=sacfW

# M03 SCFM only（scfmW）
python src/train.py model.net.mods.scfm.enabled=True \
  logger=mlflow logger.mlflow.run_name=scfmW

# M04 SACF + SCFM（sacf-scfmW）
python src/train.py model.net.mods.sacf.enabled=True \
  model.net.mods.scfm.enabled=True \
  logger=mlflow logger.mlflow.run_name=sacf-scfmW

# M05 EATS + ES-SACF
python src/train.py model.net.mods.event_cues.enabled=True \
  model.net.mods.event_aware_token_selector.enabled=True \
  model.net.mods.sacf.enabled=True model.net.mods.sacf.use_event_stability=True \
  logger=mlflow logger.mlflow.run_name=eats-essacf

# M06 EATS + SACF + SCFM（original 变体）
python src/train.py model.net.mods.event_cues.enabled=True \
  model.net.mods.event_aware_token_selector.enabled=True \
  model.net.mods.sacf.enabled=True model.net.mods.scfm.enabled=True \
  logger=mlflow logger.mlflow.run_name=eats-orig-sacfm

# M07 EATS + ES-SACF + ER-SCFM（全组合）
python src/train.py model.net.mods.event_cues.enabled=True \
  model.net.mods.event_aware_token_selector.enabled=True \
  model.net.mods.sacf.enabled=True model.net.mods.sacf.use_event_stability=True \
  model.net.mods.scfm.enabled=True model.net.mods.scfm.use_event_reliable=True \
  logger=mlflow logger.mlflow.run_name=eats-essacf-erscfm

# M08–M10 core 附加技术（在 M07 基础上叠加）
python src/train.py <M07 参数> model.use_temporal_consistency=True            # M08
python src/train.py <M07 参数> model.text_dropout_prob=0.15 model.word_dropout_prob=0.1  # M09
python src/train.py <M07 参数> model.use_temporal_consistency=True \
  model.text_dropout_prob=0.15 model.word_dropout_prob=0.1                     # M10

# M11 / M12 SE-RSACF 变体（完整参数模板见 scripts/se_rsacf_ablation.sh）
bash scripts/se_rsacf_ablation.sh   # 内含 evt-only / adaptive-topk 等全部变体

# M13 ES-SACF 单独开启（N5 反例，勿作为正面配方复现）
python src/train.py model.net.mods.sacf.enabled=True \
  model.net.mods.sacf.use_event_stability=True \
  logger=mlflow logger.mlflow.run_name=essacf-only

# M14 / M15 10-epoch 快速筛选协议
python src/train.py model.net.mods.spm.enabled=True \
  trainer.max_epochs=10 data.train_batch_size=4   # M14（batch 4 才能跑通）
python src/train.py model.net.mods.lgfp.enabled=True \
  trainer.max_epochs=10 data.train_batch_size=12  # M15
```
