CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    ckpt_path=model/SGVPR_lightning.ckpt \
    data=eprbench \
    model=eprbench \
    trainer=gpu
