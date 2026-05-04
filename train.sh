CUDA_VISIBLE_DEVICES=0 python src/train.py \
    data=eprbench \
    model=eprbench \
    trainer=gpu \
    logger=csv
