#!/bin/bash

# train paraclip mean pooling
python3 run_task.py \
    "TrainParallelSpeechClip_MeanPool" \
    --config "config/speechclip_p/train_flickr.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_mean_pool" \
    --train

# validate paraclip mean pooling with existing checkpoint (validation set)
python3 run_task.py \
    "TrainParallelSpeechClip_MeanPool" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_mean_pool_val" \
    --ckpt "<ckpt_path>" \
    --eval

# test paraclip mean pooling with existing checkpoint (test set)
python3 run_task.py \
    "TrainParallelSpeechClip_MeanPool" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_mean_pool_test" \
    --ckpt "<ckpt_path>" \
    --test

