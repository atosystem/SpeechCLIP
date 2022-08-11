#!/bin/bash

# CLIP Text Image Retrieval
python3 run_task.py \
    "OriginalCLIPTextImage" \
    --config "config/speechclip_p/train_flickr.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_original" \
    --test

# test paraclip mean pooling with existing checkpoint (test set) for text audio retrieval
python3 run_task.py \
    "TrainParallelSpeechClip_MeanPool_Text" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_mean_pool_test_ontext" \
    --ckpt "<ckpt_path>" \
    --test

# test paraclip attentive pooling with existing checkpoint (test set) for text audio retrieval
python3 run_task.py \
    "TrainParallelSpeechClip_AttPool_Text" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_att_pool_test_ontext" \
    --ckpt "<ckpt_path>" \
    --test
