#!/bin/bash

# train paraclip with attentive pooling
python3 run_task.py \
    "TrainParallelSpeechClip_AttPool" \
    --config "config/speechclip_p/train_flickr_attPool.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool" \
    --train

# validate paraclip with attentive pooling
python3 run_task.py \
    --config "config/speechclip_p/train_flickr_attPool.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool_val" \
    --ckpt "<ckpt_path>" \
    --eval

# test paraclip with attentive pooling
python3 run_task.py \
    --config "config/speechclip_p/train_flickr_attPool.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool_test" \
    --ckpt "<ckpt_path>" \
    --test


# train paraclip with attentive pooling (degraded)
# note that for degraded version, the dimension for both modalitites must be in same dimension (pre-projection is allowed)
python3 run_task.py \
    "TrainParallelSpeechClip_AttPool" \
    --config "config/speechclip_p/train_flickr_attPool_degraded.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool_degraded" \
    --train

# finegrained attentive pooling
# Hook to Residual Blocks in ViT for representations
python3 run_task.py \
    "TrainParallelSpeechClip_AttPool_FineGrainHookResBlk" \
    --config "config/speechclip_p/train_flickr_attPool_fineGrained_ResBlkHook_pre.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool_fingrained_HookResBlk" \
    --train


# Use the output of transformer of ViT as representations
python3 run_task.py \
    "TrainParallelSpeechClip_AttPool_FineGrain" \
    --config "config/speechclip_p/train_flickr_attPool_fineGrained_pre.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 16 \
    --seed 7122 \
    --save_path "exp/sc_p_attpool_fingrained" \
    --train
