#!/bin/bash

python3 run_task.py \
    "TrainKeywordCascadedSpeechClip" \
    --config "config/speechclip_c/train_flickr_kw.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 2 \
    --seed 7122 \
    --train \
    --save_path "exp/sc_c/kw_tmp0.7"
