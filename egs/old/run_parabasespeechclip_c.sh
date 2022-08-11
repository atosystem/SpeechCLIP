#!/bin/bash

python3 run_task.py \
    "TrainKeywordCascadedSpeechClip_parallel_baseline" \
    --config "config/speechclip_c/train_flickr_parabase.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 2 \
    --seed 7122 \
    --train \
    --save_path "exp/sc_c/para_base"
