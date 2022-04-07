#!/bin/bash

python3 run_task.py \
    "TrainCascadedSpeechClip" \
    --config "config/speechclip_c/train_flickr.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 2 \
    --seed 7122 \
    --save_path "exp/sc_c_tmp"
