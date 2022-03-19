#!/bin/bash

python3 run_task.py \
    "TrainParallelSpeechClip" \
    --config "config/speechclip_p/train_flickr.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 2 \
    --seed 7122 \
    --save_path "exp/sc_p_tmp"
