#!/bin/bash

python3 run_task.py \
    "TrainEPKeywordCascadedSpeechClip" \
    --config "config/speechclip_c/train_flickr_epkw.yaml" \
    --device "cuda:0" \
    --gpus 1 \
    --njobs 2 \
    --seed 7122 \
    --train
    --save "exp/ep/kw8_head1_fixep" \
