#!/bin/bash

# python3 run_task.py \
#     "TrainVQCascadedSpeechClip" \
#     --config "config/speechclip_c/train_flickr_vq.yaml" \
#     --device "cuda:0" \
#     --gpus 1 \
#     --njobs 2 \
#     --seed 7122 \
#     --save_path "exp/sc_c" \
#     --train

# python3 run_task.py \
#     "TrainKeywordProjVQCascadedSpeechClip" \
#     --config "config/speechclip_c/train_flickr_kw_projVq.yaml" \
#     --device "cuda:0" \
#     --gpus 1 \
#     --njobs 8 \
#     --seed 7122 \
#     --save_path "exp/sc_c" \
#     --train

python3 run_task.py \
    "TrainKeywordProjVQCosineCascadedSpeechClip" \
    --config "config/speechclip_c/train_COCO_kw_cosineVq.yaml" \
    --device "cuda:0" \
    --gpus 8 \
    --njobs 4 \
    --seed 7122 \
    --save_path "exp/kw_proj_cosine" \
    --train

    
