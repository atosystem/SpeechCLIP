echo "SpeechCLIP Cascaded Base on Flickr8k"
EXP_ROOT="exp_test"
DATASET_ROOT="data/flickr"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/base/flickr/cascaded/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT
