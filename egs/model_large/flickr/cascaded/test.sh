echo "[Test] SpeechCLIP Cascaded Large on Flickr8k"
EXP_ROOT="exp_test"
DATASET_ROOT="data/flickr"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/large/flickr/cascaded/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT


