echo "[Test] SpeechCLIP Parallel Base on Flickr8k"
EXP_ROOT="exp_test"
DATASET_ROOT="data/flickr"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT
