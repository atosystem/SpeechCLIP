echo "[Test] SpeechCLIP Cascaded Large on SpokenCOCO"
EXP_ROOT="exp_test"
mkdir $EXP_ROOT
DATASET_ROOT="data/coco"
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/large/coco/cascaded/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT


