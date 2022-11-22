echo "[Test] SpeechCLIP Parallel Large on SpokenCOCO"
EXP_ROOT="exp_test"
mkdir $EXP_ROOT
DATASET_ROOT="data/coco"
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/large/coco/parallel/epoch_14-step_33224-val_recall_mean_10_84.0128.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT


