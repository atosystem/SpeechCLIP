echo "[Train] SpeechCLIP Cascaded Large on SpokenCOCO"
EXP_ROOT="exp_test"
CFG="config/speechCLIP/model_large/coco/spchclp_c.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG \
    --gpus 4 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


