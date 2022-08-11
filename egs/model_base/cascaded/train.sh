echo "[Train] SpeechCLIP Cascaded Base on Flickr8k"
EXP_ROOT="exp_test"
CFG="config/speechCLIP/model_base/spchclp_c.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


