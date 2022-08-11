pwd

EXP_ROOT="exp_test"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --ckpt "/work/vjsalt22/atosystem/audio-visual-ssl/slt_ckpts/SpeechCLIP/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt" \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT


