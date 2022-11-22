# Data preparation
Please follow the instructions to download the datasets.

1. Run `bash download_dataset.sh`
2. Go to directory `data/flickr` and run `python createIdForDataPairs.py`
3. Make sure the `dataset_root` is set correctly when train or inference the models
    * `dataset_root` = `data/flickr` for Flickr8k
    * `dataset_root` = `data/coco` for SpokenCOCO
    
    > when inference, remember to overide `dataset_root` in the shell scripts
    For example: (`egs/model_base/parallel/test.sh`)
    ```bash
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
    ```
    > when training, remember to check the `dataset_root` in yaml config
    For example (`config/speechCLIP/model_base/spchclp_p.yaml`)
    ```yaml
    data:
        dataset:
            name: flickr
            dataset_root: data/flickr
            text_file: Flickr8k.token.txt
            clip_image_transform: ViT-B/32
            load_image: true
            load_audio: true
            tokenizeText: true
    ...
    ```