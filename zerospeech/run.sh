# get zerospeech embeddings 
python3 getEmbeddings.py \
    --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
    --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_16_bsz_64_reinit_last2/epoch=49-step=23399-val_recall_mean_1=9.5900.ckpt" \
    --task_name "semantic" \
    --run_dev \
    --output_result_dir "cosineVq_eachBN_16kw_1head_bsz64_hubert_last2"