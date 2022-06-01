# get zerospeech embeddings 
python3 getEmbeddings.py \
    --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
    --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8/epoch=161-step=2591-val_recall_mean_1=5.2600.ckpt" \
    --task_name "semantic" \
    --run_dev \
    --output_result_dir "cosineVq_eachBN_8kw_1head_bsz64_kw1_postBN"