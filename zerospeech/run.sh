# get zerospeech embeddings 
# out_dir="cosineVq_kw_8_bsz_64_weightedSum_fixed_1_ep48_recall_6.1"
out_dir="cosineVq_kw_8_bsz_240_weightedSum_sanity"

# cosineVq_kw_8_bsz_64_weightedSum_vqTemp_fixed_0.1
# python3 getEmbeddings.py \
#     --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
#     --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_bsz_64_weightedSum/epoch=48-step=22931-val_recall_mean_1=6.1000.ckpt" \
#     --task_name "semantic" \
#     --run_dev \
#     --output_result_dir $out_dir

python3 getEmbeddings.py \
    --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
    --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8/epoch=161-step=2591-val_recall_mean_1=5.2600.ckpt" \
    --task_name "semantic" \
    --run_dev \
    --output_result_dir $out_dir



# conda activate zerospeech2021
# python computeKWSemantic.py $out_dir

# python3 getEmbeddings.py \
#     --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
#     --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_bsz_64_reinit_last2/epoch=36-step=17315-val_recall_mean_1=6.9600.ckpt" \
#     --task_name "semantic" \
#     --run_dev \
#     --output_result_dir "cosineVq_eachBN_8kw_1head_bsz64_reInit2_postBN"


# python3 getEmbeddings.py \
#     --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
#     --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_16_bsz_64_weightedSum/epoch=47-step=22463-val_recall_mean_1=7.8000.ckpt" \
#     --task_name "semantic" \
#     --run_dev \
#     --output_result_dir "cosineVq_eachBN_16kw_1head_bsz64_weightedSum__kw2_postBN"

# python3 getEmbeddings.py \
#     --model_cls_name "KeywordCascadedSpeechClip_ProjVQ_Cosine" \
#     --model_ckpt "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_bsz_64_weightedSum/epoch=48-step=22931-val_recall_mean_1=6.1000.ckpt" \
#     --task_name "semantic" \
#     --run_dev \
#     --output_result_dir "cosineVq_eachBN_8kw_1head_bsz64_weightedSum_postBN"

