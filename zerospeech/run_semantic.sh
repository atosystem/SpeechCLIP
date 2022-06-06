#!bin/bash
# conda deactivate
# conda activate zerospeech2021

DATASET=/work/twsezjg982/dataset/zerospeech2021


# SUBMISSION=/work/twsezjg982/atosystem/audio-visual-ssl/zerospeech/cosineVq_eachBN_16kw_1head_bsz64_tophubert
# SUBMISSION="$1"
# zerospeech2021-validate --only-dev --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
# echo "Validation done!"


# zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION

# exp_dir=cosineVq_eachBN_8kw_1head_bsz64_postBN_KW_SPLIT

exp_dir=cosineVq_eachBN_8kw_1head_bsz64_postBN_KW_SPLIT/

for i in {0..7};
do
    echo "Now in kw_$i"
    cd "/work/twsezjg982/atosystem/audio-visual-ssl/zerospeech/$exp_dir/kw_$i"
    pwd
    SUBMISSION="/work/twsezjg982/atosystem/audio-visual-ssl/zerospeech/$exp_dir/kw_$i"
    zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
    echo "kw_$i done"
done


echo "Evaluation done!"