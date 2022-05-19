#!bin/bash
# conda deactivate
# conda activate zerospeech2021
DATASET=/work/twsezjg982/dataset/zerospeech2021
SUBMISSION=/work/twsezjg982/atosystem/audio-visual-ssl/zerospeech/cosineVq_eachBN_16kw_1head_bsz64_tophubert
# zerospeech2021-validate --only-dev --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
# echo "Validation done!"
zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $SUBMISSION
echo "Evaluation done!"