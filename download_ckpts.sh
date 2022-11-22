# for downloading checkpoints
mkdir slt_ckpts
mkdir slt_ckpts/SpeechCLIP
mkdir slt_ckpts/SpeechCLIP/base
mkdir slt_ckpts/SpeechCLIP/base/flickr
mkdir slt_ckpts/SpeechCLIP/base/flickr/cascaded
wget https://huggingface.co/speechclip/models/resolve/main/base/flickr/cascaded/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt -P slt_ckpts/SpeechCLIP/base/flickr/cascaded
mkdir slt_ckpts/SpeechCLIP/base/flickr/parallel
wget https://huggingface.co/speechclip/models/resolve/main/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt -P slt_ckpts/SpeechCLIP/base/flickr/parallel


mkdir slt_ckpts/SpeechCLIP/large
mkdir slt_ckpts/SpeechCLIP/large/flickr
mkdir slt_ckpts/SpeechCLIP/large/flickr/cascaded
wget https://huggingface.co/speechclip/models/resolve/main/large/flickr/cascaded/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt -P slt_ckpts/SpeechCLIP/large/flickr/cascaded
mkdir slt_ckpts/SpeechCLIP/large/flickr/parallel
wget https://huggingface.co/speechclip/models/resolve/main/large/flickr/parallel/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt -P slt_ckpts/SpeechCLIP/large/flickr/parallel

mkdir slt_ckpts/SpeechCLIP/large/coco
mkdir slt_ckpts/SpeechCLIP/large/coco/cascaded
wget https://huggingface.co/speechclip/models/resolve/main/large/coco/cascaded/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt -P slt_ckpts/SpeechCLIP/large/coco/cascaded
mkdir slt_ckpts/SpeechCLIP/large/coco/parallel
wget https://huggingface.co/speechclip/models/resolve/main/large/coco/parallel/epoch_14-step_33224-val_recall_mean_10_84.0128.ckpt -P slt_ckpts/SpeechCLIP/large/coco/parallel


echo "Done downloading all checkpoints"





