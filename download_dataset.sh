# Flickr8k Images and Captions
mkdir data/flickr/Images
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P data/flickr
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P data/flickr
unzip data/flickr/Flickr8k_text.zip -d  data/flickr
unzip data/flickr/Flickr8k_Dataset.zip -d  data/flickr/Images

# Flickr8k Audio
wget https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz -P data/flickr
tar -xzvf data/flickr/flickr_audio.tar.gz -C data/flickr


# Spoken COCO Audio
mkdir data/coco/SpokenCOCO
wget https://huggingface.co/speechclip/models/resolve/main/ksplit.tar.gz -P data/coco
tar -xzvf data/coco/ksplit.tar.gz -C data/coco/SpokenCOCO
mv data/coco/SpokenCOCO/ksplit/* data/coco/SpokenCOCO
rm -r data/coco/SpokenCOCO/ksplit/

wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P data/coco
tar -xzvf data/coco/SpokenCOCO.tar.gz -C data/coco

# Spoken COCO Images
wget http://images.cocodataset.org/zips/train2014.zip -P data/coco/mscoco_img
wget http://images.cocodataset.org/zips/val2014.zip -P data/coco/mscoco_img
unzip data/coco/mscoco_img/train2014.zip -d  data/coco/mscoco_img
unzip data/coco/mscoco_img/val2014.zip -d  data/coco/mscoco_img


echo "Done downloading Flickr8k and SpokenCOCO"
