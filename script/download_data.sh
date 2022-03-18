#!/bin/bash

data_root=/work/vjsalt22/dataset

coco_root=${data_root}/coco
places_root=${data_root}/places
flickr8k_root=${data_root}/flickr8k

if [ ! -z "${coco_root}" ]; then
    echo "Downloading SpokenCOCO to ${coco_root}"
    # images train2014 (17 GB) and val2014 (8 GB)
    mkdir -p ${coco_root}
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P ${coco_root}/mscoco_imgfeat
    unzip -q ${coco_root}/mscoco_imgfeat/train2014_obj36.zip -d ${coco_root}/mscoco_imgfeat && rm ${coco_root}/mscoco_imgfeat/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P ${coco_root}/mscoco_imgfeat
    unzip -q ${coco_root}/mscoco_imgfeat/val2014_obj36.zip -d -d ${coco_root}/mscoco_imgfeat && rm ${coco_root}/mscoco_imgfeat/val2014_obj36.zip
    # spoken captions (64G)
    wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P ${coco_root}
    cd ${coco_root}
    tar -xf SpokenCOCO.tar.gz
fi

if [ ! -z "${places_root}" ]; then
    # Images
    # follow http://places.csail.mit.edu/downloadData.html

    # spoken captions (85G)
    wget https://data.csail.mit.edu/placesaudio/placesaudio_2020_splits.tar.gz -P ${places_root}
    cd ${places_root}
    tar -xf placesaudio_2020_splits.tar.gz
fi

if [ ! -z "${flickr8k_root}" ]; then
    # images
    # download e.g. from https://www.kaggle.com/adityajn105/flickr8k/activity
    wget https://www.kaggle.com/adityajn105/flickr8k/download -P 

    # spoken captions 
    wget https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz -P ${flickr8k_root} 
    cd ${flickr8k_root}
    tar -xf flickr_audio.tar.gz
fi