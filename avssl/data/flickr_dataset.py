import os
import json
import logging
from collections import defaultdict

from .base_dataset import BaseImageCaptionDataset


class FlickrImageCaptionDataset(BaseImageCaptionDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        image_list_txt = os.path.join(
            self.dataset_root, f"Flickr_8k.{self.split}Images.txt"
        )

        wav_list = os.listdir(os.path.join(self.dataset_root, "flickr_audio", "wavs"))
        wav_names = {
            p.split("/")[-1][:-6] for p in wav_list if p.split(".")[-1] == "wav"
        }
        wav_names_to_paths = defaultdict(list)
        for p in wav_list:
            name = p.split("/")[-1][:-6]
            if name in wav_names:
                wav_names_to_paths[name].append(p)

        with open(image_list_txt, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue

                image_name = line.split(".")[0]  # removed ".jpg"
                image_path = os.path.join(dataset_root, "Images", image_name)
                if image_name in wav_names:
                    for p in wav_names_to_paths[image_name]:
                        self.data.append({"wav": p, "image": image_path})

        logging.info(f"Flickr8k ({self.split}): {len(self.data)} samples")
