import json
import logging
import os

from .base_dataset import BaseImageCaptionDataset


class PlacesImageCaptionDataset(BaseImageCaptionDataset):
    def __init__(
        self,
        audio_base_path: str,
        image_base_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        with open(self.dataset_json_file, "r") as fp:
            data_json = json.load(fp)
        self.data = data_json["data"]

        self.audio_base_path = audio_base_path
        self.image_base_path = image_base_path

        for i in len(self.data):
            self.data[i]["wav"] = os.path.join(audio_base_path, self.data[i]["wav"])
            self.data[i]["image"] = os.path.join(image_base_path, self.data[i]["image"])

        logging.info(f"Places Audio ({self.split}): {len(self.data)} samples")
