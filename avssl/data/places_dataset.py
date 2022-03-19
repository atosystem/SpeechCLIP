import os
import json
import logging

from .base_dataset import BaseImageCaptionDataset


class PlacesImageCaptionDataset(BaseImageCaptionDataset):
    def __init__(
        self,
        dataset_json_file: str,
        audio_base_path: str,
        image_base_path: str,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset_json_file=dataset_json_file,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            **kwargs,
        )

        with open(self.dataset_json_file, "r") as fp:
            data_json = json.load(fp)
        self.data = data_json["data"]

        self.audio_base_path = audio_base_path
        self.image_base_path = image_base_path

        for d in self.data:
            d["wav"] = os.path.join(audio_base_path, d["wav"])
            d["image"] = os.path.join(image_base_path, d["image"])

        logging.info(f"Places Audio ({self.split}): {len(self.data)} samples")
