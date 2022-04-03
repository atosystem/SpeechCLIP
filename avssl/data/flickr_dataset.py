import json
import logging
import os
from collections import defaultdict
from typing import List

from .base_dataset import BaseDataset, BaseImageCaptionDataset


class FlickrImageCaptionDataset(BaseImageCaptionDataset):
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            **kwargs,
        )

        image_list_txt = os.path.join(
            self.dataset_root, f"Flickr_8k.{self.split}Images.txt"
        )

        wav_base_path = os.path.join(self.dataset_root, "flickr_audio", "wavs")
        wav_list = os.listdir(wav_base_path)
        wav_names = {p[:-6] for p in wav_list if p.split(".")[-1] == "wav"}
        wav_names_to_paths = defaultdict(list)
        for p in wav_list:
            name = p.split("/")[-1][:-6]
            if name in wav_names:
                wav_names_to_paths[name].append(os.path.join(wav_base_path, p))

        id_pairs_path = os.path.join(self.dataset_root, "Flickr8k_idPairs.json")
        with open(id_pairs_path, "r") as f:
            _data = json.load(f)
            id2Filename = _data["id2Filename"]
            filename2Id = _data["filename2Id"]

        with open(image_list_txt, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue

                image_name = line.split(".")[0]  # removed ".jpg"
                image_path = os.path.join(dataset_root, "Images", line)
                if image_name in wav_names:
                    for p in wav_names_to_paths[image_name]:
                        self.data.append(
                            {
                                "wav": p,
                                "image": image_path,
                                "id": filename2Id[image_name],
                            }
                        )

        logging.info(f"Flickr8k ({self.split}): {len(self.data)} samples")


class FlickrDataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str,
        modalities: List,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            **kwargs,
        )

        assert len(modalities) > 0, "Dataset's modalities cannot be none"
        self.modalities = modalities

        image_list_txt = os.path.join(
            self.dataset_root, f"Flickr_8k.{self.split}Images.txt"
        )

        wav_base_path = os.path.join(self.dataset_root, "flickr_audio", "wavs")
        wav_list = os.listdir(wav_base_path)
        wav_names = {p[:-6] for p in wav_list if p.split(".")[-1] == "wav"}
        wav_names_to_paths = defaultdict(list)
        for p in wav_list:
            name = p.split("/")[-1][:-6]
            if name in wav_names:
                wav_names_to_paths[name].append(os.path.join(wav_base_path, p))

        caption_txt_path = os.path.join(self.dataset_root, "captions.txt")
        imageName2captions = {}
        with open(caption_txt_path, "r") as f:
            for _l in f.readlines():
                # skip first line
                if _l.strip() == "image,caption":
                    continue

                _imgName, _caption = _l.split(".jpg,")
                assert isinstance(_imgName, str)
                assert isinstance(_caption, str)
                _caption = _caption.lower().strip()
                if _caption[-1] == ".":
                    _caption = _caption[:-1]
                    _caption = _caption.strip()
                if _imgName not in imageName2captions:
                    imageName2captions[_imgName] = []
                imageName2captions[_imgName].append(_caption)

        id_pairs_path = os.path.join(self.dataset_root, "Flickr8k_idPairs.json")
        with open(id_pairs_path, "r") as f:
            _data = json.load(f)
            id2Filename = _data["id2Filename"]
            filename2Id = _data["filename2Id"]

        with open(image_list_txt, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue

                image_name = line.split(".")[0]  # removed ".jpg"
                image_path = os.path.join(dataset_root, "Images", line)
                if image_name in wav_names:
                    if "audio" in self.modalities or "text" in self.modalities:
                        for p in wav_names_to_paths[image_name]:
                            _entry = {"id": filename2Id[image_name]}
                            _subID = int(
                                os.path.basename(p).split("_")[-1].replace(".wav", "")
                            )

                            if "audio" in self.modalities:
                                _entry["wav"] = p
                            if "image" in self.modalities:
                                _entry["image"] = image_path
                            if "text" in self.modalities:
                                _entry["text"] = imageName2captions[image_name][_subID]
                            self.data.append(_entry)
                    else:
                        self.data.append(
                            {
                                "image": image_path,
                                "id": filename2Id[image_name],
                            }
                        )

        logging.info(f"Flickr8k ({self.split}): {len(self.data)} samples")
