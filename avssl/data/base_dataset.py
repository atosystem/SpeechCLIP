# Ref: https://github.com/wnhsu/ResDAVEnet-VQ/blob/master/dataloaders/image_caption_dataset.py
# Author: David Harwath, Wei-Ning Hsu

import logging
import os
import pickle
from string import Template
from typing import List, Union

import clip
import librosa
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """BaseDataset
    Generalized for modalities (Image,Audio,Text)
    """

    def __init__(
        self,
        dataset_root: str = "",
        dataset_json_file: str = "",
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        tokenizeText: bool = True,
        normalize_waveform: bool = False,
        **kwargs,
    ):
        """init

        Args:
            dataset_root (str, optional): dataset_root. Defaults to "".
            dataset_json_file (str, optional): Defaults to "".
            split (str, optional): data split. Defaults to "train".
            image_transform (, optional):  Defaults to None.
            audio_transform (, optional):  Defaults to None.
            target_sr (int, optional): Defaults to 16_000.
            load_audio (bool, optional): load audio file to tensor. Defaults to True.
            load_image (bool, optional): load image file to tensor. Defaults to True.
            tokenizeText (bool, optional): tokenize text input with clip tokenizer. Defaults to True.
        """
        self.split = split

        t = Template(dataset_root)
        self.dataset_root = dataset_root
        self.dataset_json_file = dataset_json_file
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.target_sr = target_sr
        self.load_audio = load_audio
        self.load_image = load_image
        self.tokenizeText = tokenizeText
        self.normalize_waveform = normalize_waveform
        if self.normalize_waveform:
            logger.info("Normalize input waveform")

        self.data = []

    def _LoadAudio(self, path: str):
        """Load audio from file

        Args:
            path (str): Path to waveform.

        Returns:
            torch.FloatTensor: Audio features.
        """

        if self.load_audio:
            waveform, _ = librosa.load(path, sr=self.target_sr)
            if self.audio_transform is not None:
                audio = self.audio_transform(waveform)
            else:
                audio = torch.FloatTensor(waveform)
            if self.normalize_waveform:
                audio = F.layer_norm(audio, audio.shape)
        else:
            audio = path

        return audio

    def _LoadImage(self, path: str):
        """Load image from file

        Args:
            path (str): Path to image.

        Returns:
            torch.FloatTensor: Transformed image.
        """

        if self.load_image:
            img = Image.open(path).convert("RGB")
            if self.image_transform is not None:
                img = self.image_transform(img)
        else:
            img = path

        return img

    def _TokenizeText(self, texts: Union[str, List[str]]):
        if self.tokenizeText:
            return clip.tokenize(texts=texts, context_length=77, truncate=False)
        else:
            return texts

    def __getitem__(self, index):
        """Get a sample

        Args:
            index (int): Data index.

        Returns:
            Dict
                wav : torch.FloatTensor: audio features (T, D)
                image : torch.FloatTensor: image (3, H, W)
                text : torch.LongTensor:
                id :  torch.LongTensor
        """

        ret_dict = {}
        if "wav" in self.data[index]:
            audio_feat = self._LoadAudio(self.data[index]["wav"])
            ret_dict["wav"] = audio_feat
        if "image" in self.data[index]:
            image = self._LoadImage(self.data[index]["image"])
            ret_dict["image"] = image
        if "text" in self.data[index]:
            text = self._TokenizeText(self.data[index]["text"])
            ret_dict["text"] = text
        if "id" in self.data[index]:
            ret_dict["id"] = self.data[index]["id"]

        assert len(ret_dict) > 0, "dataset getitem must not be empty"

        return ret_dict

    def __len__(self):
        return len(self.data)
