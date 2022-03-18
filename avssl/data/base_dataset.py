# Ref: https://github.com/wnhsu/ResDAVEnet-VQ/blob/master/dataloaders/image_caption_dataset.py
# Author: David Harwath, Wei-Ning Hsu

import torch
import librosa
from PIL import Image
from torch.utils.data import Dataset


class BaseImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str = "",
        dataset_json_file: str = "",
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
    ):
        assert split in {"train", "dev", "test"}
        self.split = split

        self.dataset_root
        self.dataset_json_file = dataset_json_file
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.target_sr = target_sr

        self.data = []

    def _LoadAudio(self, path: str):
        """Load audio from file

        Args:
            path (str): Path to waveform.

        Returns:
            torch.FloatTensor: Audio features.
        """
        waveform, _ = librosa.load(path, sr=self.target_sr)
        if self.audio_transform is not None:
            features = self.audio_transform(waveform)
        else:
            features = torch.FloatTensor(waveform)
        return features

    def _LoadImage(self, path: str):
        """Load image from file

        Args:
            path (str): Path to image.

        Returns:
            torch.FloatTensor: Transformed image.
        """
        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        return img

    def __getitem__(self, index):
        """Get a sample

        Args:
            index (int): Data index.

        Returns:
            torch.FloatTensor: audio features (T, D)
            torch.FloatTensor: image (3, H, W)
        """

        audio_feat = self._LoadAudio(self.data[index]["wav"])
        image = self._LoadImage(self.data[index]["image"])
        return audio_feat, image

    def __len__(self):
        return len(self.data)
