import argparse
import logging

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from avssl.base import OrderedNamespace
from avssl.data import (
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
    collate_image_captions,
)
from avssl.model import KeywordCascadedSpeechClip, VQCascadedSpeechClip

from .base_task import BaseTask, TrainSpeechClipBaseTask


class TrainVQCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(VQCascadedSpeechClip)


class TrainKeywordCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip)
