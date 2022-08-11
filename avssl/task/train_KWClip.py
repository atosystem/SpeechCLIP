import argparse
import logging
import os

import torch
import yaml
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from ..base import OrderedNamespace
from ..model import (
    KWClip_CLIP_Original,
    KWClip_GeneralTransformer,
    KWClip_GeneralTransformer_SpeechText,
    KWClip_SpeechText,
)
from .base_task import BaseTask, TrainSpeechClipBaseTask


class CheckpointAtStep(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        save_at_steps=[],
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.saved_keypoint = False
        self.save_at_steps = save_at_steps
        self.saved_steps = []

    def on_batch_end(self, trainer: Trainer, _):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        for i in self.save_at_steps:
            if not (i in self.saved_steps) and global_step >= i:
                filename = "{}_k_{}_epoch={}_global_step={}.ckpt".format(
                    self.prefix, i, epoch, global_step
                )
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
                trainer.save_checkpoint(ckpt_path)
                self.saved_steps.append(i)

        # save_keypoint = trainer.model.config.codebook_penalty.save_keypoint
        # if not self.saved_keypoint and global_step >= save_keypoint:
        #     filename = "{}_k_{}_epoch={}_global_step={}.ckpt".format(
        #         self.prefix,
        #         save_keypoint,
        #         epoch,
        #         global_step
        #     )
        #     ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        #     trainer.save_checkpoint(ckpt_path)
        #     self.saved_keypoint = True


class TrainKWClip_GeneralTransformer(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KWClip_GeneralTransformer)


class TrainKWClip_SpeechText(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KWClip_SpeechText)


class TrainKWClip_Original(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KWClip_CLIP_Original)


class TrainKWClip_GeneralSpeechText(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KWClip_GeneralTransformer_SpeechText)
