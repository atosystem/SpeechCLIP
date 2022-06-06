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
from ..data import (
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
    collate_image_captions,
)
from ..model import (
    KeywordCascadedSpeechClip,
    KeywordCascadedSpeechClip_CodeBookPenalty,
    KeywordCascadedSpeechClip_parallel_baseline,
    KeywordCascadedSpeechClip_ProjVQ,
    KeywordCascadedSpeechClip_ProjVQ_Cosine,
    KeywordCascadedSpeechClip_ProjVQ_Cosine_AttMap_Constraint,
    KeywordCascadedSpeechClip_ProjVQ_Cosine_w_Parallel,
    KeywordCascadedSpeechClipBN,
    KeywordCascadedSpeechClipBNEachKw,
    KeywordCascadedSpeechClipNLayer,
    VQCascadedSpeechClip,
    KeywordCascadedSpeechClip_CIF_BN,
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


class TrainVQCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(VQCascadedSpeechClip)


class TrainKeywordCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip,
        )


class TrainKeywordCascadedSpeechClipNLayer(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClipNLayer,
        )


class TrainKeywordProjVQCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip_ProjVQ)


class TrainKeywordProjVQCosineCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip_ProjVQ_Cosine)


class TrainKeywordCascadedSpeechClip_CodeBookPenalty(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip_CodeBookPenalty,
            # custom_trainer_callbacks=[
            #     CheckpointAtStep(
            #         0, prefix="custom", save_at_steps=[2000, 4000, 6000, 8000, 10000]
            #     )
            # ],
        )


class TrainKeywordCascadedSpeechClipBN(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClipBN,
        )


class TrainKeywordCascadedSpeechClipBNEachKw(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClipBNEachKw,
        )


class TrainKeywordCascadedSpeechClip_parallel_baseline(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip_parallel_baseline,
        )


class TrainKeywordProjVQCosineCascadedSpeechClip_w_Parallel(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip_ProjVQ_Cosine_w_Parallel,
        )


class TrainKeywordProjVQCosineCascadedSpeechClip_AttMap_Constraint(
    TrainSpeechClipBaseTask
):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip_ProjVQ_Cosine_AttMap_Constraint,
        )

class TrainKeywordProjVQCosineCascadedSpeechClip_CIF_BN(
    TrainSpeechClipBaseTask
):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(
            KeywordCascadedSpeechClip_CIF_BN,
        )
