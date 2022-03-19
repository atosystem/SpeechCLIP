import argparse
import logging

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split

from avssl.base import OrderedNamespace
from avssl.data import FlickrImageCaptionDataset, PlacesImageCaptionDataset
from avssl.model import ParallelSpeechClip


def main(args: argparse.Namespace):
    seed_everything(args.seed)

    if args.ckpt != "":
        model = ParallelSpeechClip.load_from_checkpoint(args.ckpt).to(args.device)
        config = model.config
    else:
        args.ckpt = None
        config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
        config = OrderedNamespace([args, config])
        model = ParallelSpeechClip(config).to(args.device)

    if config.data.dataset.name == "flickr":
        tr_set = FlickrImageCaptionDataset(split="train", **config.data.dataset)
        dv_set = FlickrImageCaptionDataset(split="dev", **config.data.dataset)
    elif config.data.dataset.name == "places":
        tr_set = PlacesImageCaptionDataset(split="train", **config.data.dataset)
        tr_len = int(len(tr_set) * 0.9)
        tr_set, dv_set = random_split(
            tr_set,
            [tr_len, len(tr_set) - tr_len],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        raise NotImplementedError(f"Unknown dataset {config.data.dataset.name}")

    tr_loader = DataLoader(
        tr_set,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=args.njobs,
        pin_memory=True,
        drop_last=True,
    )
    dv_loader = DataLoader(
        dv_set,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=args.njobs,
        pin_memory=True,
        drop_last=False,
    )

    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_acc:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        every_n_epochs=1,
    )

    trainer = Trainer(
        callbacks=[model_checkpoint, TQDMProgressBar()],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        gpus=args.gpus,
        **config.trainer,
    )

    trainer.fit(model, tr_loader, dv_loader, ckpt_path=args.ckpt)
