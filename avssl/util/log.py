import argparse
import logging
from typing import Union

from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from ..base import OrderedNamespace


def set_logging(args: argparse.Namespace) -> None:
    """Setup logging.

    Args:
        args (argparse.Namespace): Arguments.
    """

    level = getattr(logging, str(args.log_level).upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(filename)s.%(funcName)s %(message)s",
        datefmt="%m-%d %H:%M",
    )


def set_pl_logger(args: OrderedNamespace) -> Union[bool, LightningLoggerBase]:
    """Setup PyTorch Lightning logger.

    Args:
        args (OrderedNamespace): Arguments.

    Returns:
        Union[bool, LightningLoggerBase]: Logger.
    """

    logger_type = args.trainer.get("logger", None)

    if logger_type is None or not args.train:
        return None
    elif isinstance(logger_type, bool):
        return logger_type
    elif isinstance(logger_type, WandbLogger):
        return logger_type
    elif logger_type == "wandb":  # or isinstance(logger_type, WandbLogger):
        project = args.logger.project
        name = args.trainer.default_root_dir.split("/")[-1]
        logger = WandbLogger(
            project=project, name=name, save_dir=args.trainer.default_root_dir
        )
        # logger
        logger.experiment.config.update(
            args.to_dict(),
        )
        return logger
    else:
        raise NotImplementedError(f"Unknown logger type = {logger_type}")
