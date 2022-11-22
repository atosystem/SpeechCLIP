import argparse


def add_general_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for general usage.
    Args:
        parser (argparse.ArgumentParser): Argument parser.
    Returns:
        argparse.ArgumentParser: Argument parser with arguments added.
    """

    # Config & ckpt
    parser.add_argument("--config", type=str, default="", help="Config file (.yaml)")
    parser.add_argument(
        "--save_path", type=str, default="", help="Directory to save ckpts."
    )

    # Mode
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Run dev set")
    parser.add_argument("--test", action="store_true", help="Run test set")
    parser.add_argument("--ckpt", type=str, default="", help="Load from checkpoint")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume.")

    # Hparams
    parser.add_argument("--njobs", type=int, default=0, help="Number of workers")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=7122, help="Random seed")

    # overide dataset path
    parser.add_argument(
        "--dataset_root", type=str, default="", help="Override dataset root"
    )

    # Logging
    parser.add_argument("--log_level", type=str, default="info", help="Logging level")

    return parser
