import sys
import argparse
import warnings
import importlib

from avssl import task

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    args, unknown = parser.parse_known_args()

    runner = getattr(task, args.task)()

    parser = runner.add_args(parser)
    args = runner.parse_args(parser)

    runner.run()
