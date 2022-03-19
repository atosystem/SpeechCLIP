import abc
import argparse


class BaseTask:
    def __init__(self):
        self.args = None
        self.config = None

    @abc.abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError
