from torch import nn

__all__ = ["freeze_model", "unfreeze_model"]


def freeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True
