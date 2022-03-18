from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def get_lr(optimizer: Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def noam_scheduler(
    optimizer: Optimizer, warmup: int = 4000, last_epoch: int = -1
) -> _LRScheduler:
    def func(step: int):
        if step < warmup:
            return (step + 1) / warmup
        else:
            return (warmup / (step + 1)) ** 0.5

    return LambdaLR(optimizer, func, last_epoch)


def linear_warmup_decay_scheduler(
    optimizer: Optimizer,
    warmup: int = 4000,
    max_step: int = 1000000,
    final_lr: float = 1e-8,
) -> _LRScheduler:
    final_lr_rate = final_lr / get_lr(optimizer)

    def func(step: int):
        if step < warmup:
            return (step + 1) / warmup
        else:
            return 1.0 - (1.0 - final_lr_rate) * (step + 1 - warmup) / (
                max_step - warmup
            )

    return LambdaLR(optimizer, func)


def get_scheduler(name: str, optimizer: Optimizer, **kwargs) -> _LRScheduler:
    if name == "noam":
        return noam_scheduler(optimizer, **kwargs)
    elif name == "linear_warmup_decay":
        return linear_warmup_decay_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"Unknown lr scheduler {name}")
