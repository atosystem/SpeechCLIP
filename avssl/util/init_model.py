from torch import nn


def init_weights(m: nn.Module) -> None:
    """Initialize module's weights

    Args:
        m (nn.Module): Module.
    """
    if isinstance(m, nn.Module) and hasattr(m, "reset_parameters"):
        m.reset_parameters()
