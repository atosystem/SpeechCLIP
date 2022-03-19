import torch
from torchvision import transforms


def get_simple_image_transform(H: int, W: int = -1) -> transforms.Compose:
    """Get a simple image transformation function.

    Args:
        H (int): Height of output image.
        W (int, optional): Width of output image. Default to -1.

    Returns:
        transforms.Compose: transform function
    """
    if W <= 0:
        W = H
    transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
    return transform
