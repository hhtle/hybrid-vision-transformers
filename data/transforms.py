from typing import Tuple
import torch
from torchvision import transforms as T

# Standard normalization statistics for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# MNIST has a single channel originally, but we will turn it into 3 channels
# so we reuse the same scalar mean/std on all three channels.
MNIST_MEAN3  = (0.1307, 0.1307, 0.1307)
MNIST_STD3   = (0.3081, 0.3081, 0.3081)


class To3Channels(torch.nn.Module):
    """
    Convert a single-channel tensor [1, H, W] into a 3-channel tensor [3, H, W]
    by repeating the channel.

    This allows us to use the same model architecture (3 input channels)
    for both MNIST and CIFAR-10.
    """
    def forward(self, x):
        # x is expected to be a tensor with shape [C, H, W]
        # If C == 1, repeat along the channel dimension.
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def build_transforms(name: str, img_size: int, augment: bool) -> Tuple[T.Compose, T.Compose]:
    """
    Build train and test/eval transforms for a given dataset.

    Arguments:
      name: dataset name ("cifar10" or "mnist")
      img_size: target image size after resizing/cropping
      augment: if False, disable strong train-time augmentations

    Returns:
      train_transforms: torchvision.transforms.Compose for the training set
      test_transforms:  torchvision.transforms.Compose for validation/test sets
    """
    name = name.lower()

    if name == "cifar10":
        # CIFAR-10: 3-channel RGB images.
        # Typical augmentations: random horizontal flip + random crop with padding.
        train_tfms = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(img_size, padding=4),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        test_tfms = [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]

    elif name == "mnist":
        # MNIST: originally 1-channel 28x28 images.
        # We resize to img_size and convert to 3 channels, then normalize.
        train_tfms = [
            T.Resize(img_size),
            T.ToTensor(),
            To3Channels(),
            T.Normalize(MNIST_MEAN3, MNIST_STD3),
        ]
        test_tfms = [
            T.Resize(img_size),
            T.ToTensor(),
            To3Channels(),
            T.Normalize(MNIST_MEAN3, MNIST_STD3),
        ]

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Optionally disable strong data augmentation for training
    if not augment:
        # For CIFAR-10, if augment=False, we simply use the test transforms
        # (i.e., no random crop or flip).
        if name == "cifar10":
            train_tfms = test_tfms

    return T.Compose(train_tfms), T.Compose(test_tfms)
