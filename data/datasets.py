from typing import Tuple, Optional
import os
import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets as tvds

from .transforms import build_transforms


def _val_split(train_dataset: Dataset, val_size: int, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Split a training dataset into a smaller train set and a validation set.

    Arguments:
      train_dataset: full training dataset (before splitting)
      val_size: desired size of the validation set
      seed: random seed to make the split deterministic

    Returns:
      train_ds: training subset after splitting
      val_ds: validation subset
    """
    n = len(train_dataset)
    # Make sure we do not ask for more validation samples than available
    val_size = min(val_size, n)
    train_size = n - val_size

    # Create a generator with a fixed seed so the split is reproducible
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], generator=g)
    return train_ds, val_ds


def get_mnist(root: str, img_size: int, augment: bool, seed: int, val_size: int = 5000):
    """
    Build MNIST train/val/test datasets with a deterministic train/val split.

    We use two versions of the training set:
      - train_full: with training transforms (including augmentations)
      - train_full_eval: with evaluation transforms (no train-time augmentation)

    The split into train/val is done on train_full_eval so that:
      - val and test share the same preprocessing,
      - the split is stable and reproducible.

    After splitting, we override the dataset used by the train subset with
    train_full (the one that includes augmentations).
    """
    # Build transforms for MNIST (train and test/eval)
    train_tfms, test_tfms = build_transforms("mnist", img_size, augment)

    # Full training set with train-time transforms (augmentations)
    train_full = tvds.MNIST(root=root, train=True, download=True, transform=train_tfms)

    # Test set with eval transforms
    test_ds = tvds.MNIST(root=root, train=False, download=True, transform=test_tfms)

    # Second copy of the training set with eval transforms (no aug) for a clean val split
    train_full_eval = tvds.MNIST(root=root, train=True, download=True, transform=test_tfms)

    # Deterministic split of train_full_eval into train and val
    train_ds, val_ds = _val_split(train_full_eval, val_size, seed)

    # Replace the underlying dataset of the train subset with the augmented version
    train_ds.dataset = train_full

    num_classes = 10
    return train_ds, val_ds, test_ds, num_classes


def get_cifar10(root: str, img_size: int, augment: bool, seed: int, val_size: int = 5000):
    """
    Build CIFAR-10 train/val/test datasets with a deterministic train/val split.

    Same logic as for MNIST:
      - train_full: training transforms (with augmentation)
      - train_full_eval: evaluation transforms (without augmentation)

    The split is performed on train_full_eval for a stable validation set
    that shares preprocessing with the test set, and then the training subset
    is pointed back to train_full.
    """
    # Build transforms for CIFAR-10 (train and test/eval)
    train_tfms, test_tfms = build_transforms("cifar10", img_size, augment)

    # Full training set with train-time transforms (augmentations)
    train_full = tvds.CIFAR10(root=root, train=True, download=True, transform=train_tfms)

    # Test set with eval transforms
    test_ds = tvds.CIFAR10(root=root, train=False, download=True, transform=test_tfms)

    # Second copy for stable train/val split with eval transforms
    train_full_eval = tvds.CIFAR10(root=root, train=True, download=True, transform=test_tfms)

    # Deterministic split into train and val
    train_ds, val_ds = _val_split(train_full_eval, val_size, seed)

    # Use the augmented dataset for the training subset
    train_ds.dataset = train_full

    num_classes = 10
    return train_ds, val_ds, test_ds, num_classes


def build_datasets(
    name: str,
    root: str,
    img_size: int,
    augment: bool,
    seed: int,
    val_size: Optional[int] = None,
):
    """
    Factory function that builds train/val/test datasets for a given dataset name.

    Arguments:
      name: dataset name (e.g. "mnist" or "cifar10")
      root: root directory to store or download the data
      img_size: final image size after resizing
      augment: whether to apply training-time augmentation
      seed: random seed used for the train/val split
      val_size: size of the validation set (if None, a default is used)

    Returns:
      train_ds: training dataset
      val_ds: validation dataset
      test_ds: test dataset
      num_classes: number of classes in the dataset
    """
    name = name.lower()

    # Default validation size: 5000 for both MNIST and CIFAR-10
    if val_size is None:
        val_size = 5000 if name == "cifar10" else 5000

    # Ensure the root directory exists
    os.makedirs(root, exist_ok=True)

    if name == "mnist":
        return get_mnist(root, img_size, augment, seed, val_size)
    elif name == "cifar10":
        return get_cifar10(root, img_size, augment, seed, val_size)
    else:
        raise ValueError(f"Unknown dataset: {name}")
