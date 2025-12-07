from typing import Tuple, Optional
import os
import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets as tvds

from .transforms import build_transforms

def _val_split(train_dataset: Dataset, val_size: int, seed: int) -> Tuple[Dataset, Dataset]:
    n = len(train_dataset)
    val_size = min(val_size, n)
    train_size = n - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], generator=g)
    return train_ds, val_ds

def get_mnist(root: str, img_size: int, augment: bool, seed: int, val_size: int = 5000):
    train_tfms, test_tfms = build_transforms("mnist", img_size, augment)
    train_full = tvds.MNIST(root=root, train=True, download=True, transform=train_tfms)
    test_ds    = tvds.MNIST(root=root, train=False, download=True, transform=test_tfms)
    # on refait une version val avec mêmes tfms que test pour l’évaluation
    train_full_eval = tvds.MNIST(root=root, train=True, download=True, transform=test_tfms)
    train_ds, val_ds = _val_split(train_full_eval, val_size, seed)
    # mais on remplace la partie train par la vraie version avec aug
    train_ds.dataset = train_full
    num_classes = 10
    return train_ds, val_ds, test_ds, num_classes

def get_cifar10(root: str, img_size: int, augment: bool, seed: int, val_size: int = 5000):
    train_tfms, test_tfms = build_transforms("cifar10", img_size, augment)
    train_full = tvds.CIFAR10(root=root, train=True, download=True, transform=train_tfms)
    test_ds    = tvds.CIFAR10(root=root, train=False, download=True, transform=test_tfms)
    # même astuce pour un split val stable
    train_full_eval = tvds.CIFAR10(root=root, train=True, download=True, transform=test_tfms)
    train_ds, val_ds = _val_split(train_full_eval, val_size, seed)
    train_ds.dataset = train_full
    num_classes = 10
    return train_ds, val_ds, test_ds, num_classes

def build_datasets(name: str, root: str, img_size: int, augment: bool, seed: int,
                   val_size: Optional[int] = None):
    name = name.lower()
    if val_size is None:
        val_size = 5000 if name == "cifar10" else 5000
    os.makedirs(root, exist_ok=True)
    if name == "mnist":
        return get_mnist(root, img_size, augment, seed, val_size)
    elif name == "cifar10":
        return get_cifar10(root, img_size, augment, seed, val_size)
    else:
        raise ValueError(f"dataset inconnu: {name}")
