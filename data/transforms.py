from typing import Tuple
import torch
from torchvision import transforms as T

# stats classiques
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
MNIST_MEAN3  = (0.1307, 0.1307, 0.1307)
MNIST_STD3   = (0.3081, 0.3081, 0.3081)

class To3Channels(torch.nn.Module):
    def forward(self, x):
        # x: [1,H,W] -> [3,H,W]
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

def build_transforms(name: str, img_size: int, augment: bool) -> Tuple[T.Compose, T.Compose]:
    name = name.lower()
    if name == "cifar10":
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
        raise ValueError(f"dataset inconnu: {name}")

    if not augment:
        # retirer les aug fortes pour l’entraînement si demandé
        if name == "cifar10":
            train_tfms = test_tfms

    return T.Compose(train_tfms), T.Compose(test_tfms)
