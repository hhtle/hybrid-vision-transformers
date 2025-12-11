from typing import Dict
import torch
from torch.utils.data import DataLoader
from .datasets import build_datasets


def _worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with its own random seed.

    This makes data loading deterministic across runs:
    - we derive a worker-specific seed from torch.initial_seed()
    - we reset Python's random and NumPy's RNG with that seed

    Without this, different workers might use the same RNG state,
    which can break reproducibility when using data augmentation.
    """
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(cfg: Dict):
    """
    Build train/validation/test dataloaders given a configuration dictionary.

    Expected keys in cfg:
      - "name": dataset name (e.g. "mnist", "cifar10")
      - "root": root directory for caching/downloading data
      - "img_size": final image size after resizing
      - "augment": whether to apply data augmentation (train set)
      - "seed": random seed for deterministic splits
      - "val_size": number of validation samples
      - "batch_size": batch size for all dataloaders
      - "num_workers": number of DataLoader workers

    Returns:
      - train_loader: DataLoader for the training set
      - val_loader: DataLoader for the validation set
      - test_loader: DataLoader for the test set
      - meta: dictionary with basic dataset metadata:
          * num_classes
          * image_size
          * channels
          * train_len / val_len / test_len
    """
    # Basic dataset configuration
    name = cfg["name"]
    root = cfg.get("root", "./data/cache")
    img_size = int(cfg.get("img_size", 32))
    augment = bool(cfg.get("augment", True))
    seed = int(cfg.get("seed", 17092003))
    val_size = int(cfg.get("val_size", 5000))

    # Build the underlying datasets (train/val/test)
    train_ds, val_ds, test_ds, num_classes = build_datasets(
        name=name,
        root=root,
        img_size=img_size,
        augment=augment,
        seed=seed,
        val_size=val_size,
    )

    # Dataloader parameters
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))

    # Use pinned memory only if CUDA is available (helps host→GPU transfers)
    pin_memory = torch.cuda.is_available()

    # Keep workers alive between iterations to avoid re-forking them every epoch
    persistent_workers = num_workers > 0

    # Training dataloader: shuffle enabled, drop_last=True for stable batch sizes
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )

    # Validation dataloader: no shuffle
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )

    # Test dataloader: no shuffle
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )

    # Basic metadata describing the dataset and splits.
    # We fix "channels" to 3 so that MNIST is converted to 3-channel images,
    # which makes the model architecture uniform across MNIST and CIFAR-10.
    meta = {
        "num_classes": num_classes,
        "image_size": img_size,
        "channels": 3,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_loader.dataset),
    }

    return train_loader, val_loader, test_loader, meta
