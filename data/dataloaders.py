from typing import Dict
import torch
from torch.utils.data import DataLoader
from .datasets import build_datasets

def _worker_init_fn(worker_id):
    # rend les workers déterministes
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def build_dataloaders(cfg: Dict):
    name = cfg["name"]
    root = cfg.get("root", "./data/cache")
    img_size = int(cfg.get("img_size", 32))
    augment = bool(cfg.get("augment", True))
    seed = int(cfg.get("seed", 42))
    val_size = int(cfg.get("val_size", 5000))

    train_ds, val_ds, test_ds, num_classes = build_datasets(
        name=name, root=root, img_size=img_size, augment=augment, seed=seed, val_size=val_size
    )

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn, persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn, persistent_workers=persistent_workers
    )

    meta = {
        "num_classes": num_classes,
        "image_size": img_size,
        "channels": 3,  # on force 3 canaux pour unifier MNIST et CIFAR-10
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_loader.dataset),
    }
    return train_loader, val_loader, test_loader, meta
