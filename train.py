# load_cifar10.py
import yaml
from data import build_dataloaders

def load_cfg(path):
    # Load a YAML config file into a Python dict
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    cfg = load_cfg("configs/cifar10_baseline.yaml")

    # 2) Extract only the DataLoader-related fields
    dl_cfg = {
        "name": cfg["dataset"]["name"],
        "root": cfg["dataset"]["root"],
        "img_size": cfg["dataset"]["img_size"],
        "augment": cfg["dataset"].get("augment", True),
        "batch_size": cfg["optim"]["batch_size"],
        "num_workers": 4,
        "seed": cfg["misc"]["seed"],
        "val_size": 5000,
    }

    # 3) Build train/val/test DataLoaders
    train_loader, val_loader, test_loader, meta = build_dataloaders(dl_cfg)

    # 4) Compute the expected number of ViT tokens from img_size and patch
    patch = cfg["dataset"]["patch"]       # used by the model, not the DataLoader
    img_size = cfg["dataset"]["img_size"]
    assert img_size % patch == 0, "img_size must be divisible by patch"
    num_tokens = (img_size // patch) ** 2

    print("meta:", meta)
    print(f"img_size = {img_size}, patch = {patch}, expected tokens = {num_tokens}")

    # 5) Pull one batch to verify shapes and value range
    xb, yb = next(iter(train_loader))
    print("batch shapes:", xb.shape, yb.shape)  # expected: [B, 3, H, W], [B]
    print("dtype/range:", xb.dtype, f"[{xb.min().item():.3f}, {xb.max().item():.3f}]")
