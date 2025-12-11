import os
import time
import math
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from data.dataloaders import build_dataloaders


# ============================
# Config / Data utilities
# ============================

def load_cfg(path: str):
    """
    Load a YAML configuration file into a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders_from_cfg(cfg):
    """
    Build train/val/test dataloaders from a config dictionary.

    This uses your custom build_dataloaders(...) and the fields
    defined under the "dataset", "optim" and "misc" sections.
    """
    dl_cfg = {
        "name": cfg["dataset"]["name"],
        "root": cfg["dataset"]["root"],
        "img_size": int(cfg["dataset"]["img_size"]),
        "augment": cfg["dataset"].get("augment", True),
        "batch_size": int(cfg["optim"]["batch_size"]),
        "num_workers": max(4, os.cpu_count() - 4),
        "seed": int(cfg["misc"]["seed"]),
        "val_size": 5000,  # you can move this to YAML later if needed
    }

    print("DataLoader config:")
    for key, value in dl_cfg.items():
        print(f"  {key}: {value}")

    train_loader, val_loader, test_loader, meta = build_dataloaders(dl_cfg)
    return train_loader, val_loader, test_loader, meta


# ============================
# Model components
# ============================

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them using a convolution."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel_size=stride=patch_size creates non-overlapping patches
        # and maps them to an embedding dimension.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, n_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H', W'] where H' = W' = img_size / patch_size
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Regular multi-head self-attention."""
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class PerformerAttentionReLU(nn.Module):
    """
    Performer attention with ReLU kernel (variant 'relu').

    Uses random features to obtain a linear-time approximation
    of softmax self-attention.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_random_features=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_random_features = num_random_features

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Random features for ReLU kernel
        self.register_buffer("random_features", torch.randn(num_random_features, self.head_dim))
        self.random_features = nn.functional.normalize(self.random_features, p=2, dim=1)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale q, k
        scale = self.head_dim ** -0.5
        q = q * scale
        k = k * scale

        # Feature map φ(x) = ReLU(x W^T)
        # (B, H, N, D) @ (D, m) -> (B, H, N, m)
        phi_q = torch.relu(q @ self.random_features.T)
        phi_k = torch.relu(k @ self.random_features.T)

        # Linear attention (Performer-style) in O(N) per head
        # kv: (B, H, m, D)
        kv = phi_k.transpose(-2, -1) @ v

        # φ(Q) (φ(K)^T V): (B, H, N, m) @ (B, H, m, D) -> (B, H, N, D)
        attn_output = (phi_q @ kv) / self.num_random_features

        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class PerformerAttentionSoftmax(nn.Module):
    """
    Performer attention with softmax kernel approximation using
    positive random features (variant 'softmax').
    """
    def __init__(self, embed_dim=768, num_heads=8, num_random_features=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_random_features = num_random_features

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Random features for softmax kernel approximation
        self.register_buffer("random_features", torch.randn(num_random_features, self.head_dim))
        self.random_features = nn.functional.normalize(self.random_features, p=2, dim=1)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        q = q * scale
        k = k * scale

        # Squared L2 norm: used in exponent
        q_norm = (q ** 2).sum(dim=-1, keepdim=True) / 2
        k_norm = (k ** 2).sum(dim=-1, keepdim=True) / 2

        # Positive random features: φ(x) = exp(xW^T - ||x||^2 / 2)
        phi_q = torch.exp(q @ self.random_features.T - q_norm)  # [B, H, N, m]
        phi_k = torch.exp(k @ self.random_features.T - k_norm)  # [B, H, N, m]

        # Linearized softmax attention
        # kv: (B, H, m, D)
        kv = phi_k.transpose(-2, -1) @ v
        numerator = phi_q @ kv  # [B, H, N, D]

        # Denominator: φ(Q) @ (φ(K)^T 1)
        all_ones = torch.ones(B, self.num_heads, N, 1, device=x.device)
        k_sum = phi_k.transpose(-2, -1) @ all_ones  # [B, H, m, 1]
        denominator = phi_q @ k_sum                 # [B, H, N, 1]

        attn_output = numerator / torch.clamp(denominator, min=1e-6)

        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with regular multi-head self-attention."""
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PerformerBlock(nn.Module):
    """
    Transformer block with Performer attention (either 'relu' or 'softmax' variant).
    """
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        num_random_features=256,
        mlp_ratio=4.0,
        dropout=0.1,
        performer_variant="relu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        if performer_variant == "relu":
            self.attn = PerformerAttentionReLU(embed_dim, num_heads, num_random_features, dropout)
        elif performer_variant == "softmax":
            self.attn = PerformerAttentionSoftmax(embed_dim, num_heads, num_random_features, dropout)
        else:
            raise ValueError(f"Unknown performer_variant: {performer_variant}")

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridPerformer(nn.Module):
    """
    Hybrid Transformer where each layer type is specified explicitly in a list.

    The `layers` list controls the stack of blocks, e.g.:
        layers = ["Reg", "Perf", "Reg", "Perf"]

    - "Reg"  -> regular Transformer block with softmax self-attention
    - "Perf" -> Performer block (ReLU or softmax variant, controlled by performer_variant)

    This gives you full control from the YAML config over which layers are regular
    and which are Performer-based.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        layers=None,           # list like ["Reg", "Perf", ...]
        num_heads=12,
        num_random_features=256,
        mlp_ratio=4.0,
        dropout=0.1,
        performer_variant="relu",
    ):
        super().__init__()

        if layers is None:
            raise ValueError("You must provide a list of layer types (e.g. ['Reg', 'Perf', ...]).")

        self.layers = layers
        depth = len(layers)

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Build the sequence of blocks according to `layers`
        self.blocks = nn.ModuleList()
        for i, layer_type in enumerate(layers):
            if layer_type == "Reg":
                # Regular Transformer block with softmax attention
                self.blocks.append(
                    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                )
            elif layer_type == "Perf":
                # Performer-based block (ReLU or softmax kernel variant)
                self.blocks.append(
                    PerformerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_random_features=num_random_features,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        performer_variant=performer_variant,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown layer type '{layer_type}' at position {i}. "
                    f"Expected 'Reg' or 'Perf'."
                )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.performer_variant = performer_variant

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Standard ViT-style initialization for linear and LayerNorm layers.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, num_classes]
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, 1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, n_patches + 1, embed_dim]

        # Add positional embeddings and dropout
        x = x + self.pos_embed
        x = self.dropout(x)

        # Run through the stack of blocks defined by `layers`
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 0])  # use class token
        return x


# ============================
# Training / Evaluation loops
# ============================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None, epochs=None):
    """
    Train the model for a single epoch.

    Adds a tqdm progress bar over training batches.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if epoch is not None and epochs is not None:
        desc = f"Epoch {epoch}/{epochs} [train]"
    else:
        desc = "Training"

    for images, labels in tqdm(train_loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a given dataloader.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(data_loader)
    return avg_loss, accuracy


# ============================
# High-level training function
# ============================

def train_hybrid_performer_from_cfg(
    cfg_path: str,
    device=None,
):
    """
    Train a HybridPerformer model using a YAML configuration file and
    your custom dataloaders.

    The config is assumed to contain something like:

        experiment: cifar10_baseline
        dataset:
          name: cifar10
          root: ./data/cache
          img_size: 32
          patch: 4
          augment: true
        model:
          d_model: 256
          heads: 4
          mlp_ratio: 2.0
          depth: 8
          layers: ["Reg", "Perf", "Reg", "Perf", "Reg", "Perf", "Reg", "Perf"]
          performer:
            kind: "favor+"
            m: 64      # number of random features for Performer
        optim:
          batch_size: 128
          lr: 0.0003
          weight_decay: 0.05
          epochs: 100
          warmup_epochs: 5
          grad_clip: 1.0
        log:
          out_dir: ./runs
          save_every: 10
        misc:
          seed: 17092003
          fp16: true

    The `model.layers` list fully controls which layers are regular
    ("Reg") and which are Performer ("Perf").
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(cfg_path)

    # Read performer variant directly from config (default: 'softmax')
    performer_cfg = cfg["model"].get("performer", {})
    performer_variant = performer_cfg.get("variant", "softmax")  # 'relu' or 'softmax'


    print("\n" + "=" * 70)
    print(f"Hybrid Performer (variant={performer_variant}) from config: {cfg_path}")
    print("=" * 70 + "\n")
    print(f"Using device: {device}" + "\n")

    # Logging / checkpoint configuration
    log_cfg = cfg.get("log", {})
    base_out_dir = log_cfg.get("out_dir", "./runs")
    save_every = int(log_cfg.get("save_every", 0))

    # Use the config name for the run directory
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_name = f"{config_name}"
    run_dir = os.path.join(base_out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Metrics CSV file (one line per epoch, updated as we go)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,epoch_time_sec,train_loss,train_acc,val_loss,val_acc\n")

    # Build dataloaders
    train_loader, val_loader, test_loader, meta = build_dataloaders_from_cfg(cfg)

    # Peek one batch to infer actual input channels
    xb, yb = next(iter(train_loader))
    print("\nSample batch shapes from train_loader:", xb.shape, yb.shape)
    actual_in_channels = xb.shape[1]

    # Model hyperparameters from config
    img_size = int(cfg["dataset"]["img_size"])
    patch_size = int(cfg["dataset"]["patch"])
    embed_dim = int(cfg["model"]["d_model"])
    num_heads = int(cfg["model"]["heads"])
    mlp_ratio = float(cfg["model"]["mlp_ratio"])
    layers = cfg["model"]["layers"]  # e.g. ["Reg", "Perf", "Reg", "Perf"]
    depth = int(cfg["model"]["depth"])

    # Consistency check: depth should match len(layers)
    if len(layers) != depth:
        print(
            f"Warning: model.depth={depth} but len(model.layers)={len(layers)}. "
            f"Using len(layers)={len(layers)} as the actual depth."
        )
        depth = len(layers)

    perf_cfg = cfg["model"].get("performer", {})
    num_random_features = int(perf_cfg.get("m", 64))

    assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch ({patch_size})"

    in_channels = actual_in_channels
    num_classes = meta.get("num_classes", int(yb.max().item() + 1))

    print("\nDataset info (from config and meta):")
    print(f"  dataset name      : {cfg['dataset']['name']}")
    print(f"  image size        : {img_size} x {img_size}")
    print(f"  patch size        : {patch_size} x {patch_size}")
    print(f"  number of patches : {(img_size // patch_size) ** 2}")
    print(f"  in_channels       : {in_channels}")
    print(f"  num_classes       : {num_classes}")
    print(f"  train samples     : {len(train_loader.dataset)}")
    print(f"  val samples       : {len(val_loader.dataset)}")
    print(f"  test samples      : {len(test_loader.dataset)}")

    # Create HybridPerformer model using the explicit `layers` list
    model = HybridPerformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        layers=layers,
        num_heads=num_heads,
        num_random_features=num_random_features,
        mlp_ratio=mlp_ratio,
        dropout=0.1,
        performer_variant=performer_variant,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("\nModel:")
    print(f"  embed_dim        : {embed_dim}")
    print(f"  layers            : {layers}")
    print(f"  depth (len layers): {len(layers)}")
    print(f"  num_heads        : {num_heads}")
    print(f"  mlp_ratio        : {mlp_ratio}")
    print(f"  num_random_feat  : {num_random_features}")
    print(f"  performer_variant: {performer_variant}")
    print(f"  total params     : {num_params:,}")

    # Optimization hyperparameters
    epochs = int(cfg["optim"]["epochs"])
    lr = float(cfg["optim"]["lr"])
    weight_decay = float(cfg["optim"]["weight_decay"])

    print("\nOptimization:")
    print(f"  epochs        : {epochs}")
    print(f"  learning rate : {lr}")
    print(f"  weight decay  : {weight_decay}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\n" + "=" * 70)
    print(f"Training for {epochs} epochs...")
    print("=" * 70)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            epochs=epochs,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs:03d}: "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f}, Acc {val_acc:.2f}% | "
            f"Time {epoch_time:.2f}s"
        )

        # Append metrics to CSV (updated every epoch)
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{epoch_time:.4f},"
                f"{train_loss:.6f},{train_acc:.4f},"
                f"{val_loss:.6f},{val_acc:.4f}\n"
            )

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        # Save checkpoint every `save_every` epochs
        if save_every > 0 and epoch % save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "layers": layers,
                    "performer_variant": performer_variant,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint to {ckpt_path}")

    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.2f}%")
    print(f"Best Val Acc   : {best_val_acc:.2f}% (epoch {best_epoch})")

    return model


if __name__ == "__main__":
    cfg_path_mnist = "configs/mnist_baseline.yaml"

    model_hybrid = train_hybrid_performer_from_cfg(cfg_path_mnist)

