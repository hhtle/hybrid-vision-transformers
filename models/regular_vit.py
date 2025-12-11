import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time

from data.dataloaders import build_dataloaders

from tqdm.auto import tqdm

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
        "img_size": cfg["dataset"]["img_size"],
        "augment": cfg["dataset"].get("augment", True),
        "batch_size": cfg["optim"]["batch_size"],
        "num_workers": max(4, os.cpu_count() - 4),
        "seed": cfg["misc"]["seed"],
        "val_size": 5000,  # you can move this to the YAML if you want
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
    """
    Split image into non-overlapping patches and embed them.

    Important: this is where the convolution lives.
    The DataLoader just returns raw images [B, C, H, W];
    the convolution here creates patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel_size=stride=patch_size creates image patches
        # and maps them to an embedding dimension.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        x: tensor of shape [B, C, H, W]
        returns: tensor of shape [B, n_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H', W'] where H' = W' = img_size / patch_size
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention layer.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Single linear layer to produce Q, K and V at once
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, N, 3 * C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention weights to values
        x = attn @ v  # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Final projection
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Position-wise feed-forward network (MLP) used inside the Transformer block.
    """
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
    """
    Single Transformer encoder block:
    LayerNorm -> Self-Attention -> Residual
    LayerNorm -> MLP           -> Residual
    """
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
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model with:
      - Patch embedding via convolution
      - Learnable class token
      - Learnable positional embeddings
      - A stack of Transformer encoder blocks
      - A classification head on top of the class token
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Learnable class token (prepended to patch embeddings)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable positional embeddings for [CLS] + all patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Final layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights following a standard ViT-style scheme.
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
        returns: logits of shape [B, num_classes]
        """
        B = x.shape[0]

        # Embed patches with convolutional patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, n_patches + 1, embed_dim]

        # Add positional embeddings and dropout
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Classification head uses only the class token
        x = self.head(x[:, 0])  # [B, num_classes]
        return x


# ============================
# Training / Evaluation loops
# ============================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None, epochs=None):
    """
    Train the model for a single epoch.

    Args:
        model: Vision Transformer model
        train_loader: DataLoader for the training set
        criterion: loss function (e.g. CrossEntropyLoss)
        optimizer: optimizer (e.g. AdamW)
        device: torch device ("cpu" or "cuda")
        epoch (int, optional): current epoch index (1-based), for display
        epochs (int, optional): total number of epochs, for display

    Returns:
        avg_loss (float): average loss over the epoch
        accuracy (float): accuracy in percent over the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Description of the progress bar for this epoch
    if epoch is not None and epochs is not None:
        desc = f"Epoch {epoch}/{epochs} [train]"
    else:
        desc = "Training"

    # tqdm over the training dataloader so you see progress for each epoch
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
    
    Returns:
        avg_loss (float)
        accuracy (float in percent)
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

def train_vision_transformer_from_cfg(cfg_path: str, device=None):
    """
    Train a Vision Transformer using a YAML configuration file.

    The config is expected to have the following structure (example):

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

        optim:
          batch_size: 128
          lr: 3e-4
          weight_decay: 0.05
          epochs: 100

        misc:
          seed: 17092003

    All hyperparameters are taken from the config file,
    and the data is loaded using your custom build_dataloaders system.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(cfg_path)

    print("\n" + "=" * 60)
    print(f"Training Vision Transformer from config: {cfg_path}")
    print("=" * 60 + "\n")
    print(f"Using device: {device}" + "\n")

    # Build dataloaders from config
    train_loader, val_loader, test_loader, meta = build_dataloaders_from_cfg(cfg)

    # Peek one batch to infer actual input shape
    xb, yb = next(iter(train_loader))
    print("\nSample batch shapes from train_loader:", xb.shape, yb.shape)
    # xb: [B, C, H, W]
    actual_in_channels = xb.shape[1] # we force 3 channels in dataloader for MNIST/CIFAR10 unification

    # Read dataset / model parameters from config
    img_size = cfg["dataset"]["img_size"]
    patch_size = cfg["dataset"]["patch"]
    embed_dim = cfg["model"]["d_model"]
    depth = cfg["model"]["depth"]
    num_heads = cfg["model"]["heads"]
    mlp_ratio = cfg["model"]["mlp_ratio"]

    # Ensure patch_size divides img_size
    assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

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

    # Logging / checkpoint configuration
    log_cfg = cfg.get("log", {})
    base_out_dir = log_cfg.get("out_dir", "./runs")
    save_every = int(log_cfg.get("save_every", 0))  # 0 = no checkpointing

    # Use the config file name (without extension) to name the run directory
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_dir = os.path.join(base_out_dir, config_name)
    os.makedirs(run_dir, exist_ok=True)

    # Path for the per-epoch metrics file
    metrics_path = os.path.join(run_dir, "metrics.csv")

    # If the file does not exist yet, create it with a header
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,epoch_time_sec,train_loss,train_acc,val_loss,val_acc\n")

    # Create ViT model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("\nModel:")
    print(f"  embed_dim     : {embed_dim}")
    print(f"  depth         : {depth}")
    print(f"  num_heads     : {num_heads}")
    print(f"  mlp_ratio     : {mlp_ratio}")
    print(f"  total params  : {num_params:,}")

    # Optimization hyperparameters from config
    epochs = cfg["optim"]["epochs"]
    lr = cfg["optim"]["lr"]
    weight_decay = cfg["optim"]["weight_decay"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\n" + "=" * 60)
    print(f"Training for {epochs} epochs...")
    print("=" * 60 + "\n")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Measure epoch wall-clock time
        start_time = time.time()

        # Train one epoch (with tqdm if you kept it)
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            epochs=epochs,
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs:03d}: "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f}, Acc {val_acc:.2f}% | "
            f"Time {epoch_time:.2f}s"
        )

        # Append metrics to CSV file so it is updated every epoch
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

        # Save a checkpoint every `save_every` epochs (if enabled)
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
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint to {ckpt_path}")


    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60 + "\n")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.2f}%")
    print(f"Best Val Acc   : {best_val_acc:.2f}% (epoch {best_epoch})")

    return model


if __name__ == "__main__":
    # Example: train using your MNIST and CIFAR-10 config files
    # Adapt paths if needed.
    model_mnist = train_vision_transformer_from_cfg("configs/mnist_baseline.yaml")

    print("\n\n")

    model_cifar = train_vision_transformer_from_cfg("configs/cifar10_baseline.yaml")
