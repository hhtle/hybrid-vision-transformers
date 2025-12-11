# Hybrid Vision Transformers with Performer Attention

This project implements small hybrid Vision Transformers combining regular self-attention and Performer attention.
We compare:

* regular ViT (all softmax attention)
* full-Performer models (ReLU and softmax-kernel variants)
* hybrid models with different layer patterns (Performer→Reg, Reg→Performer, intertwined)

on MNIST and CIFAR-10, and log accuracy, parameter counts, and training time.

---

## 1. Installation and dependencies

To reproduce the environment on another machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Repository structure

The repository is organized as follows:

```text
.
├── main.py
├── data.ipynb                 # quick notebook to inspect how data and configs are loaded
├── models/
│   ├── regular_vit.py         # original regular ViT implementation (kept for reference)
│   ├── model_analysis.py      # scripts to generate graphs from runs results
│   └── hybrid_vit.py          # main Hybrid ViT + Performer implementation
├── configs/
│   ├── mnist_baseline_*.yaml
│   ├── mnist_perf_*.yaml
│   ├── mnist_reg_perf_*.yaml
│   ├── mnist_perf_reg_*.yaml
│   ├── mnist_interwined_*.yaml
│   ├── cifar10_baseline_*.yaml
│   ├── cifar10_perf_*.yaml
│   ├── cifar10_reg_perf_*.yaml
│   ├── cifar10_perf_reg_*.yaml
│   └── cifar10_interwined_*.yaml
├── data/
│   ├── dataloaders.py         # build train/val/test DataLoaders from a config dict
│   ├── datasets.py            # build MNIST / CIFAR-10 Dataset objects and splits
│   ├── transforms.py          # torchvision transforms and preprocessing pipelines
│   └── cache/                 # downloaded datasets are stored here
├── runs/
│   └── ...                    # created automatically on first run (metrics, checkpoints, etc.)
├── outputs/
│   └── ...                    # created for aggregated metrics, analysis scripts, plots
└── README.md
```

Notes:

Here’s an updated version:

* `models/hybrid_vit.py` contains:

  * the `PatchEmbedding` module
  * regular multi-head self-attention
  * Performer attention (ReLU and softmax-kernel variants)
  * the `HybridPerformer` model and the main training/evaluation loops

* `data/dataloaders.py` builds the train/val/test `DataLoader`s from a small config dictionary (dataset name, image size, augmentation flag, batch size, number of workers, etc.).

* `data/datasets.py` constructs the underlying `Dataset` objects for MNIST and CIFAR-10, including a deterministic train/validation split.

* `data/transforms.py` defines the torchvision preprocessing pipelines (resize, data augmentation, normalization, conversion of MNIST to 3 channels, …).

* `data.ipynb` (at the root of the repository) is a small notebook to quickly inspect how configs are loaded, how `build_dataloaders` works, and what the resulting batches look like before moving on to full experiments.

* `runs/` and `outputs/` are created automatically when you run the code:

  * each experiment gets its own subdirectory in `runs/` (one per config file)
  * metrics, parameter counts, and checkpoints are stored there
  * `outputs/` can be used by analysis scripts/notebooks to save aggregated results, tables, and plots.

---

## 3. Configuration files

Each experiment is defined by a YAML config in `configs/`.
A typical config contains:

* dataset settings (MNIST or CIFAR-10, image size, patch size, augmentation)

* model settings:

  * `d_model`, `heads`, `mlp_ratio`, `depth`
  * `layers`: list of "Reg" or "Perf" defining the stack of blocks
  * `performer.variant`: "softmax" or "relu"
  * `performer.m`: number of random features

* optimization settings (batch size, learning rate, weight decay, number of epochs)

* logging/output settings (`out_dir`, `save_every`)

* misc settings (random seed)

Example excerpt:

```yaml
experiment: cifar10_perf_d=2_m=64_variant=softmax
dataset:
  name: cifar10
  root: ./data/cache
  img_size: 32
  patch: 4
  augment: true
model:
  d_model: 64
  heads: 4
  mlp_ratio: 2.0
  depth: 6
  layers: ["Perf", "Perf", "Perf", "Perf", "Perf", "Perf"]
  performer:
    variant: "softmax"
    kind: "favor+"
    m: 64
optim:
  batch_size: 128
  lr: 0.0003
  weight_decay: 0.05
  epochs: 40
log:
  out_dir: ./runs
  save_every: 5
misc:
  seed: 17092003
```

---

## 4. Running experiments

The main entry point is `main.py`.
It imports the high-level training function from `models/hybrid_vit.py` and specifies which configs to run.

Example structure of `main.py`:

```python
import torch
from models.hybrid_vit import train_hybrid_performer_from_cfg

HYBRID_CONFIGS = [
    "configs/mnist_baseline_d=6.yaml",
    "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
    "configs/cifar10_baseline_d=2.yaml",
    "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
    # add or remove any config file you want to run
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for cfg_path in HYBRID_CONFIGS:
        train_hybrid_performer_from_cfg(cfg_path, device=device)

if __name__ == "__main__":
    main()
```

To launch the trainings, simply activate your environment and run:

```bash
python3 main.py
```

The code will:

1. Load each YAML config listed in `HYBRID_CONFIGS`.
2. Build the corresponding dataloaders (MNIST or CIFAR-10).
3. Construct the HybridPerformer model according to the `layers` and `performer` settings.
4. Train the model for the specified number of epochs.
5. Evaluate on the validation set after each epoch.
6. Optionally save checkpoints every `save_every` epochs.

---

## 5. Outputs and logging

For each config file `configs/XYZ.yaml`, the code creates a run directory:

```text
runs/XYZ/
    metrics.csv
    number_of_parameters.txt
    checkpoint_epoch_10.pt
    checkpoint_epoch_20.pt
    ...
```

* `metrics.csv`: one row per epoch, containing:

  * epoch index
  * epoch time
  * training loss and accuracy
  * validation loss and accuracy

* `number_of_parameters.txt`: total number of trainable parameters for that model.

* `checkpoint_epoch_*.pt`: model and optimizer state dicts saved periodically according to `save_every` in the config.

The `outputs/` directory is intended for higher-level analysis artifacts, for example:

* aggregated CSV files comparing models
* plots of accuracy vs. number of random features m
* plots of accuracy vs. number of parameters
* training/inference time comparisons

You can generate these from your own analysis scripts or notebooks by reading the CSV files and parameter counts from the `runs/` directory and writing your results into `outputs/`.

---

With this setup, you can:

* plug different configs into `HYBRID_CONFIGS` in `main.py`
* run `python3 main.py`
* then compare the models using the logs in `runs/` and any additional plots in `outputs/`.
