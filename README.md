# Hybrid Vision Transformers with Performer Attention

This project implements small hybrid Vision Transformers combining regular self-attention and Performer attention.
We compare:

* regular ViT (all softmax attention)
* full-Performer models (ReLU and softmax-kernel variants)
* hybrid models with different layer patterns (Performer→Reg, Reg→Performer, intertwined)

on MNIST and CIFAR-10, and log accuracy, parameter counts, and training time.

---
## Project Report

For a detailed analysis of the methodology, mathematical derivations of the Performer and Hybrid mechanisms, and full experimental results, please refer to the final report.
* [Access to the Final Report (PDF)](https://github.com/hhtle/hybrid-vision-transformers/blob/paper/Hybrid_ViT_Final_Report.pdf)
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
│   ├── model_analysis.py      # single-run analysis (plots + summary per config)
│   ├── model_analysis_group.py# grouped analysis (multi-config plots + full tables)
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

* `models/hybrid_vit.py` contains:

  * the patch embedding module
  * regular multi-head self-attention
  * Performer attention (ReLU and softmax-kernel variants)
  * the `HybridPerformer` model and the training/evaluation loops

* `models/model_analysis_group.py` aggregates multiple runs to produce:

  * grouped comparison plots (multiple models on the same curves)
  * dataset-level summary tables in `outputs/full_analysis/`

---

## 3. Configuration files

Each experiment is defined by a YAML config in `configs/`.
A typical config contains:

* dataset settings (MNIST or CIFAR-10, image size, patch size, augmentation)
* model settings (`d_model`, `heads`, `mlp_ratio`, `depth`, `layers`, and Performer params)
* optimization settings (batch size, LR, weight decay, epochs)
* logging/output settings (`out_dir`, `save_every`)
* misc settings (seed)

---

## 4. Running experiments

The main entry point is `main.py`.

```bash
python3 main.py
```

The code will:

1. Load each YAML config listed in `HYBRID_CONFIGS`.
2. Build dataloaders.
3. Build the model according to `layers`.
4. Train and evaluate on validation after each epoch.
5. Save logs/checkpoints in `runs/<cfg_name>/`.

### Resuming from a saved checkpoint (`start_from_save`)

The training code supports resuming a run from a previously saved checkpoint.

* If `start_from_save=False` (default): training starts from epoch 1 with fresh weights.
* If `start_from_save=True`: the script looks inside `runs/<cfg_name>/` for a saved checkpoint (typically the latest `checkpoint_epoch_*.pt`), reloads:

  * model weights
  * optimizer state
  * last epoch index
    and continues training from the next epoch.

This is useful if:

* a run was interrupted (Colab timeout, crash),
* you want to extend training beyond the original number of epochs,
* you want to avoid restarting expensive runs.

(Implementation detail: the resume path is tied to the config name, since `runs/<cfg_name>/` is the canonical run folder.)

---

## 5. Post-training analysis (group plots + tables)

### Single-run analysis (per config)

If you want plots for one run:

```bash
python3 models/model_analysis.py
```

It reads `runs/<cfg_name>/metrics.csv` and saves plots into `outputs/<cfg_name>/`.

### Grouped analysis (multiple runs on the same plots)

To reproduce the figures and aggregated tables used in the report:

```bash
python3 models/model_analysis_group.py
```

This script:

* reads multiple `runs/<cfg_name>/metrics.csv`,
* generates grouped plots (e.g. depth comparisons, kernel comparisons, feature comparisons) into `outputs/<group_name>/`,
* writes dataset-level summary tables into:

  * `outputs/full_analysis/mnist.txt`
  * `outputs/full_analysis/cifar10.txt`
    (one line per config).

---

## 6. Outputs and logging

For each config file `configs/XYZ.yaml`, the code creates:

```text
runs/XYZ/
    metrics.csv
    number_of_parameters.txt
    checkpoint_epoch_10.pt
    checkpoint_epoch_20.pt
    ...
```

* `metrics.csv`: epoch index, epoch time, train loss/acc, val loss/acc
* `number_of_parameters.txt`: total trainable parameters
* `checkpoint_epoch_*.pt`: saved periodically according to `save_every`
