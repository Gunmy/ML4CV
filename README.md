# Humpback Whale Identification — ML4CV Project

**Course**: Machine Learning for Computer Vision (91266)  
**Task**: [Kaggle Whale Identification Playground](https://www.kaggle.com/competitions/whale-categorization-playground)

---

## Quick Start

```bash
# 1. Install PyTorch with GPU support (pick your CUDA version — check with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Place Kaggle data in data/
#    data/train.csv
#    data/train/00022e1a.jpg, ...

# 5. Open the notebook
jupyter notebook 01_classification_baseline.ipynb
```

---

## Project Structure

```
whale_project/
│
├── 01_classification_baseline.ipynb  # ← MAIN NOTEBOOK — run this
│
├── config.py            # Experiment configuration
├── dataset.py           # Data loading and augmentation
├── models.py            # Neural network architecture
├── losses.py            # Loss functions (Focal Loss)
├── training.py          # Training loop
├── evaluation.py        # Validation metrics
├── experiment.py        # Checkpointing and experiment tracking
├── visualization.py     # Plotting utilities
│
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── data/                # YOUR KAGGLE DATA (not included)
│   ├── train.csv
│   └── train/
│       └── *.jpg
│
└── experiments/         # AUTO-GENERATED during training
    ├── index.json       # Summary of all experiments
    └── <experiment_id>/
        ├── config.json
        ├── metrics.json
        ├── log.txt
        └── checkpoints/
            ├── best.pth
            ├── latest.pth
            └── epoch_*.pth
```

---

## What Each File Does

### `config.py` — Experiment Configuration

A single `ExperimentConfig` dataclass that holds **every** hyperparameter:
backbone choice, image size, learning rates, loss type, augmentation settings,
freeze strategy, early stopping, etc. Changing one config object and re-running
`train()` creates an entirely new experiment with its own checkpoints and logs.
Each config auto-generates a unique experiment ID from a hash of its contents
plus a timestamp, so experiments never overwrite each other.

### `dataset.py` — Data Pipeline

Handles three things:

1. **Splitting**: Loads `train.csv` and uses a hybrid split strategy.
   Classes with enough samples are stratified across train/val/test.
   Ultra-rare classes are kept in training so split constraints remain valid
   and the classifier still learns from rare identities.

2. **Augmentation**: Whale-specific transforms that simulate realistic viewing
   variation (lighting, angle, distance) without destroying identity cues.
   Notably, we **do not** use horizontal flip (fluke notch patterns are
   asymmetric) or random erasing (could delete identifying marks).

3. **new_whale handling**: The `new_whale` label is excluded from training
   (you can't teach a classifier what "unknown" looks like) but kept in
   validation so we can measure open-set detection performance.

### `models.py` — Architecture

`WhaleClassifier` wraps any [timm](https://github.com/huggingface/pytorch-image-models)
backbone with a custom head:

```
backbone → global_pool → embedding_layer (512-d) → classifier
```

The explicit embedding layer serves double duty: it produces the vectors
we'll use for metric learning and k-NN retrieval later, and it decouples
the backbone output dimension from the classifier so you can swap backbones
freely.

Also provides:
- `freeze_backbone()` / `unfreeze_backbone()` — toggle what trains
- `set_bn_eval()` — keep BatchNorm in eval mode (critical for small batches)
- `build_optimizer()` — differential learning rates (backbone vs. head)
- `build_scheduler()` — cosine annealing with warmup, or plateau

### `losses.py` — Focal Loss

Implements Focal Loss from scratch:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

When γ=0 this is exactly cross-entropy, so it can never perform worse.
When γ=2 (default), easy/confident examples contribute less to the gradient,
focusing training on hard and rare classes. Combined with inverse-frequency
class weights (α) to handle the extreme long-tail distribution.

### `training.py` — Training Loop

The training procedure has two phases:

- **Phase 1** (epochs 0–2): Backbone frozen, only the head trains. This
  prevents the pretrained features from being corrupted by random gradients
  from the untrained head.

- **Phase 2** (epochs 3+): Full fine-tuning with the backbone at 10× lower
  learning rate than the head.

Features:
- **AMP** (Automatic Mixed Precision) — ~50% memory reduction, fits B5 in 8GB
- **Gradient accumulation** — micro-batch of 8, accumulated 4×, effective batch 32
- **Gradient clipping** — prevents exploding gradients at phase transitions
- **Cosine annealing with linear warmup** — smooth LR decay
- **Seed control** — reproducible results

### `evaluation.py` — Validation Metrics

Computes three separate metrics instead of one misleading accuracy number:

1. **Known-whale accuracy**: Of validation samples whose true identity is in
   the training set, how many does the model correctly identify? This tells
   you if the model is learning to discriminate between individuals.

2. **New-whale detection rate**: Of validation samples that are truly `new_whale`,
   what fraction does the model flag as unknown (confidence below threshold)?
   This tells you if confidence thresholding works for open-set recognition.

3. **Overall accuracy**: Combined, where a correct prediction is either
   identifying the right known whale OR correctly flagging a new_whale.

### `experiment.py` — Experiment Management

Automatic bookkeeping:
- Creates a directory per experiment with config, metrics, logs, checkpoints
- **Crash-safe metrics**: saved to disk after every epoch, not just at the end
- **Checkpoint management**: keeps `best.pth`, `latest.pth`, and top-K epoch snapshots
- **Global index** (`experiments/index.json`): one-line summary of every experiment
- **Resume support**: pass `resume_from="<experiment_id>"` to `train()` to continue

### `visualization.py` — Plotting

- `plot_training_history()` — loss, accuracy, new-whale detection, and LR curves
  for a single experiment, with automatic freeze/unfreeze boundary marker
- `compare_experiments()` — overlay a metric across multiple experiments
- `list_experiments()` — print a table of all runs with their best results

---

## Running Different Experiments

To try a different setup, just create a new config and call `train()`:

```python
config_v2 = ExperimentConfig(
    experiment_name="convnext_tiny_comparison",
    backbone="convnext_tiny",
    image_size=(384, 384),
    batch_size=12,
    freeze_bn=False,   # ConvNeXt uses LayerNorm, not BatchNorm
    # ... everything else uses defaults
)
manager = train(config_v2, data, device)
```

Previous experiments are untouched. Compare with:
```python
compare_experiments("experiments", metric="val_known_acc")
```

---

## Resuming Training

If your kernel crashes or you want to continue for more epochs:

```python
manager = train(config, data, device, resume_from="20260402_120000_baseline_abc123")
```

The experiment ID is printed at the start of training and listed in
`experiments/index.json`.

---

## Hardware Requirements

- **GPU**: 8GB VRAM minimum (tested with EfficientNet-B5 at 456px, batch size 8, AMP on)
- **RAM**: 16GB recommended
- **Disk**: ~10GB for Kaggle data + checkpoints
