"""
Experiment configuration for Humpback Whale Identification.
All hyperparameters live here — change this one object to run a new experiment.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import hashlib
import json
import time


@dataclass
class ExperimentConfig:
    """Complete configuration for a single training run."""

    # ── Experiment metadata ──────────────────────────────────────────────
    experiment_name: str = ""          # Human-readable label (e.g. "effnetb5_focal_freeze3")
    description: str = ""              # Free-form notes about this run
    seed: int = 42

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "data"             # Root containing train.csv and train/ images
    experiments_root: str = "experiments"

    # ── Data ─────────────────────────────────────────────────────────────
    test_split: float = 0.1
    val_split: float = 0.1
    min_samples_per_class: int = 1     # Minimum images a class needs to be included
    stratified_split: bool = True      # Use stratified splitting

    # ── Model ────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b5"  # Any timm model name
    image_size: Tuple[int, int] = (456, 456)
    pretrained: bool = True
    embedding_dim: int = 512           # Dimension of the embedding before the classifier head
    use_gem_pooling: bool = False      # Generalized Mean Pooling (useful for retrieval)

    # ── Freeze / Unfreeze strategy ───────────────────────────────────────
    freeze_backbone_epochs: int = 3    # Number of epochs to train only the head
    freeze_bn: bool = True             # Keep BN in eval mode during fine-tuning (critical for small batches)

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 12                # Micro-batch that fits in VRAM
    accumulation_steps: int = 2        # Effective batch = batch_size * accumulation_steps = 24
    num_workers: int = 2
    use_amp: bool = True               # Automatic Mixed Precision

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer: str = "adamw"           # "adamw" or "sgd"
    lr_head: float = 1e-3              # Learning rate for the classification head
    lr_backbone: float = 1e-4          # Learning rate for the backbone (used after unfreeze)
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0         # Gradient clipping

    # ── Scheduler ────────────────────────────────────────────────────────
    scheduler: str = "cosine"          # "cosine", "plateau", or "none"
    warmup_epochs: int = 1             # Linear warmup before main schedule
    cosine_eta_min: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 3

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_type: str = "focal"           # "focal", "ce", "triplet", or "arcface_ce"
    focal_gamma: float = 2.0
    focal_alpha: Optional[str] = "class_weights"  # "class_weights", "balanced", or None
    label_smoothing: float = 0.1
    use_class_weights: bool = True     # Apply inverse-frequency class weights

    # ── New whale handling ───────────────────────────────────────────────
    include_new_whale_in_val: bool = True   # Track new_whale detection metrics
    confidence_threshold: float = 0.5       # Below this → predict new_whale

    # ── Augmentation ─────────────────────────────────────────────────────
    aug_random_resized_crop: bool = True
    aug_crop_scale_min: float = 0.85
    aug_crop_scale_max: float = 1.0
    aug_rotation_degrees: int = 10
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_gaussian_blur: bool = True
    aug_blur_sigma_min: float = 0.1
    aug_blur_sigma_max: float = 1.0

    # ── Early stopping ───────────────────────────────────────────────────
    early_stopping_patience: int = 8   # Stop if no improvement for this many epochs
    early_stopping_metric: str = "val_known_acc"  # Metric to monitor

    # ── Checkpointing ────────────────────────────────────────────────────
    save_every_n_epochs: int = 1       # Save checkpoint every N epochs
    keep_top_k_checkpoints: int = 3    # Only keep the K best checkpoints + latest
    auto_resume: bool = True           # Automatically resume from latest checkpoint if available

    # ── Metric Learning (defaults preserve baseline classification behavior) ─
    head_type: str = "linear"          # "linear" (baseline), "arcface", or "none" (embedding only)

    # ArcFace parameters (only used when head_type="arcface")
    arcface_scale: float = 30.0        # Scale factor s — controls logit magnitude
    arcface_margin: float = 0.5        # Angular margin m — pushes classes apart (radians)

    # Triplet loss parameters (only used when loss_type="triplet")
    triplet_margin: float = 0.3        # Margin in the triplet loss hinge
    triplet_mining: str = "semi_hard"  # "semi_hard", "hard", or "all"

    # PK sampling for metric learning (only used when pk_sampling=True)
    pk_sampling: bool = False          # Use PK sampler instead of random shuffle
    pk_p: int = 16                     # Number of identities per batch
    pk_k: int = 4                      # Number of images per identity per batch
    pk_min_samples: int = 2            # Minimum images a class needs for PK sampling

    # Retrieval evaluation
    retrieval_k: int = 5               # k for k-NN retrieval evaluation

    # Pretrained checkpoint to initialize from (e.g. best baseline before metric learning)
    init_from_checkpoint: Optional[str] = None  # Path to a .pth file

    # Fields that should NOT affect the experiment identity —
    # they're workflow/convenience settings, not training hyperparameters.
    _hash_exclude = {
        "auto_resume",
        "experiments_root",
        "save_every_n_epochs",
        "keep_top_k_checkpoints",
        "early_stopping_patience",
        "early_stopping_metric",
        "num_workers",
        "description",
        "experiment_name",
    }

    def config_hash(self) -> str:
        """Stable short hash for training-relevant config only."""
        d = {k: v for k, v in asdict(self).items() if k not in self._hash_exclude}
        config_str = json.dumps(d, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def generate_id(self) -> str:
        """Generate a unique experiment ID from config hash + timestamp."""
        config_hash = self.config_hash()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name_part = self.experiment_name.replace(" ", "_")[:30] if self.experiment_name else "run"
        return f"{timestamp}_{name_part}_{config_hash}"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter out any keys that don't exist in the dataclass
        # (forward compat) and let missing keys use defaults (backward compat)
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def summary(self) -> str:
        """One-line summary for the experiment index."""
        base = (
            f"{self.backbone} | {self.image_size[0]}px | "
            f"{self.loss_type}(γ={self.focal_gamma}) | "
            f"lr={self.lr_backbone}/{self.lr_head} | "
            f"freeze={self.freeze_backbone_epochs}ep | "
            f"bs={self.batch_size}×{self.accumulation_steps}"
        )
        if self.head_type != "linear":
            base += f" | head={self.head_type}"
        if self.head_type == "arcface":
            base += f"(s={self.arcface_scale},m={self.arcface_margin})"
        if self.loss_type == "triplet":
            base += f" | triplet(m={self.triplet_margin},{self.triplet_mining})"
        if self.pk_sampling:
            base += f" | PK({self.pk_p}×{self.pk_k})"
        return base
