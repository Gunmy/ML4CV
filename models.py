"""
Model factory for whale identification.

Creates a backbone (from timm) with a configurable head:
  - "linear":  standard classification (baseline)
  - "arcface": ArcFace angular margin classification (metric learning)
  - "none":    embedding only, no classifier (for triplet loss)

Provides utilities for freeze/unfreeze, BN management, and checkpoint loading.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List, Optional


# ── ArcFace Head ─────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    (Deng et al., CVPR 2019 — course slides 31–33)

    Instead of computing logits as W·f(x), ArcFace:
      1. L2-normalizes both the weight vectors and the embedding
      2. Computes cosine similarity (= dot product of unit vectors)
      3. Adds an angular margin penalty m to the angle of the correct class
      4. Scales by factor s before softmax

    This forces the model to learn embeddings that cluster tightly by class
    on a hypersphere, which is exactly what we want for retrieval/k-NN.

    The angular margin means the model must push same-class embeddings
    closer together than standard softmax requires — it's not enough for
    the correct class to have the highest logit; it must have the highest
    logit by a margin of m radians.

    Args:
        embedding_dim: Dimension of the input embedding.
        num_classes:   Number of output classes.
        s:             Scale factor (controls logit magnitude → softmax sharpness).
        m:             Angular margin in radians (typically 0.5).
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes

        # Weight matrix — each column is a "class template" on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold to avoid numerical issues with acos
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embedding: (B, embedding_dim) — will be L2-normalized internally.
            labels:    (B,) integer class labels. Required during training
                       to know which class gets the margin penalty.
                       During inference (labels=None), returns plain cosine logits.
        Returns:
            Scaled logits of shape (B, num_classes).
        """
        # L2-normalize both embedding and weights → cosine similarity
        emb_norm = F.normalize(embedding, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb_norm, w_norm)  # (B, num_classes)

        if labels is None:
            # Inference mode: no margin, just scaled cosine
            return self.s * cosine

        # ── Training: apply angular margin to the correct class ──────────
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        cos_theta_plus_m = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: if θ + m > π, use a linear fallback
        cos_theta_plus_m = torch.where(
            cosine > self.th, cos_theta_plus_m, cosine - self.mm
        )

        # One-hot mask for the correct class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Replace the correct class's cosine with the margin-penalized version
        logits = torch.where(one_hot.bool(), cos_theta_plus_m, cosine)

        return self.s * logits


# ── Model ────────────────────────────────────────────────────────────────

class WhaleClassifier(nn.Module):
    """
    Wrapper around a timm backbone with a configurable head.

    Architecture:
      backbone → global pool → embedding → head

    Head types:
      - "linear":  nn.Linear for standard classification (default, backward compat)
      - "arcface": ArcFaceHead for angular margin metric learning
      - "none":    no classifier — returns embedding only (for triplet loss)
    """

    def __init__(self, backbone_name: str, num_classes: int,
                 embedding_dim: int = 512, pretrained: bool = True,
                 head_type: str = "linear",
                 arcface_scale: float = 30.0, arcface_margin: float = 0.5):
        super().__init__()
        self.head_type = head_type

        # Create backbone without its default classifier head
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
        )
        backbone_out_dim = self.backbone.num_features

        # Embedding projection: backbone features → compact embedding
        self.embedding = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Initialize embedding weights
        nn.init.kaiming_normal_(self.embedding[0].weight)
        nn.init.zeros_(self.embedding[0].bias)

        # Classification head (depends on head_type)
        if head_type == "linear":
            self.classifier = nn.Linear(embedding_dim, num_classes)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        elif head_type == "arcface":
            self.classifier = ArcFaceHead(
                embedding_dim, num_classes,
                s=arcface_scale, m=arcface_margin,
            )
        elif head_type == "none":
            self.classifier = None  # No classifier — embedding-only model
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                return_embedding: bool = False):
        """
        Args:
            x:       (B, 3, H, W) input images.
            labels:  (B,) class labels — required for ArcFace during training.
            return_embedding: If True, return (logits, embedding) tuple.
        """
        features = self.backbone(x)          # (B, backbone_out_dim)
        emb = self.embedding(features)       # (B, embedding_dim)

        if self.classifier is None:
            # Embedding-only mode (triplet loss)
            return F.normalize(emb, p=2, dim=1) if not return_embedding else (None, emb)

        if isinstance(self.classifier, ArcFaceHead):
            logits = self.classifier(emb, labels=labels)
        else:
            logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits

    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract L2-normalized embedding for retrieval / metric learning."""
        features = self.backbone(x)
        emb = self.embedding(features)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


# ── Freeze / Unfreeze Utilities ──────────────────────────────────────────

def freeze_backbone(model: WhaleClassifier):
    """Freeze all backbone parameters. Only the head (embedding + classifier) trains."""
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model: WhaleClassifier):
    """Unfreeze all backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = True

def set_bn_eval(model: nn.Module):
    """
    Set all BatchNorm layers to eval mode.
    
    This is CRITICAL when fine-tuning with small micro-batches:
    BN running statistics from ImageNet pretraining are far more reliable
    than statistics computed from 4-8 images.
    
    Call this AFTER model.train() in your training loop.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()


# ── Optimizer Construction ───────────────────────────────────────────────

def build_optimizer(config, model: WhaleClassifier, is_frozen: bool) -> torch.optim.Optimizer:
    """
    Build optimizer with differential learning rates.

    When backbone is frozen: only head params are optimized.
    When backbone is unfrozen: backbone gets lr_backbone, head gets lr_head.
    """
    # Collect all head parameters (embedding + classifier if it exists)
    head_params = list(model.embedding.parameters())
    if model.classifier is not None:
        head_params += list(model.classifier.parameters())

    if is_frozen:
        param_groups = [
            {"params": head_params, "lr": config.lr_head},
        ]
    else:
        backbone_params = list(model.backbone.parameters())
        param_groups = [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": head_params, "lr": config.lr_head},
        ]

    if config.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                               weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(config, optimizer, steps_per_epoch: int,
                    total_epochs: int):
    """Build LR scheduler. Returns (scheduler, is_batch_level)."""
    if config.scheduler == "cosine":
        # Total training steps for cosine annealing
        total_steps = steps_per_epoch * total_epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1),
            eta_min=config.cosine_eta_min,
        )

        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps],
            )

        return scheduler, True  # Step per batch

    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.plateau_factor,
            patience=config.plateau_patience, verbose=True,
        )
        return scheduler, False  # Step per epoch

    elif config.scheduler == "none":
        return None, False

    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


# ── Model Factory ────────────────────────────────────────────────────────

def build_model(config, num_classes: int, device: torch.device) -> WhaleClassifier:
    """
    Build a WhaleClassifier from config, optionally loading from a checkpoint.

    If config.init_from_checkpoint is set, loads backbone + embedding weights
    from that checkpoint (ignoring the classifier head, since it may have a
    different architecture — e.g. linear baseline → arcface).
    """
    model = WhaleClassifier(
        backbone_name=config.backbone,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained,
        head_type=config.head_type,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
    ).to(device)

    if config.init_from_checkpoint:
        print(f"Loading backbone + embedding from: {config.init_from_checkpoint}")
        state = torch.load(config.init_from_checkpoint, map_location=device,
                           weights_only=False)
        source_state = state.get("model_state_dict", state)

        # Load backbone and embedding weights (skip classifier — may differ)
        model_state = model.state_dict()
        loaded = 0
        for key in source_state:
            if key.startswith("backbone.") or key.startswith("embedding."):
                if key in model_state and source_state[key].shape == model_state[key].shape:
                    model_state[key] = source_state[key]
                    loaded += 1

        model.load_state_dict(model_state)
        print(f"  Loaded {loaded} parameter tensors (backbone + embedding)")

    return model


# ── Model Summary ────────────────────────────────────────────────────────

def print_model_summary(model: WhaleClassifier, config):
    """Print parameter counts: total, trainable, frozen."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    head_total = sum(p.numel() for p in model.embedding.parameters())
    if model.classifier is not None:
        head_total += sum(p.numel() for p in model.classifier.parameters())

    print(f"\nModel: {config.backbone} (head={config.head_type})")
    print(f"  Input size:     {config.image_size}")
    print(f"  Embedding dim:  {config.embedding_dim}")
    print(f"  Total params:   {total:>12,}")
    print(f"  Backbone:       {backbone_total:>12,}")
    print(f"  Head:           {head_total:>12,}")
    print(f"  Trainable:      {trainable:>12,} ({trainable / total * 100:.1f}%)")
    print(f"  Frozen:         {frozen:>12,} ({frozen / total * 100:.1f}%)")
    print()
