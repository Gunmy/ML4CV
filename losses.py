"""
Loss functions for whale identification.
Includes Focal Loss (a strict generalization of Cross-Entropy) and a factory
that builds the right loss from an ExperimentConfig.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so training focuses on hard ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0 this reduces exactly to (weighted) cross-entropy.
    When gamma>0, easy examples (high p_t) contribute less to the loss.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        gamma:           Focusing parameter. 0 = CE, 2.0 = standard focal.
        alpha:           Per-class weight tensor of shape (num_classes,), or None.
        label_smoothing: Smooth targets before computing loss.
        ignore_index:    Class index to ignore (e.g. -1 for unknown whales).
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Register alpha as a buffer so it moves with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw model outputs.
            targets: (B,) integer class labels. Values equal to ignore_index are skipped.
        """
        # Mask out ignored samples
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[mask]
        targets = targets[mask]

        num_classes = logits.size(1)

        # Apply label smoothing: convert hard targets to soft
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)     # (B, C)
        probs = log_probs.exp()                        # (B, C)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma     # (B, C)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha.unsqueeze(0)     # (1, C)
            focal_weight = focal_weight * alpha_weight

        # Compute focal loss: -alpha * (1-p)^gamma * log(p) * target
        loss = -focal_weight * log_probs * smooth_targets  # (B, C)
        loss = loss.sum(dim=1).mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with online semi-hard or hard negative mining.
    (Course slides 21–24, FaceNet / Weinberger et al.)

    Given a batch of L2-normalized embeddings and their labels, this loss:
      1. Computes all pairwise distances within the batch
      2. For each anchor, finds valid positives (same class) and negatives (different class)
      3. Mines informative triplets according to the chosen strategy
      4. Computes: L = max(0, d(a,p) - d(a,n) + margin)

    Semi-hard negatives (recommended): negatives that are farther than the positive
    but within the margin — these provide the most useful gradient signal.
    Hard negatives: the closest negative for each anchor-positive pair.

    Important: classes with only 1 image in the batch cannot form valid triplets
    and are effectively skipped.

    Args:
        margin:  Margin for the hinge loss. Typical values: 0.2–0.5.
        mining:  "semi_hard", "hard", or "all".
    """

    def __init__(self, margin: float = 0.3, mining: str = "semi_hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embedding vectors.
            labels:     (B,) integer class labels.

        Returns:
            Scalar loss. Also stores self.num_active_triplets and
            self.num_valid_triplets for monitoring.
        """
        # Pairwise squared Euclidean distances
        # For L2-normalized vectors: ||a - b||^2 = 2 - 2*a·b
        dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (B, B)

        B = embeddings.size(0)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) same-class mask
        labels_neq = ~labels_eq                                  # (B, B) diff-class mask

        # Remove self-pairs from positive mask
        eye_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_eq & ~eye_mask

        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        num_active = 0
        num_valid = 0

        for i in range(B):
            # Find positives and negatives for anchor i
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = labels_neq[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            for p_idx in pos_indices:
                d_ap = dist_mat[i, p_idx]

                if self.mining == "hard":
                    # Hardest negative: closest to anchor
                    d_an = dist_mat[i, neg_indices].min()
                    triplet_loss = F.relu(d_ap - d_an + self.margin)
                    num_valid += 1
                    if triplet_loss.item() > 0:
                        num_active += 1
                    loss = loss + triplet_loss

                elif self.mining == "semi_hard":
                    # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
                    d_an_all = dist_mat[i, neg_indices]
                    semi_hard_mask = (d_an_all > d_ap) & (d_an_all < d_ap + self.margin)

                    if semi_hard_mask.sum() > 0:
                        # Pick the hardest semi-hard negative
                        d_an = d_an_all[semi_hard_mask].min()
                    else:
                        # Fallback to hardest negative
                        d_an = d_an_all.min()

                    triplet_loss = F.relu(d_ap - d_an + self.margin)
                    num_valid += 1
                    if triplet_loss.item() > 0:
                        num_active += 1
                    loss = loss + triplet_loss

                elif self.mining == "all":
                    # All valid triplets
                    d_an_all = dist_mat[i, neg_indices]
                    triplet_losses = F.relu(d_ap - d_an_all + self.margin)
                    num_valid += len(neg_indices)
                    num_active += (triplet_losses > 0).sum().item()
                    loss = loss + triplet_losses.sum()

        # Store for monitoring
        self.num_active_triplets = num_active
        self.num_valid_triplets = max(num_valid, 1)

        return loss / max(num_valid, 1)


def build_loss(config, class_counts: np.ndarray, device: torch.device) -> nn.Module:
    """
    Build the appropriate loss function from config.

    Args:
        config:       ExperimentConfig.
        class_counts: Array of shape (num_classes,) with per-class sample counts.
        device:       Target device.
    """
    # Compute class weights (inverse frequency, normalized)
    alpha = None
    if config.use_class_weights:
        weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
        weights = weights / weights.sum() * len(class_counts)
        alpha = torch.tensor(weights, dtype=torch.float32)

    if config.loss_type == "focal":
        loss_fn = FocalLoss(
            gamma=config.focal_gamma,
            alpha=alpha,
            label_smoothing=config.label_smoothing,
            ignore_index=-1,
        )
    elif config.loss_type == "ce":
        # Use PyTorch's built-in CE (supports class weights natively)
        weight = alpha if alpha is not None else None
        loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=-1,
            label_smoothing=config.label_smoothing,
        )
    elif config.loss_type == "triplet":
        loss_fn = TripletLoss(
            margin=config.triplet_margin,
            mining=config.triplet_mining,
        )
    elif config.loss_type == "arcface_ce":
        # For ArcFace head: standard CE, no class weights needed
        # (ArcFace's angular margin handles class separation)
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=config.label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")

    return loss_fn.to(device)
