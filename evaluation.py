"""
Evaluation for whale identification during training.

We mostly just evaluate for known whales as this is cheaper than calculating
AUC for different confidence thresholds

"""

import torch
import torch.nn as nn
import numpy as np
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from typing import Dict

@torch.no_grad()
def val_loss(
    model: nn.Module,
    loader,
    criterion,
    config,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Evaluate the model on a validation/test set.

    Most metrics are just evaluated for known whales to avoid having to calculate
    whether something would be classified as a new whale or not under different
    treshold values

    The exception is val_mean_known_conf

    Returns a dict with:
      - val_loss:              Loss on known whales (ignoring new_whale)
      - val_known_acc:         Top-1 accuracy on known whale samples (in top 1)
      - val_known_top5_acc:    Top-5 accuracy on known whale samples (in top 5)
      - val_mean_known_conf:   Average softmax confidence on known whale samples
      - val_mean_new_whale_conf: Average softmax confidence on new whale samples
      - val_known_total:       Number of known whale samples
      - val_new_whale_total:   Number of new whale samples
    """
    model.eval()
    running_loss = 0.0
    loss_total = 0

    pbar = tqdm(loader, desc=f"  Val   Ep {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward
        if config.use_amp and device.type == "cuda":
            with autocast(device_type='cuda'):
                logits = model(images)
        else:
            logits = model(images)

        # Loss (only on known whales, loss handles ignore_index=-1)
        loss = criterion(logits, labels)
        known_mask = labels != -1
        if known_mask.sum() > 0 and loss.item() != 0.0:
            running_loss += loss.item() * known_mask.sum().item()
            loss_total += known_mask.sum().item()

    val_loss = running_loss / max(loss_total, 1)

    return val_loss


# ── Embedding Extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, normalize: bool = True):
    """
    Extract embeddings and labels from an entire dataset.

    Returns:
        embeddings: (N, D) tensor of embeddings.
        labels:     (N,) tensor of integer labels (-1 for new_whale).
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Extracting embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        emb = model.get_embedding(images, normalize=normalize)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)
