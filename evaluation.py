"""
Evaluation for whale identification.

Computes separate metrics for:
  1. Known whale accuracy  — can the model tell whale A from whale B?
  2. New whale detection   — can it correctly flag unknowns?
  3. Overall accuracy      — combined, where correct = right ID or correct "unknown" flag.
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
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    config,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation/test set.

    Returns a dict with:
      - val_loss:                 Loss on known whales (ignoring new_whale)
      - val_known_acc:            Accuracy on samples whose true class is known
      - val_new_whale_detection:  Fraction of new_whale samples correctly flagged
                                  (i.e. model's max confidence < threshold)
      - val_overall_acc:          Combined accuracy
      - val_mean_confidence:      Average softmax confidence on known whale predictions
    """
    model.eval()

    running_loss = 0.0
    loss_total = 0

    # Known whale metrics
    known_correct = 0
    known_total = 0

    # New whale detection metrics
    new_whale_detected = 0  # Correctly flagged as unknown
    new_whale_total = 0

    # Overall
    overall_correct = 0
    overall_total = 0

    # Confidence tracking
    confidence_sum = 0.0
    confidence_count = 0

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

        # Probabilities and predictions
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

        # ── Known whale samples ──────────────────────────────────────────
        if known_mask.sum() > 0:
            known_preds = preds[known_mask]
            known_labels = labels[known_mask]
            known_conf = max_probs[known_mask]

            known_correct += (known_preds == known_labels).sum().item()
            known_total += known_mask.sum().item()

            confidence_sum += known_conf.sum().item()
            confidence_count += known_mask.sum().item()

            # For overall accuracy: known whale is correct if prediction matches
            overall_correct += (known_preds == known_labels).sum().item()
            overall_total += known_mask.sum().item()

        # ── New whale samples (label == -1) ──────────────────────────────
        unknown_mask = labels == -1
        if unknown_mask.sum() > 0:
            unknown_conf = max_probs[unknown_mask]
            # A new_whale is "detected" if confidence is below threshold
            detected = (unknown_conf < config.confidence_threshold).sum().item()
            new_whale_detected += detected
            new_whale_total += unknown_mask.sum().item()

            # For overall accuracy: new_whale is correct if flagged as unknown
            overall_correct += detected
            overall_total += unknown_mask.sum().item()

        # Progress
        if known_total > 0:
            pbar.set_postfix({
                "known_acc": f"{known_correct / known_total * 100:.1f}%",
                "nw_det": f"{new_whale_detected / max(new_whale_total, 1) * 100:.1f}%",
            })

    # ── Compute final metrics ────────────────────────────────────────────
    val_loss = running_loss / max(loss_total, 1)
    val_known_acc = known_correct / max(known_total, 1)
    val_new_whale_det = new_whale_detected / max(new_whale_total, 1)
    val_overall_acc = overall_correct / max(overall_total, 1)
    val_mean_conf = confidence_sum / max(confidence_count, 1)

    return {
        "val_loss": val_loss,
        "val_known_acc": val_known_acc,
        "val_new_whale_detection": val_new_whale_det,
        "val_overall_acc": val_overall_acc,
        "val_mean_confidence": val_mean_conf,
        "val_known_total": known_total,
        "val_new_whale_total": new_whale_total,
    }


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


# ── Retrieval Evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_retrieval(
    model,
    gallery_loader,
    query_loader,
    config,
    device,
) -> Dict[str, float]:
    """
    Evaluate embedding quality via k-NN retrieval.
    (Course slides 8–9: face recognition as k-NN in embedding space)

    Builds a gallery from gallery_loader, then for each query:
      1. Compute query embedding
      2. Find k nearest neighbors in the gallery by cosine similarity
      3. Check if the correct identity is among the top-k

    For new_whale queries: correct if the nearest neighbor distance exceeds
    a threshold (i.e. nothing in the gallery is close enough).

    Args:
        model:          Trained model.
        gallery_loader: DataLoader for gallery images (typically training set).
        query_loader:   DataLoader for query images (typically validation set).
        config:         ExperimentConfig (for retrieval_k, confidence_threshold).
        device:         Target device.

    Returns:
        Dict with retrieval metrics: recall@1, recall@5, new_whale detection, etc.
    """
    model.eval()
    k = config.retrieval_k

    # ── Build gallery ────────────────────────────────────────────────────
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device)

    # Remove any new_whale from gallery (they're not identifiable)
    known_mask = gallery_labels >= 0
    gallery_emb = gallery_emb[known_mask]
    gallery_labels = gallery_labels[known_mask]

    print(f"  Gallery: {gallery_emb.size(0)} embeddings, "
          f"{gallery_labels.unique().size(0)} identities")

    # ── Query and evaluate ───────────────────────────────────────────────
    query_emb, query_labels = extract_embeddings(model, query_loader, device)

    # Cosine similarity (embeddings are already L2-normalized)
    # Process in chunks to avoid OOM on large galleries
    chunk_size = 256
    recall_at_1 = 0
    recall_at_k = 0
    known_total = 0
    new_whale_detected = 0
    new_whale_total = 0
    mean_pos_sim = 0.0
    mean_neg_sim = 0.0
    pos_count = 0
    neg_count = 0

    for start in range(0, query_emb.size(0), chunk_size):
        end = min(start + chunk_size, query_emb.size(0))
        q_batch = query_emb[start:end]        # (chunk, D)
        q_labels = query_labels[start:end]     # (chunk,)

        # Cosine similarities: (chunk, gallery_size)
        sims = q_batch @ gallery_emb.t()

        # Top-k most similar gallery items
        topk_sims, topk_indices = sims.topk(k, dim=1)  # (chunk, k)
        topk_labels = gallery_labels[topk_indices]       # (chunk, k)

        for i in range(q_batch.size(0)):
            # q_labels is already the current chunk slice, so index locally.
            true_label = q_labels[i].item()

            if true_label == -1:
                # new_whale: should NOT match anything closely
                new_whale_total += 1
                max_sim = topk_sims[i, 0].item()
                # Use a similarity threshold (higher threshold = stricter matching)
                # cosine similarity of 0.5 ≈ roughly the same as confidence threshold
                if max_sim < config.confidence_threshold:
                    new_whale_detected += 1
            else:
                # Known whale: correct if true label appears in top-k
                known_total += 1
                retrieved_labels = topk_labels[i].tolist()

                if true_label == retrieved_labels[0]:
                    recall_at_1 += 1
                if true_label in retrieved_labels:
                    recall_at_k += 1

                # Track positive/negative similarity distributions
                for j in range(k):
                    sim_val = topk_sims[i, j].item()
                    if retrieved_labels[j] == true_label:
                        mean_pos_sim += sim_val
                        pos_count += 1
                    else:
                        mean_neg_sim += sim_val
                        neg_count += 1

    metrics = {
        "retrieval_recall@1": recall_at_1 / max(known_total, 1),
        "retrieval_recall@k": recall_at_k / max(known_total, 1),
        "retrieval_k": k,
        "retrieval_new_whale_detection": new_whale_detected / max(new_whale_total, 1),
        "retrieval_known_total": known_total,
        "retrieval_new_whale_total": new_whale_total,
        "retrieval_mean_pos_sim": mean_pos_sim / max(pos_count, 1),
        "retrieval_mean_neg_sim": mean_neg_sim / max(neg_count, 1),
    }

    print(f"  Recall@1: {metrics['retrieval_recall@1']*100:.2f}%  "
          f"Recall@{k}: {metrics['retrieval_recall@k']*100:.2f}%  "
          f"NewWhale Det: {metrics['retrieval_new_whale_detection']*100:.2f}%")

    return metrics
