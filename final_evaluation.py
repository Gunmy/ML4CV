"""
Final comprehensive evaluation for whale identification models.

Extracts raw predictions, computes base metrics, and sweeps thresholds 
to find the optimal Kaggle MAP@5 score.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Re-use the existing extraction utility for the k-NN gallery building
from evaluation import extract_embeddings


# ── 1. Extraction Functions ────────────────────────────────────────────────

@torch.no_grad()
def get_classifier_predictions(
    model: nn.Module, 
    query_loader: DataLoader, 
    device: torch.device, 
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Top-K probabilities using standard classification (Baseline/Focal).
    Returns: (scores, predictions, labels)
    """
    model.eval()
    all_scores, all_preds, all_labels = [], [], []

    for images, labels in tqdm(query_loader, desc="Extracting Classifier Preds"):
        images = images.to(device, non_blocking=True)
        
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        
        scores, preds = probs.topk(k, dim=1)
        
        all_scores.append(scores.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores, dim=0), torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def get_retrieval_predictions(
    model: nn.Module, 
    gallery_loader: DataLoader, 
    query_loader: DataLoader, 
    device: torch.device, 
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Top-K similarities using k-NN against a gallery (Triplet/ArcFace).
    Returns: (scores, predictions, labels)
    """
    model.eval()
    
    # 1. Build Gallery (excluding new_whale)
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device)
    known_mask = gallery_labels != -1
    gallery_emb = gallery_emb[known_mask]
    gallery_labels = gallery_labels[known_mask]

    # 2. Extract Query features
    query_emb, query_labels = extract_embeddings(model, query_loader, device)

    # 3. Compute k-NN similarities in chunks
    chunk_size = 512
    all_scores, all_preds = [], []

    for start in tqdm(range(0, query_emb.size(0), chunk_size), desc="Computing k-NN"):
        end = min(start + chunk_size, query_emb.size(0))
        q_chunk = query_emb[start:end]
        
        sims = q_chunk @ gallery_emb.t()  # Cosine similarity
        scores, topk_indices = sims.topk(k, dim=1)
        preds = gallery_labels[topk_indices]
        
        all_scores.append(scores)
        all_preds.append(preds)

    return torch.cat(all_scores, dim=0), torch.cat(all_preds, dim=0), query_labels


# ── 2. Fast Metric Calculations (CPU Math) ─────────────────────────────────

def evaluate_base_metrics(
    scores: torch.Tensor, 
    predictions: torch.Tensor, 
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Calculates fundamental, unthresholded metrics.
    """
    known_mask = labels != -1
    new_mask = labels == -1
    
    # Known metrics
    known_preds = predictions[known_mask]
    known_labels = labels[known_mask]
    
    # Recall
    recall_at_1 = (known_preds[:, 0] == known_labels).float().mean().item()
    recall_at_5 = (known_preds[:, :5] == known_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Confidences
    mean_known_conf = scores[known_mask, 0].mean().item() if known_mask.any() else 0.0
    mean_new_whale_conf = scores[new_mask, 0].mean().item() if new_mask.any() else 0.0

    return {
        "known_recall@1": recall_at_1,
        "known_recall@5": recall_at_5,
        "mean_known_conf": mean_known_conf,
        "mean_new_whale_conf": mean_new_whale_conf
    }

def insert_new_whale(preds: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Inserts -1 (new_whale) into the top-k predictions based on the threshold.
    """
    result = []
    new_inserted = False
    
    for p, s in zip(preds, scores):
        if s < threshold and not new_inserted:
            result.append(-1)
            new_inserted = True
            if len(result) == 5: break
            
        result.append(p)
        if len(result) == 5: break
        
    if not new_inserted and len(result) < 5:
        result.append(-1)
        
    return np.array(result)

def calculate_map5(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculates Kaggle's Mean Average Precision @ 5."""
    map_score = 0.0
    for preds, true_label in zip(predictions, labels):
        matches = np.where(preds == true_label)[0]
        if len(matches) > 0:
            rank = matches[0] + 1
            map_score += 1.0 / rank
            
    return map_score / len(labels)


# ── 3. Threshold Sweeping ──────────────────────────────────────────────────

def sweep_thresholds(
    scores: torch.Tensor, 
    predictions: torch.Tensor, 
    labels: torch.Tensor, 
    num_steps: int = 100
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Sweeps thresholds to find optimal MAP@5.
    Returns: (thresholds_array, map5_scores_array, best_threshold, best_map5)
    """
    preds_np = predictions.numpy()
    scores_np = scores.numpy()
    labels_np = labels.numpy()
    
    min_score = max(0.0, float(np.min(scores_np)))
    max_score = min(1.0, float(np.max(scores_np)))
    
    thresholds = np.linspace(min_score, max_score, num_steps)
    map5_scores = []
    
    for t in tqdm(thresholds, desc="Sweeping Thresholds", leave=False):
        final_preds = np.array([
            insert_new_whale(p, s, t) 
            for p, s in zip(preds_np, scores_np)
        ])
        
        map5 = calculate_map5(final_preds, labels_np)
        map5_scores.append(map5)
        
    map5_scores = np.array(map5_scores)
    best_idx = np.argmax(map5_scores)
    
    return thresholds, map5_scores, float(thresholds[best_idx]), float(map5_scores[best_idx])