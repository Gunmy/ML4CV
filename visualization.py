"""
Visualization utilities for training analysis and experiment comparison.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.manifold import TSNE
import torch

from evaluation import extract_embeddings

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["figure.figsize"] = (14, 5)


def plot_training_history(metrics: Dict[str, list], title: str = ""):
    """
    Plot loss and accuracy curves from a single experiment's metrics dict.
    Shows train vs. val for both loss and known-whale accuracy.
    """
    epochs = range(1, len(metrics.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        print("No training data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Loss ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, metrics["train_loss"], "b-o", markersize=4, label="Train")
    if "val_loss" in metrics:
        ax.plot(epochs, metrics["val_loss"], "r-o", markersize=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, [x * 100 for x in metrics["train_acc"]],
            "b-o", markersize=4, label="Train")
    if "val_known_acc" in metrics:
        ax.plot(epochs, [x * 100 for x in metrics["val_known_acc"]],
                "r-o", markersize=4, label="Val (known)")
    elif "retrieval_recall@1" in metrics:
        ax.plot(epochs, [x * 100 for x in metrics["retrieval_recall@1"]],
                "r-o", markersize=4, label="Val (Recall@1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learning rate ────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # LR on secondary axis
    if "lr" in metrics:
        ax2 = ax.twinx()
        ax2.plot(epochs, metrics["lr"], "k--", alpha=0.5, label="LR")
        ax2.set_ylabel("Learning Rate", color="k")
        ax2.tick_params(axis="y", labelcolor="k")
        ax2.legend(loc="lower right")

    ax.set_title("Learning Rate")
    ax.legend(loc="upper left")

    # ── Freeze boundary ──────────────────────────────────────────────────
    # If there's a sharp change in grad_norm, mark it
    if "grad_norm" in metrics:
        norms = metrics["grad_norm"]
        for i in range(1, len(norms)):
            if norms[i] > 3 * norms[i - 1] and norms[i - 1] > 0:
                for a in axes:
                    a.axvline(x=i + 1, color="orange", linestyle="--",
                              alpha=0.7, label="Unfreeze" if a == axes[0] else "")
                break

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def compare_experiments(experiments_root: str,
                        experiment_ids: Optional[List[str]] = None,
                        metric: str = "val_known_acc"):
    """
    Plot a single metric across multiple experiments for comparison.
    """
    root = Path(experiments_root)
    index_path = root / "index.json"

    if not index_path.exists():
        print("No experiment index found.")
        return

    with open(index_path) as f:
        index = json.load(f)

    if experiment_ids is None:
        experiment_ids = list(index.keys())

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for eid in experiment_ids:
        metrics_path = root / eid / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        if metric not in metrics:
            continue

        values = metrics[metric]
        label = index.get(eid, {}).get("summary", eid)
        epochs = range(1, len(values) + 1)

        # For accuracy metrics, display as percentage
        if "acc" in metric or "recall" in metric:
            values = [v * 100 for v in values]
            ax.set_ylabel(f"{metric} (%)")
        else:
            ax.set_ylabel(metric)

        ax.plot(epochs, values, "-o", markersize=3, label=label)

    ax.set_xlabel("Epoch")
    ax.set_title(f"Experiment Comparison: {metric}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def list_experiments(experiments_root: str):
    """Print a table of all experiments and their results."""
    root = Path(experiments_root)
    index_path = root / "index.json"

    if not index_path.exists():
        print("No experiments found.")
        return

    with open(index_path) as f:
        index = json.load(f)

    print(f"{'ID':<45} {'Best Metric':>12} {'Best Epoch':>10}  Summary")
    print("-" * 120)

    for eid, info in sorted(index.items()):
        best = info.get("best_metric", "N/A")
        best_ep = info.get("best_epoch", "N/A")
        summary = info.get("summary", "")

        if isinstance(best, float):
            best = f"{best:.4f}"

        print(f"{eid:<45} {str(best):>12} {str(best_ep):>10}  {summary}")


def plot_embedding_tsne(model, loader, data, device, title="",
                        max_samples=500, top_k_classes=10,
                        min_samples_per_class=5, combine_loaders=None):
    """
    Extract embeddings and plot t-SNE colored by class.
    
    Args:
        combine_loaders: Optional list of extra DataLoaders to pool with `loader`
                         for more points (e.g. [train_loader]) — only affects
                         the plot, not evaluation metrics.
    """
    # Gather embeddings from primary + optional extra loaders
    all_emb, all_lab = extract_embeddings(model, loader, device)
    if combine_loaders:
        for extra_loader in combine_loaders:
            e, l = extract_embeddings(model, extra_loader, device)
            all_emb = torch.cat([all_emb, e], dim=0)
            all_lab = torch.cat([all_lab, l], dim=0)

    # Filter to known whales only
    known_mask = all_lab >= 0
    emb = all_emb[known_mask].numpy()
    lab = all_lab[known_mask].numpy()

    # Keep only classes with enough samples for visible clusters
    unique, counts = np.unique(lab, return_counts=True)
    frequent = unique[counts >= min_samples_per_class]
    if len(frequent) == 0:
        print(f"No classes have ≥{min_samples_per_class} samples. "
              f"Lowering threshold to 2.")
        frequent = unique[counts >= 2]
    if len(frequent) == 0:
        print("Not enough data to plot meaningful t-SNE.")
        return

    # Pick top-K most frequent among those
    freq_counts = {c: counts[unique == c][0] for c in frequent}
    top_classes = sorted(freq_counts, key=freq_counts.get, reverse=True)[:top_k_classes]
    class_mask = np.isin(lab, top_classes)
    emb = emb[class_mask]
    lab = lab[class_mask]

    # Subsample if needed
    if len(emb) > max_samples:
        idx = np.random.choice(len(emb), max_samples, replace=False)
        emb, lab = emb[idx], lab[idx]

    n_points = len(emb)
    n_classes = len(np.unique(lab))
    print(f"t-SNE: {n_points} points, {n_classes} classes "
          f"(min {min_samples_per_class}/class)")

    # Auto-tune perplexity: should be much less than n_points,
    # but high enough to capture local structure
    perplexity = min(max(n_points // 8, 5), 50)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1500,
        init="pca",
        learning_rate="auto",
        n_jobs=-1,
    )
    emb_2d = tsne.fit_transform(emb)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Use a colormap with good perceptual separation
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20")
    classes_sorted = sorted(np.unique(lab))

    for i, c in enumerate(classes_sorted):
        mask = lab == c
        color = cmap(i % cmap.N)
        whale_name = data["idx_to_id"].get(int(c), str(c))[:12]
        count = mask.sum()

        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=50, alpha=0.8, color=color, edgecolors="white",
            linewidths=0.3, label=f"{whale_name} ({count})", zorder=2,
        )

        # Centroid label for each cluster
        cx, cy = emb_2d[mask, 0].mean(), emb_2d[mask, 1].mean()
        ax.annotate(
            whale_name, (cx, cy), fontsize=7, fontweight="bold",
            color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                      alpha=0.7, lw=0.5),
            zorder=3,
        )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        fontsize=7, markerscale=1.5, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
        frameon=True, framealpha=0.9,
    )

    plt.tight_layout()
    plt.show()

    print(f"  Perplexity used: {perplexity} | KL divergence: {tsne.kl_divergence_:.2f}")