"""
Visualization utilities for training analysis and experiment comparison.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    if "val_overall_acc" in metrics:
        ax.plot(epochs, [x * 100 for x in metrics["val_overall_acc"]],
                "g-o", markersize=4, label="Val (overall)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── New whale detection + Learning rate ──────────────────────────────
    ax = axes[2]
    if "val_new_whale_detection" in metrics:
        ax.plot(epochs, [x * 100 for x in metrics["val_new_whale_detection"]],
                "m-o", markersize=4, label="New whale detection %")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detection Rate (%)", color="m")
    ax.tick_params(axis="y", labelcolor="m")
    ax.grid(True, alpha=0.3)

    # LR on secondary axis
    if "lr" in metrics:
        ax2 = ax.twinx()
        ax2.plot(epochs, metrics["lr"], "k--", alpha=0.5, label="LR")
        ax2.set_ylabel("Learning Rate", color="k")
        ax2.tick_params(axis="y", labelcolor="k")
        ax2.legend(loc="lower right")

    ax.set_title("New Whale Detection & LR")
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
        if "acc" in metric or "detection" in metric:
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
