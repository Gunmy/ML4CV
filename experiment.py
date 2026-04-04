"""
Experiment manager: handles directory creation, checkpointing, metric logging,
experiment indexing, and training resumption.
"""

import os
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import torch


class ExperimentManager:
    """
    Manages the lifecycle of a single experiment:
      - Creates directory structure
      - Saves/loads checkpoints (model + optimizer + scheduler + epoch)
      - Logs metrics after every epoch (crash-safe)
      - Maintains a global index of all experiments
      - Supports resumption from latest checkpoint
    """

    def __init__(self, config, experiment_id: Optional[str] = None):
        self.config = config
        self.experiment_id = experiment_id or config.generate_id()

        # Directory structure
        self.root = Path(config.experiments_root) / self.experiment_id
        self.checkpoint_dir = self.root / "checkpoints"
        self.log_path = self.root / "log.txt"
        self.metrics_path = self.root / "metrics.json"
        self.config_path = self.root / "config.json"
        self.index_path = Path(config.experiments_root) / "index.json"

        # State
        self.metrics_history: Dict[str, List[float]] = {}
        self.best_metric_value = -float("inf")
        self.best_epoch = -1

    def setup(self, save_config: bool = True):
        """Create directories and optionally save config. Call once at the start."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        if save_config:
            self.config.save(str(self.config_path))

        # Initialize or load existing metrics
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics_history = json.load(f)
            self.log(f"Resumed experiment: {self.experiment_id}")
        else:
            self.metrics_history = {}
            self.log(f"New experiment: {self.experiment_id}")

        self.log(f"Config: {self.config.summary()}")
        self._update_index()

        return self

    def log(self, message: str):
        """Print and append to log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Metrics ──────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Record metrics for one epoch. Saved to disk immediately (crash-safe).

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict of metric_name -> value, e.g.
                     {"train_loss": 2.3, "val_known_acc": 0.35, ...}
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Persist immediately
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Check if this is the best epoch
        monitor = self.config.early_stopping_metric
        if monitor in metrics:
            if metrics[monitor] > self.best_metric_value:
                self.best_metric_value = metrics[monitor]
                self.best_epoch = epoch
                return True  # Signal: new best
        return False

    def should_stop_early(self, current_epoch: int) -> bool:
        """Check if early stopping criterion is met."""
        if self.best_epoch < 0:
            return False
        patience = self.config.early_stopping_patience
        return (current_epoch - self.best_epoch) >= patience

    # ── Checkpoints ──────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, model, optimizer, scheduler,
                        scaler=None, is_best: bool = False,
                        extra: Optional[Dict] = None):
        """
        Save a training checkpoint. Keeps top-K best + latest.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "config": self.config.to_dict(),
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        if extra is not None:
            state["extra"] = extra

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(state, latest_path)

        # Save epoch checkpoint
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
            shutil.copy2(latest_path, epoch_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            shutil.copy2(latest_path, best_path)
            self.log(f"  ✓ New best model saved (epoch {epoch}, "
                     f"{self.config.early_stopping_metric}={self.best_metric_value:.4f})")

        # Prune old checkpoints
        self._prune_checkpoints()

    def load_checkpoint(self, model, optimizer=None, scheduler=None,
                        scaler=None, checkpoint: str = "latest") -> int:
        """
        Load a checkpoint. Returns the epoch to resume from.
        """
        if checkpoint in ("latest", "best"):
            path = self.checkpoint_dir / f"{checkpoint}.pth"
        else:
            path = Path(checkpoint)

        if not path.exists():
            self.log(f"No checkpoint found at {path}. Starting from scratch.")
            return 0

        self.log(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])

        self.best_metric_value = state.get("best_metric_value", -float("inf"))
        self.best_epoch = state.get("best_epoch", -1)

        resume_epoch = state["epoch"] + 1
        self.log(f"Resuming from epoch {resume_epoch}")
        return resume_epoch

    def get_extra(self, checkpoint: str = "best") -> Optional[Dict]:
        """Load the 'extra' dict from a checkpoint (e.g. class mappings)."""
        if checkpoint in ("latest", "best"):
            path = self.checkpoint_dir / f"{checkpoint}.pth"
        else:
            path = Path(checkpoint)
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        return state.get("extra", None)

    def _prune_checkpoints(self):
        """Remove old epoch checkpoints, keeping top-K and special files."""
        keep = {"latest.pth", "best.pth"}
        epoch_files = sorted([
            f for f in self.checkpoint_dir.iterdir()
            if f.name.startswith("epoch_") and f.suffix == ".pth"
        ])
        k = self.config.keep_top_k_checkpoints
        to_remove = epoch_files[:-k] if len(epoch_files) > k else []
        for f in to_remove:
            if f.name not in keep:
                f.unlink()

    # ── Experiment Index ─────────────────────────────────────────────────

    def _update_index(self):
        """Update the global experiment index file."""
        index = {}
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

        index[self.experiment_id] = {
            "summary": self.config.summary(),
            "name": self.config.experiment_name,
            "description": self.config.description,
            "created": datetime.now().isoformat(),
            "path": str(self.root),
        }

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def finalize(self, final_metrics: Optional[Dict[str, float]] = None):
        """Update index with final results after training completes."""
        index = {}
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

        if self.experiment_id in index:
            index[self.experiment_id]["completed"] = datetime.now().isoformat()
            index[self.experiment_id]["best_epoch"] = self.best_epoch
            index[self.experiment_id]["best_metric"] = self.best_metric_value
            if final_metrics:
                index[self.experiment_id]["final_metrics"] = final_metrics

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    # ── Comparison Utility ───────────────────────────────────────────────

    @staticmethod
    def compare_experiments(experiments_root: str,
                            experiment_ids: Optional[List[str]] = None):
        """
        Load metrics from multiple experiments for comparison.
        Returns a dict of {experiment_id: metrics_history}.
        """
        root = Path(experiments_root)
        index_path = root / "index.json"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            return {}

        if experiment_ids is None:
            experiment_ids = list(index.keys())

        results = {}
        for eid in experiment_ids:
            metrics_path = root / eid / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                summary = index.get(eid, {}).get("summary", "unknown")
                results[eid] = {
                    "metrics": metrics,
                    "summary": summary,
                }

        return results

    # ── Auto-resume ──────────────────────────────────────────────────────

    @staticmethod
    def find_latest_matching_experiment(config) -> Optional[str]:
        """
        Find the most recent experiment ID with the same config hash.

        Matching is based on the trailing config hash in experiment IDs:
        YYYYmmdd_HHMMSS_name_<hash>
        """
        root = Path(config.experiments_root)
        if not root.exists():
            return None

        suffix = f"_{config.config_hash()}"
        candidates = []

        for child in root.iterdir():
            if not child.is_dir() or not child.name.endswith(suffix):
                continue

            latest_ckpt = child / "checkpoints" / "latest.pth"
            if not latest_ckpt.exists():
                continue

            # IDs begin with timestamp "YYYYmmdd_HHMMSS"; lexical order matches time order.
            candidates.append(child.name)

        if not candidates:
            return None

        return sorted(candidates)[-1]

    @staticmethod
    def is_experiment_completed(experiments_root: str, experiment_id: str) -> bool:
        """
        Check whether an experiment has been finalized.

        Completed experiments are marked by `finalize()` in index.json.
        """
        index_path = Path(experiments_root) / "index.json"
        if not index_path.exists():
            return False

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            return False

        entry = index.get(experiment_id, {})
        return bool(entry.get("completed"))
