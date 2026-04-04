"""
Training loop for whale identification.

Features:
  - Automatic Mixed Precision (AMP) for memory efficiency
  - Gradient accumulation for effective large batch sizes
  - Two-phase training: frozen backbone → full fine-tuning
  - Supports classification (baseline), ArcFace, and triplet loss
  - Separate metrics for known whales vs. new_whale detection
  - Gradient norm monitoring
  - Auto-resume from latest matching experiment
"""

import time
import torch
import torch.nn as nn
import numpy as np

# AMP compatibility: torch.amp (2.4+) vs torch.cuda.amp (2.0–2.3)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from tqdm.auto import tqdm
from typing import Dict, Optional

from config import ExperimentConfig
from experiment import ExperimentManager
from models import (
    WhaleClassifier, build_model, freeze_backbone, unfreeze_backbone,
    set_bn_eval, build_optimizer, build_scheduler, print_model_summary,
)
from losses import build_loss
from evaluation import evaluate


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: WhaleClassifier,
    loader,
    criterion,
    optimizer,
    scheduler,
    scheduler_is_batch_level: bool,
    scaler: Optional[GradScaler],
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation and AMP.
    Handles classification, ArcFace, and triplet loss transparently.
    """
    model.train()
    if config.freeze_bn:
        set_bn_eval(model.backbone)

    running_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []
    accumulation_counter = 0

    is_triplet = config.loss_type == "triplet"
    is_arcface = config.head_type == "arcface"

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Train Ep {epoch}", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass (branched by loss type) ─────────────────────────
        if scaler is not None:
            with autocast(device_type='cuda'):
                if is_triplet:
                    embeddings = model(images)
                    loss = criterion(embeddings, labels) / config.accumulation_steps
                elif is_arcface:
                    logits = model(images, labels=labels)
                    loss = criterion(logits, labels) / config.accumulation_steps
                else:
                    logits = model(images)
                    loss = criterion(logits, labels) / config.accumulation_steps
            scaler.scale(loss).backward()
        else:
            if is_triplet:
                embeddings = model(images)
                loss = criterion(embeddings, labels) / config.accumulation_steps
            elif is_arcface:
                logits = model(images, labels=labels)
                loss = criterion(logits, labels) / config.accumulation_steps
            else:
                logits = model(images)
                loss = criterion(logits, labels) / config.accumulation_steps
            loss.backward()

        accumulation_counter += 1

        # ── Step optimizer every accumulation_steps batches ──────────────
        if accumulation_counter % config.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            ).item()
            grad_norms.append(grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None and scheduler_is_batch_level:
                scheduler.step()

        # ── Metrics ──────────────────────────────────────────────────────
        with torch.no_grad():
            if is_triplet:
                total += labels.size(0)
                running_loss += loss.item() * config.accumulation_steps * labels.size(0)
            else:
                mask = labels != -1
                if mask.sum() > 0:
                    preds = logits[mask].argmax(dim=1)
                    correct += (preds == labels[mask]).sum().item()
                    total += mask.sum().item()
                running_loss += loss.item() * config.accumulation_steps * images.size(0)

        # ── Progress bar ─────────────────────────────────────────────────
        current_loss = running_loss / max(total, 1)
        current_acc = correct / max(total, 1)
        if is_triplet:
            active_ratio = criterion.num_active_triplets / max(criterion.num_valid_triplets, 1)
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "active": f"{active_ratio * 100:.0f}%",
            })
        else:
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc * 100:.1f}%",
            })

    # Handle any remaining gradients from incomplete accumulation
    if accumulation_counter % config.accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        ).item()
        grad_norms.append(grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    lr = optimizer.param_groups[0]["lr"]

    return {
        "train_loss": running_loss / max(total, 1),
        "train_acc": correct / max(total, 1),
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "lr": lr,
    }


def train(config: ExperimentConfig, data: Dict, device: torch.device,
          resume_from: Optional[str] = None) -> ExperimentManager:
    """
    Full training pipeline.

    Args:
        config:      ExperimentConfig with all hyperparameters.
        data:        Dict from dataset.prepare_data().
        device:      Target device.
        resume_from: Experiment ID to resume, "auto" to reuse latest matching
                     config, or None for fresh start.
    """
    set_seed(config.seed)

    requested_auto_resume = (resume_from == "auto")
    completed_auto_match = None

    # Auto-resume mode: find latest experiment with same config hash.
    if resume_from == "auto" or (resume_from is None and config.auto_resume):
        resume_from = ExperimentManager.find_latest_matching_experiment(config)
        requested_auto_resume = True
        if resume_from and ExperimentManager.is_experiment_completed(
            config.experiments_root, resume_from
        ):
            completed_auto_match = resume_from
            resume_from = None

    # If auto-resume found a completed run, skip retraining and reuse it.
    if completed_auto_match is not None:
        manager = ExperimentManager(config, experiment_id=completed_auto_match)
        manager.setup(save_config=False)
        manager.log(
            "Matching experiment is already completed. "
            "Skipping training and reusing existing checkpoints."
        )
        return manager

    # ── Setup experiment manager ─────────────────────────────────────────
    if resume_from:
        manager = ExperimentManager(config, experiment_id=resume_from)
    else:
        manager = ExperimentManager(config)
    manager.setup()

    if resume_from:
        manager.log(f"Resume mode active for experiment: {resume_from}")
    elif requested_auto_resume:
        manager.log("No matching resume target found. Starting new experiment.")

    # ── Build model ──────────────────────────────────────────────────────
    model = build_model(config, data["num_classes"], device)

    # Pre-read latest checkpoint metadata so optimizer phase matches resume state.
    resume_optimizer_groups = None
    if resume_from:
        latest_path = manager.checkpoint_dir / "latest.pth"
        if latest_path.exists():
            resume_state = torch.load(latest_path, map_location="cpu", weights_only=False)
            opt_state = resume_state.get("optimizer_state_dict", {})
            if isinstance(opt_state, dict):
                groups = opt_state.get("param_groups", [])
                if isinstance(groups, list) and len(groups) > 0:
                    resume_optimizer_groups = len(groups)

    # Start with backbone frozen unless resume checkpoint indicates unfrozen optimizer.
    is_frozen = config.freeze_backbone_epochs > 0
    if resume_optimizer_groups == 1:
        is_frozen = True
    elif resume_optimizer_groups is not None and resume_optimizer_groups > 1:
        is_frozen = False

    if is_frozen:
        freeze_backbone(model)
        manager.log(f"Backbone FROZEN for first {config.freeze_backbone_epochs} epochs")
    elif resume_from:
        manager.log("Resume checkpoint indicates UNFROZEN optimizer state.")
    print_model_summary(model, config)

    # ── Build loss ───────────────────────────────────────────────────────
    criterion = build_loss(config, data["class_counts"], device)
    manager.log(f"Loss: {config.loss_type} | Head: {config.head_type}")

    # ── Build dataloaders ────────────────────────────────────────────────
    if config.pk_sampling:
        from dataset import build_metric_dataloaders
        train_loader, val_loader = build_metric_dataloaders(config, data)
        steps_per_epoch = len(train_loader)
    else:
        from dataset import build_dataloaders
        train_loader, val_loader = build_dataloaders(config, data)
        steps_per_epoch = len(train_loader) // config.accumulation_steps

    # ── Build optimizer & scheduler ──────────────────────────────────────
    optimizer = build_optimizer(config, model, is_frozen)

    remaining_epochs = config.epochs
    scheduler, scheduler_is_batch = build_scheduler(
        config, optimizer, steps_per_epoch, remaining_epochs
    )

    # ── AMP setup ────────────────────────────────────────────────────────
    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 0
    if resume_from:
        start_epoch = manager.load_checkpoint(
            model, optimizer, scheduler, scaler, checkpoint="latest"
        )

    # ── Class mappings to save with checkpoints ──────────────────────────
    extra = {
        "id_to_idx": data["id_to_idx"],
        "idx_to_id": data["idx_to_id"],
        "num_classes": data["num_classes"],
    }

    # ── Training loop ────────────────────────────────────────────────────
    effective_bs = config.batch_size * config.accumulation_steps
    if config.pk_sampling:
        effective_bs = config.pk_p * config.pk_k
    manager.log(f"\nStarting training: {config.epochs} epochs, "
                f"effective batch size = {effective_bs}")

    total_start = time.time()
    all_metrics = {}  # Will be set inside the loop

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # ── Phase transition: unfreeze backbone ──────────────────────────
        if is_frozen and epoch >= config.freeze_backbone_epochs:
            unfreeze_backbone(model)
            is_frozen = False
            manager.log(f"\n{'='*60}")
            manager.log(f"BACKBONE UNFROZEN at epoch {epoch}")
            manager.log(f"{'='*60}\n")

            optimizer = build_optimizer(config, model, is_frozen=False)
            scheduler, scheduler_is_batch = build_scheduler(
                config, optimizer, steps_per_epoch,
                config.epochs - epoch,
            )
            if scaler is not None:
                scaler = GradScaler()

            print_model_summary(model, config)

        # ── Train one epoch ──────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scheduler_is_batch, scaler, config, device, epoch,
        )

        # ── Validate ─────────────────────────────────────────────────────
        # For triplet loss, classification-based eval still works — the model
        # has no classifier, but we still extract embeddings and can do retrieval.
        # We pass criterion; evaluate() handles ignore_index for unknown labels.
        if config.loss_type == "triplet":
            # For triplet: do retrieval evaluation instead of classification eval
            from evaluation import evaluate_retrieval
            from dataset import build_dataloaders as build_std_loaders
            gallery_loader, _ = build_std_loaders(config, data)
            val_metrics = evaluate_retrieval(
                model, gallery_loader, val_loader, config, device,
            )
            # Map retrieval metrics to standard names for early stopping
            val_metrics["val_known_acc"] = val_metrics.get("retrieval_recall@1", 0)
            val_metrics["val_loss"] = train_metrics["train_loss"]  # No val loss for triplet
        else:
            val_metrics = evaluate(model, val_loader, criterion, config, device, epoch)

        # ── Step epoch-level scheduler ───────────────────────────────────
        if scheduler is not None and not scheduler_is_batch:
            if config.scheduler == "plateau":
                scheduler.step(val_metrics.get("val_known_acc", 0))
            else:
                scheduler.step()

        # ── Merge and log metrics ────────────────────────────────────────
        all_metrics = {**train_metrics, **val_metrics}
        all_metrics["epoch_time_min"] = (time.time() - epoch_start) / 60
        all_metrics["epoch"] = epoch

        is_best = manager.log_epoch(epoch, all_metrics)

        # ── Print epoch summary ──────────────────────────────────────────
        phase = "FROZEN" if is_frozen else "UNFROZEN"
        acc_key = "val_known_acc"
        nw_key = "val_new_whale_detection"
        if config.loss_type == "triplet":
            acc_key = "retrieval_recall@1"
            nw_key = "retrieval_new_whale_detection"

        manager.log(
            f"Epoch {epoch:>3d}/{config.epochs} [{phase}] | "
            f"Train Loss: {train_metrics['train_loss']:.4f}  "
            f"Acc: {train_metrics['train_acc']*100:.1f}% | "
            f"Val {acc_key}: {val_metrics.get(acc_key, 0)*100:.1f}%  "
            f"NewWhale Det: {val_metrics.get(nw_key, 0)*100:.1f}% | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {all_metrics['epoch_time_min']:.1f}m"
        )

        # ── Save checkpoint ──────────────────────────────────────────────
        manager.save_checkpoint(
            epoch, model, optimizer, scheduler, scaler,
            is_best=is_best, extra=extra,
        )

        # ── Early stopping ───────────────────────────────────────────────
        if manager.should_stop_early(epoch):
            manager.log(f"\nEarly stopping triggered (no improvement for "
                        f"{config.early_stopping_patience} epochs)")
            break

    # ── Wrap up ──────────────────────────────────────────────────────────
    total_time = (time.time() - total_start) / 60
    manager.log(f"\nTraining complete in {total_time:.1f} minutes.")
    manager.log(f"Best epoch: {manager.best_epoch} "
                f"({config.early_stopping_metric}={manager.best_metric_value:.4f})")

    manager.finalize(final_metrics=all_metrics)
    return manager
