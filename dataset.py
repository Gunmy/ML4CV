"""
Data pipeline for Humpback Whale Identification.

Handles:
  - Loading and splitting the CSV with robust stratification
  - Whale-specific augmentations (conservative — preserving identity cues)
  - Proper new_whale handling across train/val/test splits
  - Standard DataLoader construction
  - PK Sampling DataLoader for metric learning (triplet / contrastive loss)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# ── Augmentation Pipelines ───────────────────────────────────────────────

def build_train_transform(config) -> transforms.Compose:
    """
    Training augmentations designed for whale fluke identification.

    Philosophy: simulate realistic viewing variation (lighting, angle, distance)
    WITHOUT destroying identity cues (notch shapes, pigmentation patterns,
    trailing-edge contours).

    What we DO:
      - Mild rotation (±10°): whales surface at slightly different angles
      - Brightness/contrast jitter: sea conditions vary
      - RandomResizedCrop with conservative scale: slight framing differences
      - Gaussian blur at low sigma: distance/focus variation

    What we DON'T:
      - Horizontal flip: left/right notch patterns are distinct per individual
      - RandomErasing/CutOut: could erase the exact notch that identifies the whale
      - Heavy geometric distortion: changes trailing-edge shape
      - Aggressive color changes: destroys pigmentation contrast
    """
    ops = []

    if config.aug_random_resized_crop:
        ops.append(transforms.RandomResizedCrop(
            config.image_size,
            scale=(config.aug_crop_scale_min, config.aug_crop_scale_max),
            ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ))
    else:
        ops.append(transforms.Resize(config.image_size,
                                     interpolation=transforms.InterpolationMode.BICUBIC))

    ops.append(transforms.RandomRotation(degrees=config.aug_rotation_degrees))

    ops.append(transforms.ColorJitter(
        brightness=config.aug_brightness,
        contrast=config.aug_contrast,
    ))

    if config.aug_gaussian_blur:
        ops.append(transforms.GaussianBlur(
            kernel_size=3,
            sigma=(config.aug_blur_sigma_min, config.aug_blur_sigma_max),
        ))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(ops)


def build_val_transform(config) -> transforms.Compose:
    """Deterministic transform for validation / test — no augmentation."""
    return transforms.Compose([
        transforms.Resize(config.image_size,
                           interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────

class WhaleDataset(Dataset):
    """
    Dataset for whale fluke images.

    Each sample returns (image_tensor, label) where:
      - label >= 0  for known whale identities (mapped to contiguous indices)
      - label == -1 for new_whale (unknown identity)
    """

    def __init__(self, df: pd.DataFrame, image_dir: str,
                 id_to_idx: Dict[str, int],
                 transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_to_idx = id_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image"])
        image = Image.open(img_path).convert("RGB")

        whale_id = row["Id"]
        label = self.id_to_idx.get(whale_id, -1)  # -1 for unknown / new_whale

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Robust Stratification Helpers ────────────────────────────────────────

def _can_stratify(labels: pd.Series, test_size, ) -> bool:
    """Check sklearn stratified split feasibility for a binary partition."""
    n_samples = len(labels)
    n_classes = labels.nunique()
    if n_samples < 2 or n_classes < 2:
        return False

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test

    min_class_count = int(labels.value_counts().min())
    return n_test >= n_classes and n_train >= n_classes and min_class_count >= 2


# ── Data Splitting ───────────────────────────────────────────────────────

def prepare_data(config) -> Dict:
    """
    Load CSV, split into train/val/test, build class mappings.

    Uses a hybrid stratification strategy:
      - Classes with ≥3 images: stratified split (preserves class proportions)
      - Rare classes (<3 images): kept in training to avoid impossible constraints
      - Falls back to shuffled split when stratification isn't feasible

    Returns a dict with:
      - train_df, val_df, test_df: DataFrames
      - id_to_idx, idx_to_id: class mappings (known whales only)
      - class_counts: per-class sample count in training set
      - num_classes: number of known whale identities
    """
    csv_path = os.path.join(config.data_dir, "train.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values("Image").reset_index(drop=True)

    # ── Split into train_val / test ──────────────────────────────────────
    if config.stratified_split:
        # Hybrid strategy:
        # - Stratify only classes with enough samples for stable splitting.
        # - Keep ultra-rare classes in training to avoid impossible constraints.
        class_counts_raw = df["Id"].value_counts()
        stratifiable_ids = class_counts_raw[class_counts_raw >= 3].index

        common_df = df[df["Id"].isin(stratifiable_ids)].reset_index(drop=True)
        rare_df = df[~df["Id"].isin(stratifiable_ids)].reset_index(drop=True)

        if len(common_df) < 2 or common_df["Id"].nunique() < 2:
            print("[split] Too few stratifiable classes; using shuffled split on full dataset.")
            train_val, test = train_test_split(
                df, test_size=config.test_split,
                random_state=config.seed, shuffle=True,
            )
        else:
            stratify_common = common_df["Id"]
            if _can_stratify(stratify_common, config.test_split):
                train_val, test = train_test_split(
                    common_df, test_size=config.test_split,
                    random_state=config.seed, shuffle=True,
                    stratify=stratify_common,
                )
            else:
                print("[split] Stratified test split infeasible on frequent classes; using shuffled split.")
                train_val, test = train_test_split(
                    common_df, test_size=config.test_split,
                    random_state=config.seed, shuffle=True,
                )

            # Merge rare samples back into train_val (they all go to training)
            if len(rare_df) > 0:
                train_val = pd.concat([train_val, rare_df], axis=0, ignore_index=True)
                train_val = train_val.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        # ── Split train_val into train / val ─────────────────────────────
        val_relative = config.val_split / (1 - config.test_split)

        train_val_common = train_val[train_val["Id"].isin(stratifiable_ids)].reset_index(drop=True)
        train_val_rare = train_val[~train_val["Id"].isin(stratifiable_ids)].reset_index(drop=True)

        if len(train_val_common) < 2:
            train_common = train_val_common.copy()
            val = train_val_common.iloc[0:0].copy()
            print("[split] No stratifiable samples available for val split; validation set from common is empty.")
        elif len(train_val_common) >= 2 and train_val_common["Id"].nunique() >= 2 and \
                _can_stratify(train_val_common["Id"], val_relative):
            train_common, val = train_test_split(
                train_val_common, test_size=val_relative,
                random_state=config.seed, shuffle=True,
                stratify=train_val_common["Id"],
            )
        else:
            print("[split] Stratified val split infeasible on frequent classes; using shuffled split.")
            train_common, val = train_test_split(
                train_val_common, test_size=val_relative,
                random_state=config.seed, shuffle=True,
            )

        if len(train_val_rare) > 0:
            train = pd.concat([train_common, train_val_rare], axis=0, ignore_index=True)
            train = train.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
        else:
            train = train_common

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

    else:
        train_val, test = train_test_split(
            df, test_size=config.test_split, random_state=config.seed, shuffle=True)
        val_relative = config.val_split / (1 - config.test_split)
        train, val = train_test_split(
            train_val, test_size=val_relative, random_state=config.seed, shuffle=True)

    # ── Build class mappings from TRAINING known whales only ─────────────
    train_known = train[train["Id"] != "new_whale"]

    # Optionally filter out extremely rare classes
    if config.min_samples_per_class > 1:
        counts = train_known["Id"].value_counts()
        valid_ids = counts[counts >= config.min_samples_per_class].index
        train_known = train_known[train_known["Id"].isin(valid_ids)]

    train_labels = sorted(train_known["Id"].unique().tolist())
    id_to_idx = {name: i for i, name in enumerate(train_labels)}
    idx_to_id = {i: name for name, i in id_to_idx.items()}
    num_classes = len(id_to_idx)

    # For training: exclude new_whale (can't train a classifier on "unknown")
    train_final = train[train["Id"] != "new_whale"].reset_index(drop=True)
    # Further filter to only classes in our mapping
    train_final = train_final[train_final["Id"].isin(id_to_idx)].reset_index(drop=True)

    # For validation: keep everything (including new_whale) if configured
    if config.include_new_whale_in_val:
        val_final = val.reset_index(drop=True)
    else:
        val_final = val[val["Id"] != "new_whale"].reset_index(drop=True)
        val_final = val_final[val_final["Id"].isin(id_to_idx)].reset_index(drop=True)

    # Compute class counts for loss weighting
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for whale_id, count in train_final["Id"].value_counts().items():
        if whale_id in id_to_idx:
            class_counts[id_to_idx[whale_id]] = count

    # ── Summary stats ────────────────────────────────────────────────────
    n_new_train = (train["Id"] == "new_whale").sum()
    n_new_val = (val["Id"] == "new_whale").sum()
    n_new_test = (test["Id"] == "new_whale").sum()

    print(f"Dataset split:")
    print(f"  Train:  {len(train_final):>6d} images, {num_classes} known classes "
          f"(filtered {n_new_train} new_whale)")
    print(f"  Val:    {len(val_final):>6d} images "
          f"({n_new_val} are new_whale)")
    print(f"  Test:   {len(test):>6d} images "
          f"({n_new_test} are new_whale)")
    if len(class_counts) > 0:
        print(f"  Class distribution: min={int(class_counts.min())} "
              f"median={int(np.median(class_counts))} "
              f"max={int(class_counts.max())} samples/class")

    return {
        "train_df": train_final,
        "val_df": val_final,
        "test_df": test,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "class_counts": class_counts,
        "num_classes": num_classes,
    }


# ── DataLoader Construction ──────────────────────────────────────────────

def build_dataloaders(config, data: Dict) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders (standard random batching)."""
    image_dir = os.path.join(config.data_dir, "train")

    train_dataset = WhaleDataset(
        df=data["train_df"],
        image_dir=image_dir,
        id_to_idx=data["id_to_idx"],
        transform=build_train_transform(config),
    )

    val_dataset = WhaleDataset(
        df=data["val_df"],
        image_dir=image_dir,
        id_to_idx=data["id_to_idx"],
        transform=build_val_transform(config),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid tiny last batch that destabilizes BN
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Can use larger batch for eval (no grads)
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ── PK Sampling for Metric Learning ──────────────────────────────────────

class PKSampler(torch.utils.data.Sampler):
    """
    PK Batch Sampler: samples P identities, K images per identity per batch.
    (Course slide 24 — FaceNet's batching strategy)

    This ensures every mini-batch contains multiple images per identity,
    which is required for forming valid anchor-positive pairs for triplet
    loss or contrastive loss.

    Classes with fewer than K images have their images repeated to reach K.
    Classes with fewer than min_samples images are excluded entirely.

    Args:
        labels:      List/array of integer class labels for the dataset.
        p:           Number of identities per batch.
        k:           Number of images per identity per batch.
        min_samples: Minimum images a class needs to be included.
    """

    def __init__(self, labels, p: int = 16, k: int = 4, min_samples: int = 2):
        self.p = p
        self.k = k

        # Build a dict: class_idx → list of dataset indices
        self.class_to_indices = {}
        for idx, label in enumerate(labels):
            label_int = label if isinstance(label, int) else int(label)
            if label_int < 0:
                continue  # Skip new_whale / unknown
            if label_int not in self.class_to_indices:
                self.class_to_indices[label_int] = []
            self.class_to_indices[label_int].append(idx)

        # Filter out classes with too few samples
        self.class_to_indices = {
            c: idxs for c, idxs in self.class_to_indices.items()
            if len(idxs) >= min_samples
        }
        self.classes = list(self.class_to_indices.keys())

        if len(self.classes) < p:
            print(f"  WARNING: Only {len(self.classes)} classes with ≥{min_samples} "
                  f"images, but p={p}. Reducing p to {len(self.classes)}.")
            self.p = len(self.classes)

        self.batch_size = self.p * self.k
        # Approximate number of batches per epoch
        total_images = sum(len(v) for v in self.class_to_indices.values())
        self._num_batches = max(total_images // self.batch_size, 1)

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = []
            selected_classes = np.random.choice(
                self.classes, size=self.p, replace=False
            )
            for c in selected_classes:
                indices = self.class_to_indices[c]
                if len(indices) >= self.k:
                    chosen = np.random.choice(indices, size=self.k, replace=False)
                else:
                    # Repeat images to fill K slots
                    chosen = np.random.choice(indices, size=self.k, replace=True)
                batch.extend(chosen.tolist())
            yield batch

    def __len__(self):
        return self._num_batches


def build_metric_dataloaders(config, data: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for metric learning with PK sampling.

    Train loader uses PK sampling (P identities × K images per identity).
    Val loader is standard (same as classification — used for retrieval eval).
    """
    image_dir = os.path.join(config.data_dir, "train")
    id_to_idx = data["id_to_idx"]

    # Get labels as integers for the sampler
    train_df = data["train_df"]
    train_labels = [id_to_idx.get(row["Id"], -1) for _, row in train_df.iterrows()]

    train_dataset = WhaleDataset(
        df=train_df,
        image_dir=image_dir,
        id_to_idx=id_to_idx,
        transform=build_train_transform(config),
    )

    pk_sampler = PKSampler(
        labels=train_labels,
        p=config.pk_p,
        k=config.pk_k,
        min_samples=config.pk_min_samples,
    )

    usable_classes = len(pk_sampler.classes)
    total_classes = data["num_classes"]
    print(f"PK Sampling: {usable_classes}/{total_classes} classes have ≥{config.pk_min_samples} images")
    print(f"  Batch: {config.pk_p} identities × {config.pk_k} images = {pk_sampler.batch_size} per batch")
    print(f"  ~{len(pk_sampler)} batches per epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_dataset = WhaleDataset(
        df=data["val_df"],
        image_dir=image_dir,
        id_to_idx=id_to_idx,
        transform=build_val_transform(config),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
