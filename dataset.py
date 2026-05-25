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

from config import ExperimentConfig


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


def split_data(df: pd.DataFrame, seed: int = 42):
    """
    Deterministic per-class train/val/test split for long-tail re-ID data.

    Rules:
      - new_whale: excluded from train, split 50/50 into val/test
      - 1 image:   train only
      - 2-3:       train + 1 test
      - 4+:        train + 1 val + 1 test (proportional for larger classes)

    Returns train_df, val_df, test_df.
    """
    rng = np.random.RandomState(seed)
    df = df.sort_values("Image").reset_index(drop=True)

    new_whale_df = df[df["Id"] == "new_whale"].reset_index(drop=True)
    known_df = df[df["Id"] != "new_whale"].reset_index(drop=True)

    train_idx, val_idx, test_idx = [], [], []

    for _, group in known_df.groupby("Id"):
        rows = group.index.tolist()
        rng.shuffle(rows)
        n = len(rows)

        # We do a little gambling to decide what ends up in test 
        if n == 1:
            n_test = 0
        elif n == 2:
            n_test = 1 if rng.random() < 0.33 else 0
        elif n == 3:
            n_test = 1 if rng.random() < 0.66 else 0
        else:
            n_test = max(1, round(n * 0.1))
        remaining = n - n_test
        n_val = max(1, round(n * 0.1)) if remaining >= 3 else 0

        train_idx.extend(rows[: n - n_test - n_val])
        val_idx.extend(rows[n - n_test - n_val : n - n_test])
        test_idx.extend(rows[n - n_test :])

    # new_whale → 50/50 val/test
    nw_idx = list(range(len(new_whale_df)))
    rng.shuffle(nw_idx)
    mid = len(nw_idx) // 2

    train = known_df.iloc[train_idx].reset_index(drop=True)
    val = pd.concat([known_df.iloc[val_idx], new_whale_df.iloc[nw_idx[:mid]]], ignore_index=True)
    test = pd.concat([known_df.iloc[test_idx], new_whale_df.iloc[nw_idx[mid:]]], ignore_index=True)

    return train, val, test


def prepare_data(config) -> Dict:
    """Load CSV, split, build class mappings and loss-weighting counts."""
    df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
    train, val, test = split_data(df, seed=config.seed)

    # ── Class mappings from training set ────────────────────────────────
    train_labels = sorted(train["Id"].unique().tolist())
    id_to_idx = {name: i for i, name in enumerate(train_labels)}
    idx_to_id = {i: name for name, i in id_to_idx.items()}
    num_classes = len(id_to_idx)

    # ── Class counts for loss weighting ─────────────────────────────────
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for whale_id, count in train["Id"].value_counts().items():
        class_counts[id_to_idx[whale_id]] = count

    print(f"Split (seed={config.seed}): "
          f"train={len(train)} ({num_classes} cls), "
          f"val={len(val)} ({(val['Id'] == 'new_whale').sum()} new_whale), "
          f"test={len(test)} ({(test['Id'] == 'new_whale').sum()} new_whale)")

    return {
        "train_df": train,
        "val_df": val,
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


class OpenSetRandomSampler(torch.utils.data.Sampler):
    """
    Shuffles ALL indices (including rare whales and new_whale classes)
    into standard sequential batches to maximize the negative pool
    available for online semi-hard triplet mining.
    """

    def __init__(self, labels, batch_size: int = 128):
        self.batch_size = batch_size
        self.all_indices = list(range(len(labels)))
        self.total_images = len(self.all_indices)
        self._num_batches = max(self.total_images // self.batch_size, 1)

    def __iter__(self):
        shuffled_indices = np.random.permutation(self.all_indices)
        for i in range(self._num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            yield shuffled_indices[start_idx:end_idx].tolist()

    def __len__(self):
        return self._num_batches


def build_metric_dataloaders(config, data: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for metric learning with PK sampling or open-set random batching.

    Train loader uses PK sampling (P identities x K images per identity) when enabled.
    Otherwise, it uses open-set random batching to maximize the negative pool.
    Val loader is standard (same as classification -- used for retrieval eval).
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

    batch_sampler: PKSampler | OpenSetRandomSampler
    
    if config.pk_sampling:
        batch_sampler = PKSampler(
            labels=train_labels,
            p=config.pk_p,
            k=config.pk_k,
            min_samples=config.pk_min_samples,
        )

        usable_classes = len(batch_sampler.classes)
        total_classes = data["num_classes"]
        print(f"PK Sampling: {usable_classes}/{total_classes} classes have ≥{config.pk_min_samples} images")
        print(f"  Batch: {config.pk_p} identities × {config.pk_k} images = {batch_sampler.batch_size} per batch")
        print(f"  ~{len(batch_sampler)} batches per epoch")
    else:
        batch_sampler = OpenSetRandomSampler(
            labels=train_labels,
            batch_size=config.batch_size,
        )
        print(f"Open-set random sampling: {batch_sampler.batch_size} images per batch")
        print(f"  ~{len(batch_sampler)} batches per epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
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

def build_retrieval_eval_loaders(config, data):
    """Deterministic gallery/query loaders for comparable retrieval metrics."""
    image_dir = os.path.join(config.data_dir, "train")
    eval_transform = build_val_transform(config)

    gallery_dataset = WhaleDataset(
        df=data["train_df"],
        image_dir=image_dir,
        id_to_idx=data["id_to_idx"],
        transform=eval_transform,
    )
    query_dataset = WhaleDataset(
        df=data["val_df"],
        image_dir=image_dir,
        id_to_idx=data["id_to_idx"],
        transform=eval_transform,
    )

    batch_size = config.batch_size * 2
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return gallery_loader, query_loader

def get_eval_config_and_data() -> tuple[ExperimentConfig, dict]:
    """
    Returns
        standard eval config
        data
    
    """
    config = ExperimentConfig(
        data_dir="data",
        seed=42,
    )
    data = prepare_data(config)

    return config, data

def build_test_loader(config, data: Dict) -> DataLoader:
    """
    Build DataLoader for the test set.
    
    Automatically preprocesses the test DataFrame so that any whale identities 
    not seen in the training set are reclassified as 'new_whale'.
    """
    image_dir = os.path.join(config.data_dir, "train")
    test_df = data["test_df"].copy()
    
    test_dataset = WhaleDataset(
        df=test_df,
        image_dir=image_dir,
        id_to_idx=data["id_to_idx"],
        transform=build_val_transform(config),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return test_loader