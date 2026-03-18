"""
dataset.py

PyTorch Dataset and DataLoader setup

Responsibilities:
    1. Scan data/raw/Original_Dataset and build (image_path, label) pairs
    2. Drop Pterygium (anterior segment image)
    3. Split into 80/10/10 split
    4. Return DataLoaders ready from training
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from loguru import logger

from ocular_path_classif.config import RAW_DATA_DIR
from ocular_path_classif.transforms import get_transforms


EXCLUDED_CLASSES = {"Pterygium"}

MINORITY_CLASSES = {
    "Central Serous Chorioretinopathy", # 101
    "Disc Edema",                       # 127
    "Retinal Detachment",               # 125
    "Retinitis Pigmentosa",             # 139
}

def _scan_dataset(data_dir: Path) -> tuple[list, list]:
    """
    Build list of (image_path, label_index) pairs.

    Parameters
    - data_dir : Path
        Path directory of raw original dataset

    Returns
    - samples : list of (Path, int)
    - class_names : sorted list of class names
    """

    # Sorts directory names in data_dir that is not excluded
    class_names = sorted([
        dir.name for dir in data_dir.iterdir()
        if dir.is_dir() and dir.name not in EXCLUDED_CLASSES
    ])

    logger.info(f"Found {len(class_names)} classes (Pterygium excluded)")
    logger.info(f"Classes: {class_names}")

    # Stores all images paths and class index in samples
    samples = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for img_path in image_files:
            samples.append((img_path, label_idx))

        logger.info(f"{class_name}: {len(image_files)} images (label={label_idx})")
    
    logger.info(f"Total samples: {len(samples)}")

    return samples, class_names

def _make_weighted_sampler(train_samples: list, class_names: list) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so minority classes appear more in each batch.
    Changes sampling frequency based on a weight for each sample = 1 / (count of its class).

    Parameters
    - train_samples : list
        All samples in training set
    - class_names : list
        Sorted and non-excluded class names

    Returns
    - WeightedRandomSampler
        Weights for each sample
    """

    # Count how many samples exist for each class
    labels = [label for (_, label) in train_samples]
    class_counts = np.bincount(labels, minlength=len(class_names))

    logger.info("Class counts in training set:")
    for name, count in zip(class_names, class_counts):
        logger.info(f" {name}: {count}")

    # Prevent division by zero
    eps = 1e-6
    class_weights = 1.0 / (class_counts + eps)
    sample_weights = [class_weights[label] for (_, label) in train_samples]

    return WeightedRandomSampler(
        weights=sample_weights,         # Sets weight proportional to class size
        num_samples=len(train_samples), # Draws n train_samples from weighted sampler per epoch
        replacement=True,               # Allows same image to be drawn each epoch (important for minority classes)
    )

class OcularDataset(Dataset):
    """
    PyTorch Dataset

    DataLoader requires:
    - __len__ : returns total number of samples
    - __getitem__ : returns one (image_tensor, label) pair at a given index
    """

    def __init__(
        self,
        samples: list,
        class_names: list,
        split: str = "train",
        image_size: int = 384,
    ):
        self.samples = samples
        self.class_names = class_names
        self.split = split
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Implementing lazy loading to prevent loading all into RAM at once
        # Apply transformations each time
        img_path, label = self.samples[idx]
        class_name = self.class_names[label]

        # Convert to RGB explicitly
        image = np.array(Image.open(img_path).convert("RGB"))

        # Flag for if it's a minority class
        is_minority = (self.split == "train") and (class_name in MINORITY_CLASSES)

        # Apply correct transform pipeline
        transform = get_transforms(
          split=self.split,
          image_size=self.image_size,
          minority_flag=is_minority,  
        )

        # A.Compose returns a dict '{"image": tensor}' so index
        tensor = transform(image=image)["image"]

        return tensor, label
    
def get_dataloaders(
    data_dir: Optional[Path] = None,
    image_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = .10,
    test_size: float = .10,
    random_state: int = 80,
) -> tuple:
    """
    Build and return train, val, and test Dataloaders.
    
    Parameters
    - data_dir : Path
        Root folder containing class directories
    - image_size : int
        Default resize target for all images
    - batch_size : int
        Number of images per batch
    - num_workers : int
        Parallel workers for data loading
    - val_size : float
        Fraction of data for validation set
    - test_size : float
        Fraction of data for test set
    - random_state : int
        Seed for reproducibility (80 is eyes and a mouth)

    Returns
    - train_loader, val_loader, test_loader : DataLoader
    - class_names : list of str
    """

    if data_dir is None:
        data_dir = RAW_DATA_DIR / "Original_Dataset"
    
    # Scan all image files
    all_samples, class_names = _scan_dataset(data_dir)

    # Stratified split
    indices = list(range(len(all_samples)))
    labels = [label for _, label in all_samples]

    # Create test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Create val set

    adjusted_val = val_size / (1.0 - test_size)

    train_val_labels = [labels[i] for i in train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=adjusted_val,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # Build sample lists for each split
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    logger.info(f"Split sizes - train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    # Build Dataset objects
    train_dataset = OcularDataset(train_samples, class_names, split="train", image_size=image_size)
    val_dataset = OcularDataset(val_samples, class_names, split="val", image_size=image_size)
    test_dataset = OcularDataset(test_samples, class_names, split="test", image_size=image_size)

    # Make weighted sampler for training set
    weighted_sampler = _make_weighted_sampler(train_samples, class_names)

    # Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=True, # Keep loaded batches in pinned CPU memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,   # deterministic order
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names

