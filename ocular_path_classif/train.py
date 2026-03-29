"""
train.py

Training loop

Roadmap:
    1. Build DataLoaders, model, loss func, and optim
    2. Train for N epochs with train and val phases
    3. Checkpoint best model by val loss to prevent overfitting
    4. Return loss history for plotting
"""

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from ocular_path_classif.config import MODELS_DIR
from ocular_path_classif.dataset import get_dataloaders
from ocular_path_classif.model import build_model

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Training constants
NUM_CLASSES = 9
IMAGE_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 8

def _train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch and return average loss.
    
    Args:
        model: The CNN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        
    Returns:
        Average training loss across all batches.
    """
    
    model.train()

    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

def _val_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
) -> float:
    """Run one validation epoch and return average loss.
    
    Args:
        model: The CNN model.
        loader: Validation Dataloader.
        criterion: Loss function.
        
    Returns:
        Average validation loss across all batches.
    """

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
        
    return total_loss / len(loader)


def train(
    data_dir: Path = None,
    checkpoint_path: Path = None,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> dict:
    """Train OcularCNN and return loss history.
    
    Args:
        data_dir: Path to Original Dataset folder.
        checkpoint_path: Where to save best model weights.
        num_epochs: Maximum number of training epochs.
        batch_size: Images per batch.
        learning_rate: Adam learning rate.
        patience: Early stopping patience in epochs.
    
    Returns:
        Dict with keys "train_loss" and "val_loss" used for plotting.
    """
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "best_model.pt"
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        num_workers=4,
    )

    logger.info(f"Classes: {class_names}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)

    counts = torch.zeros(NUM_CLASSES)
    for _, labels in train_loader:
        for label in labels:
            counts[label] += 1
        
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info(f"Starting training for {num_epochs} epochs on {DEVICE}.")

    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = _val_one_epoch(model, val_loader, criterion)
    
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
    
        logger.info(
            f"Epoch {epoch:>3}/{num_epochs} |"
            f"train loss: {train_loss:.4f} |"
            f"val loss: {val_loss:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f" val loss improved -> saved checkpoint")
        else:
            epochs_without_improvement += 1
            logger.info(f" no improvement ({epochs_without_improvement}/{patience})")
        
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    logger.success(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.success(f"Best model saved to: {checkpoint_path}")

    return history

if __name__ == "__main__":
    history = train()

    import json
    with open("history.json", "w") as f:
        json.dump(history, f)