"""
evaluate.py

Model evaluation for classification

Roadmap:
    1. Load best model checkpoint
    2. Run inference on test set
    3. Compute per-class accuracy, precision, recall, F1, and macro-F1
    4. Return results for plotting
"""

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from ocular_path_classif.config import MODELS_DIR, REPORTS_DIR
from ocular_path_classif.dataset import get_dataloaders
from ocular_path_classif.model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 9
IMAGE_SIZE = 256
BATCH_SIZE = 64

def _get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> tuple[list, list]:
    """Run inference on test set and collect predictions and true labels.
    
    Args:
        model: Trained OcularCNN in eval mode.
        loader: Test DataLoader.
        
    Returns:
        Tuple of (all_preds, all_labels) as flat Python lists of ints.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Loop through test batches
        for images, labels in loader:
            images = images.to(DEVICE)
            
            # Forward pass
            logits = model(images)

            # Pick highest score among classes
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            # Add batch preds and true labels to list
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
        
    return all_preds, all_labels

def evaluate(
    checkpoint_path: Path = None,
    data_dir: Path = None,
) -> dict:
    """Load best model and evaluate on test set.
    
    Args:
        checkpoint_path: Path to saved model weights (.pt)
        data_dir: Path to Original_Dataset folder
    
    Returns:
        Dictionary containing:
            - "report": per-class metrics
            - "confusion_matrix": numpy array
            - "macro_f1": macro-averaged F1 score
            - "class_names":  list of class name strings
            - "all_preds": list of pred class indicies
            - "all_labels": true class indicies
    """

    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "latest_run_model.pt"
    logger.info(f"Testing on data directory: {data_dir}")
    # Creates OcularCNN object and loads the trained weights into it
    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Load test loader and labels
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    logger.info(f"Test batches: {len(test_loader)}")

    # Run inference on test set
    all_preds, all_labels = _get_predictions(model, test_loader)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(all_labels, all_preds)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    
    logger.info(f"\n{report}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
        "class_names": class_names,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }

if __name__ == "__main__":
    results = evaluate()
    print(results["report"])

    report_path =  REPORTS_DIR / "latest_run_report002.txt"
    with open(report_path, "w") as f:
        f.write(results["report"])

    logger.info(f"Report saved to {report_path}")