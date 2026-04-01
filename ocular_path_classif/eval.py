"""
evaluate.py

Model evaluation for ocular pathology classification.

Responsibilities:
    1. Load best model checkpoint
    2. Run inference on test set
    3. Compute per-class precision, recall, F1 and macro-F1
    4. Generate Grad-CAM heatmaps for visual interpretability
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from ocular_path_classif.config import MODELS_DIR, FIGURES_DIR
from ocular_path_classif.dataset import get_dataloaders
from ocular_path_classif.model import build_model


# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 9
IMAGE_SIZE  = 384
BATCH_SIZE  = 32


# ── Inference ──────────────────────────────────────────────────────────────────

def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> tuple[list, list]:
    """Run inference on a DataLoader and collect predictions and true labels.

    Args:
        model: Trained OcularCNN model in eval mode.
        loader: DataLoader for the split to evaluate (typically test).

    Returns:
        Tuple of (all_preds, all_labels) as flat Python lists of ints.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            logits = model(images)

            # LEARNING NOTE — torch.argmax(dim=1) picks the index of the
            # highest logit per image — that's the predicted class.
            # .cpu() moves the tensor back to CPU before converting to list.
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: list,
    all_labels: list,
    class_names: list,
) -> dict:
    """Compute per-class and macro metrics from predictions and true labels.

    Args:
        all_preds: Flat list of predicted class indices.
        all_labels: Flat list of true class indices.
        class_names: Ordered list of class name strings.

    Returns:
        Dictionary containing:
            - "report": sklearn classification report string
            - "confusion_matrix": numpy array of shape (num_classes, num_classes)
            - "macro_f1": float, macro-averaged F1 score
    """
    # LEARNING NOTE — classification_report gives precision, recall, F1
    # per class plus macro and weighted averages. output_dict=False gives
    # a readable string for logging. zero_division=0 avoids warnings for
    # classes with no predictions.
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    # Extract macro F1 from report dict for logging
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]

    logger.info(f"\n{report}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
    }


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM implementation for OcularCNN.

    Generates a heatmap showing which regions of the input image most
    influenced the model's prediction for a given class.

    LEARNING NOTE — How Grad-CAM works:
        1. Run a forward pass and get the predicted class score
        2. Compute gradients of that score w.r.t. the last conv layer's
           feature maps — these gradients tell us how much each feature
           map channel contributed to the prediction
        3. Global average pool the gradients to get one weight per channel
        4. Take a weighted sum of the feature maps using those weights
        5. ReLU the result (we only care about positive contributions)
        6. Resize to input image size and normalize to [0, 1]

    The result highlights the discriminative regions the model focused on.
    """

    def __init__(self, model: nn.Module):
        """Initialize GradCAM and register hooks on the last conv block.

        Args:
            model: Trained OcularCNN model.
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # LEARNING NOTE — hooks let you intercept the forward and backward
        # pass at any layer without modifying the model code.
        # We attach them to the last ConvBlock (index 3 in self.features)
        # because that layer has the richest semantic feature maps.
        last_block = model.features[-1]

        # Forward hook: captures the output (activations) of the last block
        last_block.register_forward_hook(self._save_activations)

        # Backward hook: captures the gradients flowing back through the last block
        last_block.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook — saves feature map activations."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook — saves gradients of the target class score."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for one image.

        Args:
            image_tensor: Single image tensor of shape (1, 3, H, W) on DEVICE.
            class_idx: Class index to generate the heatmap for. If None,
                uses the predicted class (highest logit).

        Returns:
            Heatmap as a numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()

        # LEARNING NOTE — we need gradients here, so do NOT use torch.no_grad()
        # Enable grad even if model is in eval mode
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # LEARNING NOTE — we only want gradients for the predicted class score,
        # not all classes. zero_grad clears any existing gradients, then we
        # backpropagate only the score for class_idx.
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients: (C, H, W) -> (C,)
        # Each value is the importance weight for one feature map channel
        weights = self.gradients[0].mean(dim=(1, 2))

        # Weighted sum of feature maps: (C, H, W) weighted by (C,) -> (H, W)
        cam = torch.zeros(self.activations.shape[2:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Resize to input image size and normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: Path = None,
    data_dir: Path = None,
) -> dict:
    """Load the best model and evaluate on the test set.

    Args:
        checkpoint_path: Path to saved model weights (.pt file).
            Defaults to MODELS_DIR/best_model.pt.
        data_dir: Path to Original_Dataset folder.

    Returns:
        Dictionary containing:
            - "metrics": output of compute_metrics()
            - "class_names": list of class name strings
            - "all_preds": flat list of predicted class indices
            - "all_labels": flat list of true class indices
            - "model": loaded model (for Grad-CAM use in notebook)
            - "test_loader": test DataLoader (for Grad-CAM use in notebook)
    """
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "best_model.pt"

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)

    # LEARNING NOTE — load_state_dict loads saved weights into the model.
    # map_location ensures weights load correctly regardless of whether
    # they were saved on GPU but you're now running on CPU or vice versa.
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    logger.info(f"Test batches: {len(test_loader)}")

    # ── Inference and metrics ─────────────────────────────────────────────
    all_preds, all_labels = get_predictions(model, test_loader)
    metrics = compute_metrics(all_preds, all_labels, class_names)

    return {
        "metrics": metrics,
        "class_names": class_names,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "model": model,
        "test_loader": test_loader,
    }