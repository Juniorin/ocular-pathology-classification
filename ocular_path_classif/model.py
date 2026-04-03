"""
model.py

CNN for ocular pathology classification.

Architecture:
    - 4 conv blocks (32, 64, 128, 256): dataset size ~5k images -> 3-5 blocks, 
                                        analyzing fine details needed, 
                                        class similiarity=more depth needed,
                                        won't know if overfits until after training->regulariziation to minimize

    - Channels double each block:  spatial resolution halves via MaxPool,
                                    keep info capacity roughly constant
    - BatchNorm after every Conv2d: stabilizes training,
                                    regularizer
    - AdaptiveAvgPool2d(1, 1): avoids too many parameters->tries to avoid overfitting
    - Dropout(0.5): regularization constant
    - Two Linear layers (256->128->9): enough capacity to separate 9 classes
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Creates a convolutional block for the CNN: Conv2d -> BatchNorm -> ReLU -> MaxPool"""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize ConvBlock layers.
        
        Args:
            in_channels: Number of input feature map channels
            out_channels: Number of output feature map channels
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv -> BN -> ReLU -> MaxPool.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).
        
        Returns:
            Output tensor of shape (B, out_channels, H/2, W/2).
        """
        return self.block(x)
    
class OcularCNNModel(nn.Module):
    """CNN Model for classification."""

    def __init__(self, num_classes: int=9, dropout_rate: float=0.5):
        """Initialize OcularCNN layers.
        
        Args:
            num_classes: Number of output classes.
            dropout_rate: Dropout probability in the classifier head.
        """
        super().__init__()

        # Each block doubles channels, but halves spatial dimensions 384->192->96->48->24
        self.features = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )

        # Collapse spatial dims from (B, 256, 24, 24) -> (B, 256, 1, 1) to prevent overfitting
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(), # (B, 256, 1, 1) -> (B, 256)
            nn.Linear(512, 256), # Combines 512 features into 256
            nn.ReLU(inplace=True), # Non-linearity for two linear layers
            nn.Dropout(p=dropout_rate), # Randomly zeroes p of neurons to prevent overfitting
            nn.Linear(256, num_classes), # Map 256 neurons to 9 class scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.
        
        Args:
            x: Batch of images, shape (B, 3, H, W)
        
        Returns:
            Raw class scores (logits) of shape (B, num_classes)
        """

        x = self.features(x)    # (B, 3, 384, 384) -> (B, 256, 24, 24)
        x = self.pool(x)        # (B, 256, 24, 24) -> (B, 256, 1, 1)
        x = self.classifier(x)  # (B, 256, 1, 1)   -> (B, 9)

        return x
    
def build_model(num_classes: int=9, dropout_rate: float=0.5) -> OcularCNNModel:
    """Instantiate and return an OcularCNN model.
    
    Args:
        num_classes: Number of output classes.
        dropout_rate: Dropout probability in the classifier.
        
    Returns:
        Initialized OcularCNN model with all weights set to PyTorch defaults.
    """

    model = OcularCNNModel(num_classes=num_classes, dropout_rate=dropout_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OcularCNN - trainable parameters: {total_params:,}")

    return model

    