"""
transforms.py

Albumentations augmentation for ocular pathology classification.

Explanation for albumentation over torchvision transforms:
 - albumentations operates on numpy arrays
 - significantly faster with OpenCV
 - Easy conversion from numpy -> torch tensor via ToTensorV2
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet mean and std used to normalize RGB images.
# Normalize each channel as: (x - mean) / std.
# Standard baseline for pre-trained models -> optionally replace with dataset-specific stats.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

def _spatial_augments() -> list:
    """
    Geometric transformations for fundus images.

    Accounts for variations in the actual fundus imaging process.
    
    The p= argument controls the probability the transform is applied to an image.
    """

    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=0, p=0.7), # rotates +/- n limit degrees and fills new pixels with 0/black
        A.Perspective(scale=(0.02, 0.05), p=0.3), # warps perspective by 2-5% of image size
    ]

def _color_augments() -> list:
    """
    Photometric transformations for fundus images.

    Color augmentations changes pixel values not geometry.
    This accounts for illumination/lighting variances in our EDA.
    """

    return [
        A.CLAHE(clip_limit=3, tile_grid_size=(8, 8), p=0.5), # Enhances contrast to view vessels and retinal structure better
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6), # +/- 20% brightness & contrast change
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4), # Slight variations in hue but not too much as it is important in diagnostics
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3), # Small noise to simulate photography noise
        
        # Randomly choose a blurring augmentation for % of the images (10%)
        A.OneOf([
            A.GaussianBlur(blur_limit=(2, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0)
            ], 
            p=0.2),
    ]

def _minority_classes_extra_augments() -> list:
    """
    More variations for minority classes (online oversampling) to improve dataset balance.
    
    Coarse Dropout - Randomly blacks out small regions of the image to strengthen feature extraction.

    Grid Distortion - Slightly warps image to mimic retinal surface curvature.
    """

    return [
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, p=0.4), # 1-3 16-32 pixel black squares
        A.GridDistortion(num_steps=5, distort_limit=0.15, border_mode=0, p=0.2), # Mild distortion with black edges
    ]

def get_transforms(split: str = "train", image_size: int = 384, minority_flag: bool = False) -> A.Compose:
    """
    Return the augmentation pipeline for the given dataset split.
    
    Parameters
     - split : "train" | "val" | "test"
        Train gets full augmentation. Val/test get resize + normalize only
        (we do not augment during evaluation, else it would make noisy and unreliable metrics)
    
    - image_size : int
        All images are resized to (image_size x image_size). 384 is a good balance for fundus images
    
    - minority_flag : bool
        Extra augmentation for minority classes if True
    
    Returns
    - A.Compose
        A pipeline of augmentations for a NumPy image -> PyTorch tensor    
    """
    
    split = split.lower().strip()

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split!r}")
    
    if split == "train":
        # Build the list of transformations dynamically
        aug_list = (
            _spatial_augments()
            + _color_augments()
            + (_minority_classes_extra_augments() if minority_flag else [])
        )

        return A.Compose([
            A.Resize(image_size, image_size), # Resize before augments to save resources
            *aug_list, # Unpacks list of transforms 
            A.Normalize(mean=_MEAN, std=_STD), # Normalizes colors
            ToTensorV2(), 
        ])
    
    else:
        # No augmentations to measure model on real data
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])


