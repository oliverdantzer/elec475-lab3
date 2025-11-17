from pathlib import Path
import os
from typing import Callable, Optional, Tuple, cast
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import numpy as np
from torchvision.models.segmentation import FCN_ResNet50_Weights
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Get dataset path from environment variable
dataset_path = Path(os.getenv("DATASET_PATH", ""))
if not dataset_path or not dataset_path.exists():
    raise ValueError(
        "DATASET_PATH not found in .env file or path doesn't exist. "
        "Please run download_dataset.py first to download the dataset and create the .env file."
    )


class VOC2012SegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset

    Args:
        root_dir: Root directory of the dataset
        split: One of 'train', 'val', or 'trainval'
        use_augmentation: Whether to apply data augmentation (horizontal flip, color jitter)
    """

    # PASCAL VOC 2012 class names (21 classes including background)
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        use_augmentation: bool = False,
        crop_size: int = 520,
    ):
        self.root_dir = Path(root_dir) / \
            "versions/1/VOC2012_train_val/VOC2012_train_val"
        self.split = split
        self.use_augmentation = use_augmentation
        self.crop_size = crop_size

        # Paths to different components
        self.images_dir = self.root_dir / "JPEGImages"
        self.masks_dir = self.root_dir / "SegmentationClass"
        self.splits_dir = self.root_dir / "ImageSets" / "Segmentation"

        # Read the split file to get image IDs
        split_file = self.splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # Get normalization parameters from pretrained weights
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.FCN_ResNet50_Weights
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Color jitter for augmentation
        if use_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            )
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image ID
        image_id = self.image_ids[idx]

        # Load image and mask
        img_path = self.images_dir / f"{image_id}.jpg"
        mask_path = self.masks_dir / f"{image_id}.png"

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Step 1: Resize so shortest side is at least crop_size
        w, h = image.size
        if w < h:
            new_w = self.crop_size
            new_h = int(h * self.crop_size / w)
        else:
            new_h = self.crop_size
            new_w = int(w * self.crop_size / h)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Step 2: Crop to crop_size x crop_size
        if self.use_augmentation:
            # Random crop for training
            i = torch.randint(0, new_h - self.crop_size + 1, (1,)).item()
            j = torch.randint(0, new_w - self.crop_size + 1, (1,)).item()
        else:
            # Top-left crop for validation
            i = 0
            j = 0

        image = image.crop((j, i, j + self.crop_size, i + self.crop_size))
        mask = mask.crop((j, i, j + self.crop_size, i + self.crop_size))

        # Step 3: Apply augmentations if enabled
        if self.use_augmentation:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Color jitter (only to image)
            if self.color_jitter is not None:
                image = self.color_jitter(image)

        # Step 4: Convert to tensors
        image_tensor = F.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        # Step 5: Normalize image
        image_tensor = self.normalize(image_tensor)

        return image_tensor, mask_tensor


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    crop_size: int = 520,
    use_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        root_dir: Root directory of the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        crop_size: Size to crop images to (images resized so shortest side >= crop_size, then cropped)
        use_augmentation: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = VOC2012SegmentationDataset(
        root_dir=root_dir,
        split='train',
        use_augmentation=use_augmentation,
        crop_size=crop_size,
    )

    val_dataset = VOC2012SegmentationDataset(
        root_dir=root_dir,
        split='val',
        use_augmentation=False,  # Never augment validation set
        crop_size=crop_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Main execution
if __name__ == "__main__":
    print(f"Dataset path: {dataset_path}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir=str(dataset_path),
        batch_size=8,
        crop_size=520
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of classes: {len(VOC2012SegmentationDataset.CLASSES)}")

    # Test loading a batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Masks: {masks.shape}")
    print(f"Mask value range: [{masks.min()}, {masks.max()}]")
