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

dataset_path = Path(os.path.expanduser(
    "~/.cache/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset"))


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
    img_transform: transforms.Compose
    mask_transform: transforms.Compose

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        use_augmentation: bool = False,
    ):
        self.root_dir = Path(root_dir) / \
            "versions/1/VOC2012_train_val/VOC2012_train_val"
        self.split = split
        self.use_augmentation = use_augmentation

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

        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.FCN_ResNet50_Weights
        self.img_transform = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
        img_size = 520

        # Base mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.squeeze(0).long())
        ])

        # Data augmentation transforms (applied before img_transform)
        if use_augmentation:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
            ])
        else:
            self.augmentation = None

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

        # Apply transforms to convert to tensors
        image_tensor = cast(torch.Tensor, self.img_transform(image))
        mask_tensor = cast(torch.Tensor, self.mask_transform(mask))

        # Apply data augmentation if enabled (on tensors)
        if self.use_augmentation:
            # Apply horizontal flip to both image and mask with same probability
            if np.random.rand() < 0.5:
                image_tensor = F.hflip(image_tensor)
                mask_tensor = F.hflip(mask_tensor)

            # Apply color jitter only to image (not mask)
            color_jitter = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            )
            image_tensor = color_jitter(image_tensor)

        return image_tensor, mask_tensor


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    img_size: int = 256,
    use_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        root_dir: Root directory of the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        img_size: Size to resize images to
        use_augmentation: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = VOC2012SegmentationDataset(
        root_dir=root_dir,
        split='train',
        use_augmentation=use_augmentation,
    )

    val_dataset = VOC2012SegmentationDataset(
        root_dir=root_dir,
        split='val',
        use_augmentation=False,  # Never augment validation set
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
        img_size=256
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
