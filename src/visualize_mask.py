from pathlib import Path
import typer
from typing import Literal, Optional
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F


def visualize_mask(image: torch.Tensor, mask: torch.Tensor, filename: str, num_classes: int = 21) -> None:
    """
    Visualize segmentation mask overlaid on image using PyTorch utilities

    Args:
        image: RGB image tensor (C, H, W), normalized with ImageNet mean/std
        mask: Segmentation mask tensor (H, W) with class indices
        filename: Name of the output image file (with extension)
        num_classes: Number of classes (default 21 for PASCAL VOC)
    """
    # Create output directory if it doesn't exist
    output_dir = Path("mask-images")
    output_dir.mkdir(exist_ok=True)

    # Denormalize the image (reverse ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean

    # Clamp to [0, 1] range and convert to uint8
    image_denorm = torch.clamp(image_denorm, 0, 1)
    image_uint8 = (image_denorm * 255).to(torch.uint8)

    # Create boolean masks for each class
    # Shape: (num_classes, H, W)
    all_classes_masks = mask[None] == torch.arange(num_classes)[:, None, None]

    # Draw segmentation masks on the image
    image_with_masks = draw_segmentation_masks(image_uint8, masks=all_classes_masks, alpha=0.6)

    # Convert to PIL Image and save
    output_image = F.to_pil_image(image_with_masks)
    output_path = output_dir / filename
    output_image.save(output_path)

def main(index: int, model_id: Optional[Literal["resnet50-fcn"]] = None):
    from dataset import VOC2012SegmentationDataset, dataset_path
    from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
    from torchmetrics import JaccardIndex
    val_dataset = VOC2012SegmentationDataset(
        root_dir=str(dataset_path),
        split='val',
        use_augmentation=False
    )
    image, gt_mask = val_dataset[index]
    if model_id == "resnet50-fcn":
        # Load pretrained FCN ResNet50 model
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        model.eval()

        # Run inference
        with torch.no_grad():
            # Add batch dimension and run model
            input_batch = image.unsqueeze(0)
            output = model(input_batch)
            # Extract segmentation mask from output dict and remove batch dimension
            pred_mask = output['out'].squeeze(0).argmax(0)

        # Calculate mIoU using torchmetrics
        # JaccardIndex with ignore_index=255 to handle boundary/ignore pixels
        jaccard = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255)
        # Add batch dimension for torchmetrics
        miou = jaccard(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
        print(f"Mean IoU: {miou}")

        # Visualize predicted mask
        visualize_mask(image, pred_mask, f"{model_id}-mask-{index}.png")
        print(f"Saved predicted mask to mask-images/pred-mask-{index}.png")
    else:
        visualize_mask(image, gt_mask, f"gt-mask-{index}.png")
        print(f"Saved ground truth mask to mask-images/gt-mask-{index}.png")

if __name__ == "__main__":
    typer.run(main)