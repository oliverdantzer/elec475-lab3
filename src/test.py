import typer
from typing import Literal, Optional
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from dataset import VOC2012SegmentationDataset, dataset_path


def main(model_id: Literal["resnet50-fcn", "chungusnet"], batch_size: int, params_file: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = None
    if model_id == "resnet50-fcn":
        from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    elif model_id == "chungusnet":
        if params_file is None:
            raise ValueError("weights file must be defined for chungusnet")
        else:
            from chungusnet import ChungusNet
            model = ChungusNet()
            model.load_state_dict(torch.load(params_file, map_location=device))

    model = model.to(device)
    model.eval()

    val_dataset = VOC2012SegmentationDataset(
        root_dir=str(dataset_path),
        split='val',
        use_augmentation=False
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize JaccardIndex metric (same as in visualize_mask.py)
    jaccard = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255)

    miou_scores = []
    inference_times = []

    print(f"Testing {model_id} on validation set...")
    print(f"Device: {device}")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Measure inference time
            start_time = time.time()

            if model_id == "resnet50-fcn":
                outputs = model(images)['out']
            else:  # chungusnet
                outputs = model(images)

            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time / images.size(0))  # Per-image time

            # Calculate mIoU for each image in batch
            pred_masks = outputs.argmax(dim=1)  # Convert from logits to class indices
            for i in range(images.size(0)):
                miou = jaccard(pred_masks[i].cpu(), targets[i].cpu()).item()
                miou_scores.append((batch_idx * batch_size + i, miou))

    # Sort by mIoU score
    miou_scores.sort(key=lambda x: x[1])

    # Get worst and best 4
    worst_4 = miou_scores[:4]
    best_4 = miou_scores[-4:]

    print("\nWorst 4 images by mIoU:")
    for idx, score in worst_4:
        print(f"  Index: {idx}, mIoU: {score:.4f}")

    print("\nBest 4 images by mIoU:")
    for idx, score in best_4:
        print(f"  Index: {idx}, mIoU: {score:.4f}")

    mean_miou = np.mean([score for _, score in miou_scores])
    mean_inference_time = np.mean(inference_times)

    print(f"\nMean mIoU: {mean_miou:.4f}")
    print(f"Mean inference speed: {mean_inference_time:.2f} ms per image")
    

if __name__ == "__main__":
    typer.run(main)