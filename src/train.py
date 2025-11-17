import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import os
from torchmetrics import JaccardIndex

from chungusnet import ChungusNet
from dataset import create_dataloaders, dataset_path
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Response-Based Distillation

    Args:
        temperature: Temperature for softening probability distributions
        alpha: Weight for teacher loss (1-alpha will be weight for GT loss)
    """
    def __init__(self, temperature=6.0, alpha=0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: (B, C, H, W) logits from student
            teacher_logits: (B, C, H, W) logits from teacher
            targets: (B, H, W) ground truth labels

        Returns:
            Combined loss
        """
        # Ground truth loss (standard cross-entropy)
        gt_loss = self.ce_loss(student_logits, targets)

        # Distillation loss (KL divergence between softened distributions)
        # Soften the predictions using temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence loss
        distill_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by T^2 as per Hinton et al.

        # Weighted combination
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * gt_loss

        return total_loss, distill_loss.item(), gt_loss.item()


def load_teacher_model(device):
    """
    Load pretrained FCN-ResNet50 as teacher model

    Args:
        device: Device to load model on

    Returns:
        Teacher model in eval mode
    """
    print("Loading teacher model (FCN-ResNet50)...")
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    teacher = fcn_resnet50(weights=weights)
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    print("Teacher model loaded successfully")
    return teacher




def train_epoch_solo(model, train_loader, criterion, optimizer, metric, device, epoch):
    """
    Train for one epoch in solo mode (standard training)

    Args:
        model: Student model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        metric: JaccardIndex metric
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    metric.reset()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Solo]")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metric.update(preds, targets)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'mIoU': f'{metric.compute().item():.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_miou = metric.compute().item()
    return avg_loss, avg_miou


def train_epoch_distill(model, teacher, train_loader, criterion, optimizer, metric, device, epoch):
    """
    Train for one epoch in student-teacher mode (knowledge distillation)

    Args:
        model: Student model
        teacher: Teacher model
        train_loader: Training data loader
        criterion: Distillation loss function
        optimizer: Optimizer
        metric: JaccardIndex metric
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (avg_total_loss, avg_distill_loss, avg_gt_loss, avg_miou)
    """
    model.train()
    metric.reset()
    total_loss = 0.0
    total_distill_loss = 0.0
    total_gt_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Distill]")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass - student
        student_logits = model(images)

        # Forward pass - teacher (no grad)
        with torch.no_grad():
            teacher_output = teacher(images)
            # FCN returns a dict with 'out' key
            teacher_logits = teacher_output['out']

        # Compute distillation loss
        loss, distill_loss, gt_loss = criterion(student_logits, teacher_logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_distill_loss += distill_loss
        total_gt_loss += gt_loss
        preds = student_logits.argmax(dim=1)
        metric.update(preds, targets)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'distill': f'{distill_loss:.4f}',
            'gt': f'{gt_loss:.4f}',
            'mIoU': f'{metric.compute().item():.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_distill = total_distill_loss / len(train_loader)
    avg_gt = total_gt_loss / len(train_loader)
    avg_miou = metric.compute().item()

    return avg_loss, avg_distill, avg_gt, avg_miou


def validate(model, val_loader, metric, device):
    """
    Validate the model

    Args:
        model: Model to validate
        val_loader: Validation data loader
        metric: JaccardIndex metric
        device: Device to validate on

    Returns:
        Tuple of (avg_loss, avg_miou)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metric.reset()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            metric.update(preds, targets)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{metric.compute().item():.4f}'
            })

    avg_loss = total_loss / len(val_loader)
    avg_miou = metric.compute().item()

    return avg_loss, avg_miou


def main():
    parser = argparse.ArgumentParser(description='Train ChungusNet for semantic segmentation')
    parser.add_argument('--mode', type=str, required=True, choices=['solo', 'student-teacher'],
                        help='Training mode: solo or student-teacher')
    parser.add_argument('--weights_file', type=str, required=True,
                        help='Path to save model weights')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Creating ChungusNet...")
    model = ChungusNet(num_classes=21, pretrained=True)
    model = model.to(device)

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        root_dir=str(dataset_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Setup training based on mode
    if args.mode == 'solo':
        print("\n=== SOLO TRAINING MODE ===")
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        teacher = None
    else:  # student-teacher
        print("\n=== STUDENT-TEACHER TRAINING MODE ===")
        print("Temperature: 6.0")
        print("Loss weights: 0.4 teacher, 0.6 ground truth")
        criterion = DistillationLoss(temperature=6.0, alpha=0.4)
        teacher = load_teacher_model(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Setup metrics
    train_metric = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255).to(device)
    val_metric = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255).to(device)

    print(f"\nOptimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}\n")

    # Training loop
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")

        # Train
        if args.mode == 'solo':
            train_loss, train_miou = train_epoch_solo(
                model, train_loader, criterion, optimizer, train_metric, device, epoch
            )
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        else:
            train_loss, distill_loss, gt_loss, train_miou = train_epoch_distill(
                model, teacher, train_loader, criterion, optimizer, train_metric, device, epoch
            )
            print(f"Train Loss: {train_loss:.4f} (Distill: {distill_loss:.4f}, GT: {gt_loss:.4f}) | Train mIoU: {train_miou:.4f}")

        # Validate
        val_loss, val_miou = validate(model, val_loader, val_metric, device)
        print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss,
                'mode': args.mode
            }, args.weights_file)
            print(f"✓ Saved best model (mIoU: {best_miou:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = args.weights_file.replace('.pth', f'_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss,
                'mode': args.mode
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Model saved to: {args.weights_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
