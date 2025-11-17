import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class LightFusionBlock(nn.Module):
    """Ultra-lightweight fusion block with depthwise separable convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise separable conv for efficiency
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChungusNet(nn.Module):
    """
    ChungusNet: Semantic Segmentation Model

    Architecture:
    - Backbone: MobileNetV3-Small (chopped to 14x14 bottleneck)
    - Decoder: Ultra-lightweight with only 2 taps and bilinear upsampling
    - Output: 520x520x21 semantic segmentation mask

    Args:
        num_classes: Number of segmentation classes (default: 21)
        pretrained: Whether to use pretrained MobileNetV3 weights (default: True)
    """

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Load MobileNetV3-Small backbone
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # Extract feature layers from MobileNetV3
        # MobileNetV3 features structure:
        # - features[0-3]: Early layers (stride 2, 4)
        # - features[4-6]: Mid layers (stride 8)
        # - features[7-12]: Later layers (stride 16)
        # - features[13-16]: Final layers (stride 32, but we stop earlier)

        self.features = backbone.features

        # Use only 2 taps for lighter architecture
        # Assuming input is 520x520:
        # - Tap 1 (early): 65x65 (stride 8) - features[:4]
        # - Tap 2 (bottleneck): ~14x14 (stride ~37) - features[:12]

        # Define channel dimensions
        # MobileNetV3-Small channel progression: 16, 16, 24, 48, 96
        self.early_channels = 24   # features[:4]
        self.bottleneck_channels = 96  # features[:12] (chopped backbone)

        # Encoder with only 2 stages
        self.encoder_early = nn.Sequential(*self.features[:4])  # -> 65x65x24
        self.encoder_bottleneck = nn.Sequential(*self.features[4:12]) # -> 17x17x96

        # Lightweight decoder with only 2 taps
        # Decoder path: 17x17 -> 65x65 -> 520x520

        # Reduce bottleneck channels (lighter than before)
        self.bottleneck_reduce = nn.Conv2d(self.bottleneck_channels, 64, kernel_size=1, bias=False)

        # Single fusion stage with early features
        self.lateral_early = nn.Conv2d(self.early_channels, 64, kernel_size=1, bias=False)
        self.decoder_fusion = LightFusionBlock(64, 32)

        # Minimal segmentation head
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 3, 520, 520)

        Returns:
            Output tensor of shape (B, num_classes, 520, 520)
        """
        # Store input size for final upsampling
        input_size = x.shape[2:]  # (520, 520)

        # Encoder: Extract features at 2 taps only
        early = self.encoder_early(x)  # (B, 24, 65, 65)
        bottleneck = self.encoder_bottleneck(early)  # (B, 96, 17, 17)

        # Reduce bottleneck channels
        x = self.bottleneck_reduce(bottleneck)  # (B, 64, 17, 17)

        # Upsample to early feature resolution
        x = F.interpolate(x, size=early.shape[2:], mode='bilinear', align_corners=False)  # (B, 64, 65, 65)

        # Fuse with early features (single fusion stage)
        lateral = self.lateral_early(early)  # (B, 64, 65, 65)
        x = x + lateral  # (B, 64, 65, 65)
        x = self.decoder_fusion(x)  # (B, 32, 65, 65)

        # Final upsampling to target resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)  # (B, 32, 520, 520)

        # Segmentation head
        x = self.seg_head(x)  # (B, num_classes, 520, 520)

        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = ChungusNet()
    batch_size = 16
    summary(model, input_size=(batch_size, 3, 520, 520))

