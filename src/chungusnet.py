import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class FPNFusionBlock(nn.Module):
    """Lightweight FPN fusion block for combining multi-scale features"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChungusNet(nn.Module):
    """
    ChungusNet: Semantic Segmentation Model

    Architecture:
    - Backbone: MobileNetV3-Large (chopped to 14x14 bottleneck)
    - Decoder: Lightweight FPN-like feature fusion with bilinear upsampling
    - Output: 520x520x21 semantic segmentation mask

    Args:
        num_classes: Number of segmentation classes (default: 21)
        pretrained: Whether to use pretrained MobileNetV3 weights (default: True)
    """

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Load MobileNetV3-Large backbone
        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)

        # Extract feature layers from MobileNetV3
        # MobileNetV3 features structure:
        # - features[0-3]: Early layers (stride 2, 4)
        # - features[4-6]: Mid layers (stride 8)
        # - features[7-12]: Later layers (stride 16)
        # - features[13-16]: Final layers (stride 32, but we stop earlier)

        self.features = backbone.features

        # Identify feature extraction points for FPN-like fusion
        # We'll extract features at different spatial resolutions
        # Assuming input is 520x520:
        # - Stage 1 (early): 130x130 (stride 4) - features[:4]
        # - Stage 2 (mid): 65x65 (stride 8) - features[:7]
        # - Stage 3 (late): 33x33 (stride 16) - features[:13]
        # - Bottleneck: ~14x14 (stride ~37) - features[:16]

        # Define channel dimensions at each stage
        # MobileNetV3-Large channel progression: 16, 24, 40, 80, 112, 160, 960
        self.stage1_channels = 24   # features[:4]
        self.stage2_channels = 40   # features[:7]
        self.stage3_channels = 112  # features[:13]
        self.bottleneck_channels = 960  # features[:16] (chopped backbone)

        # Chop the backbone to get ~14x14 bottleneck
        # Using features[:16] gives us the desired bottleneck before the final layers
        self.encoder_stage1 = nn.Sequential(*self.features[:4])   # -> 130x130x24
        self.encoder_stage2 = nn.Sequential(*self.features[4:7])  # -> 65x65x40
        self.encoder_stage3 = nn.Sequential(*self.features[7:13]) # -> 33x33x112
        self.encoder_bottleneck = nn.Sequential(*self.features[13:16]) # -> ~14x14x960

        # FPN-like decoder with feature fusion
        # Decoder path: 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224 -> 520x520

        # Reduce bottleneck channels
        self.bottleneck_reduce = nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder stage 3: Fuse bottleneck with stage3 features
        self.lateral3 = nn.Conv2d(self.stage3_channels, 256, kernel_size=1)
        self.decoder3 = FPNFusionBlock(256, 128)

        # Decoder stage 2: Fuse with stage2 features
        self.lateral2 = nn.Conv2d(self.stage2_channels, 128, kernel_size=1)
        self.decoder2 = FPNFusionBlock(128, 64)

        # Decoder stage 1: Fuse with stage1 features
        self.lateral1 = nn.Conv2d(self.stage1_channels, 64, kernel_size=1)
        self.decoder1 = FPNFusionBlock(64, 64)

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

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

        # Encoder: Extract multi-scale features
        s1 = self.encoder_stage1(x)        # (B, 24, 130, 130)
        s2 = self.encoder_stage2(s1)       # (B, 40, 65, 65)
        s3 = self.encoder_stage3(s2)       # (B, 112, 33, 33)
        bottleneck = self.encoder_bottleneck(s3)  # (B, 960, ~14, ~14)

        # Reduce bottleneck channels
        x = self.bottleneck_reduce(bottleneck)  # (B, 256, ~14, ~14)

        # Decoder: FPN-like fusion with bilinear upsampling

        # Decode stage 3: Upsample and fuse with s3
        x = F.interpolate(x, size=s3.shape[2:], mode='bilinear', align_corners=False)
        lateral3 = self.lateral3(s3)
        x = x + lateral3
        x = self.decoder3(x)  # (B, 128, 33, 33)

        # Decode stage 2: Upsample and fuse with s2
        x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        lateral2 = self.lateral2(s2)
        x = x + lateral2
        x = self.decoder2(x)  # (B, 64, 65, 65)

        # Decode stage 1: Upsample and fuse with s1
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        lateral1 = self.lateral1(s1)
        x = x + lateral1
        x = self.decoder1(x)  # (B, 64, 130, 130)

        # Final upsampling to target resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Segmentation head
        x = self.seg_head(x)  # (B, num_classes, 520, 520)

        return x

