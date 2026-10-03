"""
Kiến trúc cho Macro-Expression Recognition trên ảnh khuôn mặt tĩnh.

Hỗ trợ 2 chế độ (xem config.USE_TRANSFER_LEARNING):
    - False: FacialExpressionCNN tự thiết kế, học from scratch.
    - True : backbone ResNet18/MobileNetV2 pretrain trên ImageNet (torchvision), thay
      lớp FC cuối bằng classifier NUM_CLASSES lớp — khuyến nghị khi dataset nhỏ.
"""

import torch
import torch.nn as nn

from config import BACKBONE_NAME, FREEZE_BACKBONE, NUM_CLASSES, USE_TRANSFER_LEARNING


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class FacialExpressionCNN(nn.Module):
    """4 khối Conv2D+BN+ReLU+MaxPool, sau đó global average pool + fully-connected classifier."""

    def __init__(self, num_classes=NUM_CLASSES, in_channels=3, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)
        return logits


def _build_transfer_learning_model(num_classes=NUM_CLASSES, backbone_name=BACKBONE_NAME, freeze_backbone=FREEZE_BACKBONE):
    import torchvision.models as tv_models

    if backbone_name == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif backbone_name == "mobilenet_v2":
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.last_channel, num_classes),
        )
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    return model


def build_model():
    if USE_TRANSFER_LEARNING:
        return _build_transfer_learning_model()
    return FacialExpressionCNN()


if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(2, 3, 112, 112)  # (B, C, H, W)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # kỳ vọng: (2, NUM_CLASSES)


