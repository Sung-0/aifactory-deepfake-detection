import torch
import torch.nn as nn
from torchvision import models


# ======================================
# scSE Block (채널 + 공간 주의)
# ======================================
class scSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_ch = max(channels // reduction, 8)

        # 채널 주의 (Channel Squeeze & Excitation)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 공간 주의 (Spatial Squeeze & Excitation)
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 학습 코드와 동일한 수식
        return x * (1 + 0.5 * (self.cSE(x) + self.sSE(x)))


# ======================================
# ScSEModel (ResNet18 백본 기반)
# ======================================
class ScSEModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # ImageNet 가중치 미사용 (대회 규정 고려)
        base = models.resnet18(weights=None)

        # ResNet 기본 레이어 + scSE 블록 삽입 (학습 코드와 동일)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = nn.Sequential(*base.layer1, scSEBlock(64))
        self.layer2 = nn.Sequential(*base.layer2, scSEBlock(128))
        self.layer3 = nn.Sequential(*base.layer3, scSEBlock(256))
        self.layer4 = nn.Sequential(*base.layer4, scSEBlock(512))

        # 분류기 헤드
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # 가중치 초기화 (학습 코드와 동일)
        self._init_weights()
        self._set_bn_momentum(0.01)

    # ---------------------------
    # 가중치 초기화
    # ---------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ---------------------------
    # BatchNorm 모멘텀 안정화
    # ---------------------------
    def _set_bn_momentum(self, momentum=0.05):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = momentum

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)