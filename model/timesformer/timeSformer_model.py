import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel


class TimeSformerModel(nn.Module):
    def __init__(self, num_classes=2, num_frames=8, img_size=224):
        super().__init__()

        # 최적화된 Config
        config = TimesformerConfig(
            num_frames=num_frames,
            num_labels=num_classes,
            image_size=img_size,
            patch_size=16,
            attention_type="divided_space_time",

            num_hidden_layers=4,          
            hidden_size=768,              # 그대로
            num_attention_heads=12,
            intermediate_size=3072,

            dropout=0.1,                  # 안정화
            attention_dropout=0.1
        )

        self.backbone = TimesformerModel(config)

        # 개선된 Head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        return self.cls_head(cls_token)