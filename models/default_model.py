import torch.nn as nn
import torchvision.models as models
import torch
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # ResNet18 모델 불러오기
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # feature extractor로만 사용
        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class BasicCBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1))
        x = x * sa
        return x


class ConvNeXtWithCBAM(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=26):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        channels = [f['num_chs'] for f in self.backbone.feature_info]
        
        # 각 stage별 CBAM 삽입
        self.cbams = nn.ModuleList([
            BasicCBAM(ch) for ch in channels
        ])
        
        # feature projection 후 concat하여 최종 classifier로
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, 256)  # 각 feature를 256-d로 통일
            )
            for ch in channels
        ])
        
        self.classifier = nn.Linear(256 * len(channels), num_classes)

    def forward(self, x):
        feats = self.backbone(x)  # list of 4 features: [C3, C4, C5, C6]
        out_feats = []

        for i in range(len(feats)):
            feat = self.cbams[i](feats[i])       # attention 삽입
            feat = self.proj_heads[i](feat)      # projection
            out_feats.append(feat)

        x = torch.cat(out_feats, dim=1)
        return self.classifier(x)


class ConvNextQuarter(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=26, load_path: str = '/workspace/HVAI/output_convnext/250531_convnextbase_0.2501763867.pth'):
        super().__init__()
        self.backbone_whole = timm.create_model(model_name, num_classes=396)
        self.backbone_quarter1 = timm.create_model(model_name, num_classes=396)
        self.backbone_quarter2 = timm.create_model(model_name, num_classes=396)
        self.backbone_quarter3 = timm.create_model(model_name, num_classes=396)
        self.backbone_quarter4 = timm.create_model(model_name, num_classes=396)

        ckpt = torch.load(load_path)
        self.backbone_whole.load_state_dict(ckpt)
        self.backbone_quarter1.load_state_dict(ckpt)
        self.backbone_quarter2.load_state_dict(ckpt)
        self.backbone_quarter3.load_state_dict(ckpt)
        self.backbone_quarter4.load_state_dict(ckpt)

        self.backbone_whole.head.fc = nn.Linear(self.backbone_whole.head.in_features, num_classes, bias=True)
        self.backbone_quarter1.head.fc = nn.Linear(self.backbone_quarter1.head.in_features, num_classes, bias=True)
        self.backbone_quarter2.head.fc = nn.Linear(self.backbone_quarter2.head.in_features, num_classes, bias=True)
        self.backbone_quarter3.head.fc = nn.Linear(self.backbone_quarter3.head.in_features, num_classes, bias=True)
        self.backbone_quarter4.head.fc = nn.Linear(self.backbone_quarter4.head.in_features, num_classes, bias=True)

        # 학습 가능한 가중치 (초기값은 기존 비율과 유사하게 설정)
        self.raw_weights = nn.Parameter(torch.tensor([-0.6931, -2.0794, -2.0794, -2.0794, -2.0794], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature 추출
        feats_whole = self.backbone_whole(x)
        feats_quarter1 = self.backbone_quarter1(x[:, :, :x.shape[2] // 2, :x.shape[3] // 2])
        feats_quarter2 = self.backbone_quarter2(x[:, :, :x.shape[2] // 2, x.shape[3] // 2:])
        feats_quarter3 = self.backbone_quarter3(x[:, :, x.shape[2] // 2:, :x.shape[3] // 2])
        feats_quarter4 = self.backbone_quarter4(x[:, :, x.shape[2] // 2:, x.shape[3] // 2:])

        # softmax로 정규화된 가중치
        weights = torch.nn.functional.softmax(self.raw_weights, dim=0)

        # 가중합
        output = (
            feats_whole     * weights[0] +
            feats_quarter1  * weights[1] +
            feats_quarter2  * weights[2] +
            feats_quarter3  * weights[3] +
            feats_quarter4  * weights[4]
        )

        return output
        