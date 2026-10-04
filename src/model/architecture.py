import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class GROGUArchitecture(nn.Module):
    def __init__(self, num_classes=4, meta_features=13, freeze_visual=False):
        super(GROGUArchitecture, self).__init__()
        
        # 1. Visual Branch (MobileNetV2)
        weights = MobileNet_V2_Weights.DEFAULT
        self.visual_model = mobilenet_v2(weights=weights)
        
        self.visual_feature_dim = self.visual_model.last_channel  # 1280
        self.visual_model.classifier = nn.Identity()
        
        if freeze_visual:
            self.set_visual_trainable(unfreeze=False)
        
        # 2. Metadata Branch (Tabular MLP)
        self.metadata_model = nn.Sequential(
            nn.BatchNorm1d(meta_features),
            nn.Linear(meta_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.meta_feature_dim = 32
        
        # 3. Late Fusion Classification Head
        self.fusion_dim = self.visual_feature_dim + self.meta_feature_dim  # 1312
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def set_visual_trainable(self, unfreeze=True, from_block=14):
        """Allows progressive fine-tuning of the MobileNetV2 feature blocks."""
        for i, block in enumerate(self.visual_model.features):
            requires = unfreeze and (i >= from_block)
            for param in block.parameters():
                param.requires_grad = requires
        
    def forward(self, visual_x, metadata_x):
        # Extract visual features (N, 1280)
        v_feat = self.visual_model(visual_x)
        
        # Extract metadata features (N, 32)
        m_feat = self.metadata_model(metadata_x)
        
        # Late Fusion: Concatenate along feature dimension
        fused = torch.cat((v_feat, m_feat), dim=1)  # (N, 1312)
        
        # Classification
        logits = self.classifier(fused)  # (N, 4)
        return logits

