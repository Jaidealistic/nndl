import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class GROGUArchitecture(nn.Module):
    def __init__(self, num_classes=4, meta_features=13, freeze_visual=False):
        super(GROGUArchitecture, self).__init__()
        
        # 1. Visual Branch (MobileNetV2)
        # We use pretrained ImageNet weights for robust edge/texture detection
        weights = MobileNet_V2_Weights.DEFAULT
        self.visual_model = mobilenet_v2(weights=weights)
        
        if freeze_visual:
            for param in self.visual_model.parameters():
                param.requires_grad = False
                
        # Remove the final classification layer of MobileNetV2
        # MobileNetV2 outputs 1280 features before the classifier
        self.visual_feature_dim = self.visual_model.last_channel # 1280
        self.visual_model.classifier = nn.Identity() 
        
        # 2. Metadata Branch (Tabular MLP)
        self.metadata_model = nn.Sequential(
            nn.Linear(meta_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.meta_feature_dim = 32
        
        # 3. Late Fusion Classification Head
        # Concatenate 1280 (visual) + 32 (metadata) = 1312 dimensions
        self.fusion_dim = self.visual_feature_dim + self.meta_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, visual_x, metadata_x):
        # Extract visual features (N, 1280)
        v_feat = self.visual_model(visual_x)
        
        # Extract metadata features (N, 32)
        m_feat = self.metadata_model(metadata_x)
        
        # Late Fusion: Concatenate along feature dimension
        fused = torch.cat((v_feat, m_feat), dim=1) # (N, 1312)
        
        # Classification
        logits = self.classifier(fused) # (N, 4)
        return logits
