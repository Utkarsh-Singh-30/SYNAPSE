import torch
import torch.nn as nn
from torchvision import models
import timm  # Library for state-of-the-art models (EfficientNet, SE-ResNeXt)

# ---------------- CUSTOM ARCHITECTURE: THORAXNET ----------------
class ThoraxNet(nn.Module):
    def __init__(self, num_classes):
        super(ThoraxNet, self).__init__()
        # Backbone: ResNet50 (Standard and strong)
        self.backbone = models.resnet50(weights='DEFAULT')
        num_ftrs = self.backbone.fc.in_features
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Attention Block (Focuses on relevant features)
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 16),
            nn.ReLU(),
            nn.Linear(num_ftrs // 16, num_ftrs),
            nn.Sigmoid()
        )
        
        # New Classifier
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        attn_weights = self.attention(features)
        weighted_features = features * attn_weights
        out = self.classifier(weighted_features)
        return out

# ---------------- FACTORY FUNCTION ----------------
def get_model(model_name, num_classes, device):
    print(f"[INFO] Initializing {model_name}...")
    
    if model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'densenet169':
        model = models.densenet169(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet101':
        model = models.resnet101(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'seresnext50_32x4d':
        # SE-ResNeXt-50 (Requires 'timm' library)
        model = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes)
        
    elif model_name == 'efficientnet_b1':
        # EfficientNet-B1 (Good balance of speed/accuracy)
        model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)
        
    elif model_name == 'thoraxnet':
        model = ThoraxNet(num_classes)
        
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model.to(device) 