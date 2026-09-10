from torch import nn
from torchvision import models
from torchvision import transforms
from typing import Tuple


def create_model(num_classes: int = 101) -> Tuple[nn.Module, transforms.Compose, transforms.Compose]:
    weights = models.EfficientNet_B2_Weights.DEFAULT
    
    model = models.efficientnet_b2(weights=weights)
    
    model = model.to("cpu")
    
    test_transforms = weights.transforms()
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=288, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=test_transforms.mean,
                            std=test_transforms.std)
    ])
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    for param in model.features[-2:].parameters():
        param.requires_grad = True
        
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, 
                out_features=num_classes,
                bias=True)
    )    
    
    return model, train_transforms, test_transforms