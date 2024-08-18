import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet50

def get_vgg16_model(pretrained=True, num_classes=1):
    """
    Returns a VGG16 model modified for binary classification.
    
    Parameters:
    - pretrained (bool): Whether to use a pre-trained VGG16 model.
    - num_classes (int): Number of output classes. For binary classification, this should be 1.
    
    Returns:
    - model (nn.Module): Modified VGG16 model.
    """
    model = vgg16(pretrained=pretrained)
    
    # Modify the final layer to match the binary classification task
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model

def get_resnet50_model(pretrained=True, num_classes=1):
    """
    Returns a ResNet-50 model modified for binary classification.
    
    Parameters:
    - pretrained (bool): Whether to use a pre-trained ResNet-50 model.
    - num_classes (int): Number of output classes. For binary classification, this should be 1.
    
    Returns:
    - model (nn.Module): Modified ResNet-50 model.
    """
    model = resnet50(pretrained=pretrained)
    
    # Modify the final fully connected layer to match the binary classification task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


from torchvision.models import alexnet

def get_alexnet_model(pretrained=True, num_classes=1):
    model = alexnet(pretrained=pretrained)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
