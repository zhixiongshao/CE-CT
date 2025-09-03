import torch.nn as nn
import torchvision.models as models

class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18FineTune, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze only layer4 and the fully connected (fc) layer
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer to match num_classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet.fc.requires_grad = True  # Only train layer4 and fc layers

    def forward(self, x):
        return self.resnet(x)

