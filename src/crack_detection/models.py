"""Two-class CNN and MobileNetV2 feature extractor."""
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(start_dim=1))


def make_model(name: str, pretrained=True):
    if name == "cnn":
        return SmallCNN()
    if name == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        return model
    raise ValueError(f"Unknown model: {name}")
