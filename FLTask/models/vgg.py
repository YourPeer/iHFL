import math

import torch
import torch.nn as nn


class vgg11(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),

        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 128,bias=False),
            nn.ReLU(),
            nn.Linear(128, 128,bias=False),
            nn.ReLU(),
            nn.Linear(128, 10,bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# model=vgg11()
# for name,module in model.named_modules():
#     if module.__class__.__name__=="BatchNorm2d":
#         print(module.bias)
# for name, module in model.named_modules():
#     if hasattr(module, 'weight'):
#         print(module.weight)