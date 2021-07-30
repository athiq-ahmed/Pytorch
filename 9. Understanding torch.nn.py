import torch.nn as nn
import torch

""""
Module
"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, features):
        x = self.layer1(features); print(x.shape)
        x = self.layer2(x); print(x.shape)
        x = self.layer3(x); print(x.shape)
        return x

model = Model()
features = torch.rand((2, 128))
print(features)
model(features)
features.device

features = features.to("cuda")
model = Model().to("cuda")
model(features)


"""
Sequential
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(128, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1)
        )

    def forward(self, features):
        x = self.base(features); print(x.shape)
        return x

model = Model()
features = torch.rand((2, 128))
print(features)
model(features)
features.device

