import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
print(X[0][0])

logits = model(X)
print(logits)

pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)

y_pred = pred_probab.argmax(1)
print(y_pred)

# Model layers
input_image = torch.rand(3, 28,28)
print(input_image.shape)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.shape)

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.shape)

print(f'Before ReLu: {hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLu: {hidden1}')

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model parameters
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f'Layers: {name}| Size: {param.size()} | Values: {param[:2]} \n')

