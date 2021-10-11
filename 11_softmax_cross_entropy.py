"""
SOFTMAX

"""

import numpy as np
import torch
import torch.nn as nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 1.0])
outputs = softmax(x)
print(outputs)

x = torch.tensor([2.0, 1.0, 1.0])
outputs = torch.softmax(x, dim=0)
print(outputs)


"""
CROSS ENTROPHY LOSS

"""
import numpy as np
import torch
import torch.nn as nn


def cross_entropy(actual, predictions):
    loss = -np.sum(actual * np.log(predictions))
    return loss


Y = np.array([1, 0, 0])
Y_pred = np.array([0.7, 0.2, 0.1])
loss = cross_entropy(Y, Y_pred)
print(f"{loss:.4f}")

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])  # nsamples*n_classes = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 3.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f"{l1.item():.4f}")
print(f"{l2.item():.4f}")

chk1, predictions1 = torch.max(Y_pred_good, 1)
chk2, predictions2 = torch.max(Y_pred_bad, 1)

print(chk1, predictions1)
print(chk2, predictions2)


"""
MULTI CLASS
"""
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        # No softmax at the end
        return x


model = NeuralNet(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # it applies softmax by default


"""
BINARY CLASS
"""
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=1):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


model = NeuralNet(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion = nn.BCELoss()  # it applies softmax by default
