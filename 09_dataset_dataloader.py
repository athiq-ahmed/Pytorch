import math

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("./data/wine.txt", delimiter=",", skiprows=1, dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
features, labels = dataset[0]
print(features, labels)

batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, outputs) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(
                f"epoch {epoch+1}/ {num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}"
            )
