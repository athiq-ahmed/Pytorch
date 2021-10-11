import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt("./data/wine.txt", skiprows=1, delimiter=",")
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
x, y = dataset[0]
print(x, y)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
x, y = dataset[0]
print(x, y)
