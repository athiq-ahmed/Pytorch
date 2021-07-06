import torch
from sklearn.datasets import make_classification

class CustomDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {
            "x": torch.tensor(current_sample, dtype=torch.float),
            "y": torch.tensor(current_target, dtype=torch.long),
        }

data, targets = make_classification(n_samples=1000)
print(data[:1])
print(targets)
print(data.shape)
print(targets.shape)

custom_dataset = CustomDataset(data=data, targets=targets)
print(custom_dataset)
len(custom_dataset)
custom_dataset[0] # between 0 and 999 (n-1)
custom_dataset[0]["x"].shape
custom_dataset[0]["x"]
custom_dataset[0]["y"]

for idx in range(len(custom_dataset)):
    print(custom_dataset[idx])
    break

