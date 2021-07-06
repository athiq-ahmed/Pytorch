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

dataset = CustomDataset(data, targets)
len(dataset)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
print(train_loader)

for data in train_loader:
    print(data)
    break

for data in train_loader:
    print(data["x"].shape)
    print(data["y"].shape)
    break

for _ in range(10):
    for data in train_loader:
        x = data["x"]
        y = data["y"]
        outputs = model(x, y)