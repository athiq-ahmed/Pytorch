import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# hyper parameters
input_size = 784  # (28*28)
hidden_size = 100
learning_rate = 0.001
batch_size = 100
num_epochs = 2
num_classes = 10

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=False
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

samples, labels = iter(train_dataloader).next()
print(samples.shape, labels.shape)
print(samples[0])
print(labels[0])

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
plt.show()


class MNISTModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTModule, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    # def __init__(self, input_size, hidden_size, num_classes):
    #     super(MNISTModule, self).__init__()
    #     self.input_size = input_size
    #     self.l1 = nn.Linear(input_size, hidden_size)
    #     self.relu = nn.ReLU()
    #     self.l2 = nn.Linear(hidden_size, num_classes)

    # def forward(self, x):
    #     out = self.l1(x)
    #     out = self.relu(out)
    #     out = self.l2(out)
    #     # no activation and no softmax at the end
    #     return out


model = MNISTModule(input_size, hidden_size, num_classes).to(DEVICE)
print(model)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_dataloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )


# for epoch in range(num_epochs):
#     model.train()
#     for i, (images, labels) in enumerate(train_dataloader):
#         images = images.reshape(-1, 28 * 28).to(DEVICE)
#         lables = labels.to(DEVICE)

#         # forward - predictions and loss
#         predictions = model(images)
#         loss = criterion(predictions, labels)

#         # Backward - gradients and update weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(
#                 f"epoch: [{epoch+1} / {num_epochs}], step: [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
#             )

# validation loop
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_dataloader:
#         images = images.reshape(-1, 28 * 28).to(DEVICE)
#         lables = labels.to(DEVICE)

#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)  # max returns value, index
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f"accuracy: {acc} %")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, 28 * 28).to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network on the 10000 test images: {acc} %")
