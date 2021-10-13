"""
-- No longer required


model.train
model.eval

model.device

no backward pass - optimizer.zero_grad(), loss.backward(),optimizer.step()

with torch.no_grad(): pass

x = x.detach()

"""

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# hyper parameters
input_size = 784  # (28*28)
hidden_size = 100
learning_rate = 0.001
batch_size = 100
num_epochs = 2
num_classes = 10

# MNIST
class MNISTModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTModule, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)

        # Logging to TensorBoard by default
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )

        return train_dataloader

    def validation_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )

        return val_dataloader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        tensorboard_log = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_log}


if __name__ == "__main__":
    trainer = Trainer(gpus=0, max_epochs=num_epochs, fast_dev_run=False)
    model = MNISTModule(input_size, hidden_size, num_classes)
    trainer.fit(model)
