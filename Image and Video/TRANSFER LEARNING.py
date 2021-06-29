"""
In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning.

Two major transfer learning scenarios look as follows:

Finetuning the convnet: Instead of random initialization, we initialize the network with a pretrained network,
    like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the
    final fully connected layer. This last fully connected layer is replaced with a new one with random weights
    and only this layer is trained.
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

# 1. Load data
# Data Augmentation and normalization for training whereas Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(image_datasets)
print(dataloaders)
print(dataset_sizes)
print(class_names)
print(device)

# Visualize a few images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp * std + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print(len(inputs))
print(classes)

# make a grid from the batch
out = torchvision.utils.make_grid(inputs)
print(len(out))

imshow(out, title=[class_names[x] for x in classes])

# Training a model
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since=time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}{num_epochs-1}')
        print('-'*10)

        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase =='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inpus, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # zero the parameter gradients

                # forward and track history only if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} {epoch_loss:4f} {epoch_acc:4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print(f'Best val Acc: {best_acc}')

        # load best models
        model.load_state_dict(best_model_wts)
        return model


# Visualizing the model predictions
