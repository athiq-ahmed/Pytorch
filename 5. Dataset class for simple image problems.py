import torch
import cv2
import numpy as np

class CustomDataset:
    def __init__(self, image_paths, targets, augmentations):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        target = self.targets[item]
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.augmentations(image=image)
        image = augmented["image"]
        # tensor.unsqueeze(0)     # adding channel if the image is gray size
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image":torch.tensor(image),
            "target":torch.tensor(target)
        }

    