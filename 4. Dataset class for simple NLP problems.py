import torch

class CustomDataset:
    def __init__(self, data, targets, tokenizer):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item, :]
        target = self.targets[item]
        input_ids = self.tokenizer(text)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target),
        }