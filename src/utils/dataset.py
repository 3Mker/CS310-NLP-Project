import os
import json
import torch


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(self.inputs, dict):
            item = {key: val[idx] for key, val in self.inputs.items()}
        elif isinstance(self.inputs, list):
            item = self.inputs[idx]  # Directly use the dictionary from the list
        else:
            raise TypeError("Unsupported type for inputs")
        item['labels'] = self.labels[idx]
        return item

