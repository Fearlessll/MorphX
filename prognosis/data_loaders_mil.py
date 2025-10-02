import torch
from torch.utils.data import Dataset
import os
import random


class MyDatasetMIL(Dataset):
    """Multiple Instance Learning (MIL) dataset class for survival analysis.
    """

    def __init__(self, data, labels, survivetimes, transform=None):
        """Initialize the MIL dataset.

        Args:
            data (list): List of paths to feature tensors (.pt files)
            labels (list): List of event indicators (0=censored, 1=event)
            survivetimes (list): List of survival times
            transform (callable, optional): Optional transform to be applied
        """
        super(MyDatasetMIL, self).__init__()
        self.data = data
        self.labels = labels
        self.survivetimes = survivetimes
        self.transform = transform

    def __getitem__(self, index):

        # Load pre-computed features from disk
        instance_feats_path = self.data[index]
        instance_feats = torch.load(instance_feats_path)
        num_rows = instance_feats.size(0)

        # Set random seed for reproducibility
        torch.manual_seed(2024)

        # if num_rows < 1000:
        #     random_indices = torch.randint(0, num_rows, (1000,))
        # else:
        #     random_indices = torch.randperm(num_rows)[:1000]
        # instance_feats = instance_feats[random_indices]

        # Convert survival data to tensors
        T = torch.tensor(self.survivetimes[index]).type(torch.FloatTensor)
        O = torch.tensor(self.labels[index]).type(torch.FloatTensor)

        return instance_feats, T, O

    def __len__(self):
        return len(self.data)