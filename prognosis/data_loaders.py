import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np
import cv2 as cv
import os

class MyDataset(Dataset):
    def __init__(self, data, labels, survivetimes, transform=None):
        super(MyDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.survivetimes = survivetimes
        self.transform = transform

    def __getitem__(self, index):
        feature_maps_path = self.data[index]
        feature_maps = np.load(feature_maps_path)
        if self.transform is not None:
            feature_maps = self.transform(image=feature_maps)["image"]
        T = torch.tensor(self.survivetimes[index]).type(torch.FloatTensor)
        O = torch.tensor(self.labels[index]).type(torch.FloatTensor)
        return feature_maps, T, O

    def __len__(self):
        return len(self.data)

class MyFusionDataset(Dataset):
    def __init__(self, data, labels, survivetimes, patchs_feats_file, channel, transform=None):
        super(MyFusionDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.survivetimes = survivetimes
        self.patchs_feats_file = patchs_feats_file
        self.transform = transform
        self.channel = channel

    def __getitem__(self, index):
        feature_maps_path = self.data[index]
        feature_maps = np.load(feature_maps_path).astype(np.float64)

        if self.transform is not None:
            feature_maps = self.transform(image=feature_maps)["image"]
        T = torch.tensor(self.survivetimes[index]).type(torch.FloatTensor)
        O = torch.tensor(self.labels[index]).type(torch.FloatTensor)

        patchs_feats_file = pd.read_csv(self.patchs_feats_file)
        pd_index = patchs_feats_file[patchs_feats_file['wsi_id'].isin([os.path.basename(feature_maps_path)])].index.values[0]
        patchs_feats = patchs_feats_file.iloc[pd_index, 0:self.channel]
        patchs_feats = patchs_feats.to_numpy().astype(np.float64)
        np.nan_to_num(patchs_feats,0)

        tensor_data = torch.tensor(patchs_feats)
        data_load = [tensor_data, feature_maps]
        return data_load, T, O

    def __len__(self):
        return len(self.data)
