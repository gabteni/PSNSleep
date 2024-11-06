import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
class trainEEGDataset(Dataset):
    def __init__(self, data,aug,train_index):
        self.data = data[train_index]
        self.aug =aug
        self.len=len(self.data)
        self.train_index=train_index
        data_mean = torch.mean(self.data, axis=(0, 2))
        data_std = torch.std(self.data, axis=(0, 2))
        print(data_mean)
        print(data_std)
        self.transform = transforms.Normalize(mean=data_mean, std=data_std)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_sample = self.data[idx]
        eeg_sample = self.transform(eeg_sample.reshape(1, -1, 1)).reshape(eeg_sample.shape)
        eeg_sample=np.array(eeg_sample).astype(float)

        #eeg_sample=self.aug(eeg_sample,100)
        eeg_sample1=self.aug(eeg_sample,100)
        if np.random.rand()<0.6:
            #if idx+1 in self.train_index:
            adj_ep=self.data[(idx+1)%self.len]
            adj_ep= self.transform(adj_ep.reshape(1, -1, 1)).reshape(adj_ep.shape)
            adj_ep=np.array(adj_ep).astype(float)
            eeg_sample1=np.add((eeg_sample1*0.7),(0.3*adj_ep))
        return (eeg_sample,eeg_sample1),0

class infEEGDataset(Dataset):
    def __init__(self, data,aug,train_index,lbls,norm_index):
        self.data = data[train_index]
        self.lbls=lbls[train_index]
        self.aug =aug
        self.len=len(self.data)
        self.augmentation = "noise_timeShift"#"mask_noise"#"permute_timeShift_scale_noise"
        
        data_mean = torch.mean(data[norm_index], axis=(0, 2))
        data_std = torch.std(data[norm_index], axis=(0, 2))
        print(data_mean)
        print(data_std)
        self.transform = transforms.Normalize(mean=data_mean, std=data_std)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_sample = self.data[idx]
        eeg_sample = self.transform(eeg_sample.reshape(1, -1, 1)).reshape(eeg_sample.shape)
        eeg_sample=np.array(eeg_sample).astype(float)
        return eeg_sample,self.lbls[idx]