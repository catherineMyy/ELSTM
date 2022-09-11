import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, period_input, target,):
        super(SyntheticDataset, self).__init__()  
        self.X_input = X_input
        self.period_input = period_input
        self.target = target
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx,:,:], self.period_input[idx,:,:], self.target[idx,:,:])