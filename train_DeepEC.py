import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from efficient_kan.src.efficient_kan import KAN 
from DeepEC_KAN import DeepEC_KAN

class CustomDataset(Dataset):
    
    def __init__(self):
            
        self.df = pd.DataFrame(data = np.load('your_data.npy', allow_pickle=True), columns = ['Sequence', 'EC'])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['EC'].iloc[idx]).float()
    
dataset = CustomDataset() 

# train_index = your train indices
# val_index = your val indices

trainset = torch.utils.data.Subset(dataset, dataset.df[dataset.df.index.isin(train_index)].index)
valset = torch.utils.data.Subset(dataset, dataset.df[dataset.df.index.isin(val_index)].index)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)

model = DeepEC_KAN()

device = torch.device("cuda")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_lambda = lambda epoch : 0.95
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

min_val_loss = float('inf')

for epoch in range(80):

    # Train
    model.train()
    step_loss = []
    for sequences, labels in trainloader:
        sequences = sequences.to(device)
        optimizer.zero_grad()
        output = model(sequences)
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        step_loss.append(loss.item())

    train_loss = np.array(step_loss).mean()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequences, labels in valloader:
            sequences = sequences.to(device)
            output = model(sequences)
            val_loss += criterion(output, labels.to(device)).item()
    val_loss /= len(valloader)

    scheduler.step()
    
    if val_loss < min_val_loss : 
        min_val_loss = val_loss
        torch.save(model.state_dict(), f'models/deepec_kan_model.ckpt')
