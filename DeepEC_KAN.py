import torch
import torch.nn as nn
from kan import KAN

class DeepEC_KAN(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(512)
        self.conv1 = nn.Conv1d(21, 128, 4)
        self.conv2 = nn.Conv1d(21, 128, 8)
        self.conv3 = nn.Conv1d(21, 128, 16)
        
        self.max1 = nn.MaxPool1d(997)
        self.max2 = nn.MaxPool1d(993)
        self.max3 = nn.MaxPool1d(985)
        
        self.KAN = KAN([384, 512, 229], 3, k=3, device=device)
        if speed : 
            self.KAN.speed()
        
        self.LN = nn.LayerNorm(128*3)

        self.BN = nn.BatchNorm1d(128*3)
                
        self.dropout = nn.Dropout(0.1)
      
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x1 = self.max1(nn.functional.relu(self.conv1(x)))
        x2 = self.max2(nn.functional.relu(self.conv2(x)))
        x3 = self.max3(nn.functional.relu(self.conv3(x)))
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x3 = self.norm3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.BN(x)
            
        x = self.LN(x)
            
        x = self.dropout(x)
            
        x = self.KAN(x)
            
        return x