import torch
import torch.nn as nn
from efficient_kan.src.efficient_kan import KAN 

class CLEAN_KAN(nn.Module):
    def __init__(self):
        super(LayerNormNet, self).__init__()

        self.fc1 = KAN([1280, 512], grid_size=3, spline_order=3)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = KAN([512, 229], grid_size=3, spline_order=3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x
