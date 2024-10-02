import torch
import torch.nn as nn
from kan import KAN

class DeepEC_KAN(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(512)
        self.conv1 = nn.Conv1d(21, 128, 4)
        self.conv2 = nn.Conv1d(21, 128, 8)
        self.conv3 = nn.Conv1d(21, 128, 16)
        
        self.max1 = nn.MaxPool1d(997, return_indices=True)
        self.max2 = nn.MaxPool1d(993, return_indices=True)
        self.max3 = nn.MaxPool1d(985, return_indices=True)
        
        self.KAN = KAN([384, 512, 229], 3, k=3)
        
        self.LN = nn.LayerNorm(128*3)

        self.BN = nn.BatchNorm1d(128*3)
                
        self.dropout = nn.Dropout(0.1)
      
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x1, indices1 = self.max1(nn.functional.relu(self.conv1(x)))
        x2, indices2 = self.max2(nn.functional.relu(self.conv2(x)))
        x3, indices3 = self.max3(nn.functional.relu(self.conv3(x)))
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
            
        return x, indices1, indices2, indices3

def interpretation(model, final_score) : 
    
    model.eval()
    
    index_class = final_score.argmax()
    final_score = final_score[:,final_score.argmax()]

    previous_scores = final_score.expand(model.spline_postacts[-1][0,index_class,:].shape)

    importance_scores = torch.div(model.spline_postacts[-1][0,index_class,:].clip(min=0), previous_scores)

    for layer_index in range(-2, -len(model.spline_postacts)-1, -1) : 

        importance_first = torch.div(model.spline_postacts[layer_index][0], model.spline_postacts[layer_index][0].sum(1).expand_as(model.spline_postacts[layer_index][0].T).T).abs()
        importance_scores = importance_first * importance_scores[:,None]
        importance_scores = importance_scores.sum(0) 

    importance_scores /= importance_scores.max()
    
    return importance_scores.detach()

def conv_to_sequence(conv_index, conv_size) : 
    return conv_index, conv_index+conv_size

def map_inter_to_input(importance_scores, indices1, indices2, indices3) : 
    
    final_importance_scores = torch.zeros(1000)
            
    for index, score in enumerate(importance_scores) : 
        if index < 128 : 
            conv_size = 4
            conv_index = indices1.flatten()[index]
            start, end = conv_to_sequence(conv_index, conv_size)
            final_importance_scores[start:end] += score
        elif index < 256 : 
            conv_size = 8
            conv_index = indices2.flatten()[index-128]
            start, end = conv_to_sequence(conv_index, conv_size)
            final_importance_scores[start:end] += score
        else : 
            conv_size = 16
            conv_index = indices3.flatten()[index-256]
            start, end = conv_to_sequence(conv_index, conv_size)
            final_importance_scores[start:end] += score

    mean, std, var = torch.mean(final_importance_scores), torch.std(final_importance_scores), torch.var(final_importance_scores) 
    final_importance_scores  = (final_importance_scores-mean)/std   
    
    return final_importance_scores