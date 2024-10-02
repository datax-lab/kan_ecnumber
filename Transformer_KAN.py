import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from efficient_kan.src.efficient_kan import KAN  

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TransformerEmbedding(nn.Module): 

    def __init__(self, vocab_size, embed_size=256, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=1001)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, KANMHA, GridSize, SplineOrder, dropout):
        super().__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        
        if KANMHA == 0 :
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
        else : 
            self.query = KAN([d_model, d_model], grid_size=GridSize, spline_order=SplineOrder)
            self.key = KAN([d_model, d_model], grid_size=GridSize, spline_order=SplineOrder)
            self.value = KAN([d_model, d_model], grid_size=GridSize, spline_order=SplineOrder)
            
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, mask=None):
        
        x = self.norm(x)
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / math.sqrt(query.size(-1))

        p_attn = self.softmax(scores)

        x = torch.matmul(p_attn, value)
        
        batch_size = query.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        x = self.dropout(x)
        
        return x
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, dimension, KANLayers, KANsize1, KANsize2, KANsize3, GridSize, SplineOrder):
        super(PositionwiseFeedForward, self).__init__()
        KANsizes = [KANsize1, KANsize2, KANsize3]
        if KANLayers == 0 :
            self.KAN1 = KAN([dimension, dimension], grid_size=GridSize, spline_order=SplineOrder)
        else : 
            self.KAN1 = KAN([dimension, KANsize1], grid_size=GridSize, spline_order=SplineOrder)
            
        self.KANS = []
        for index in range(KANLayers) : 
            if index + 1 == KANLayers : 
                self.KANS.append(KAN([KANsizes[index], dimension], grid_size=GridSize, spline_order=SplineOrder))
            else : 
                self.KANS.append(KAN([KANsizes[index], KANsizes[index+1]], grid_size=GridSize, spline_order=SplineOrder))
        self.KANS = nn.ModuleList(self.KANS)

    def forward(self, x):
        x = self.KAN1(x)
        for layer in self.KANS : 
            x = layer(x)
        return x
    
class SublayerConnection(nn.Module):
    
    def __init__(self, size, KANLayers, KANsize1, KANsize2, KANsize3, GridSize, SplineOrder, dropout):
        super(SublayerConnection, self).__init__()
        self.feed_forward = PositionwiseFeedForward(size, KANLayers, KANsize1, KANsize2, KANsize3, GridSize, SplineOrder)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return x1 + self.dropout(self.feed_forward(self.norm(x2)))
    
class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, KANLayers, KANsize1, KANsize2, KANsize3, KANMHA, GridSize, SplineOrder, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, KANMHA, GridSize, SplineOrder, dropout)
        self.output_sublayer = SublayerConnection(hidden, KANLayers, KANsize1, KANsize2, KANsize3, GridSize, SplineOrder, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x1 = self.attention(x, mask)
        x = x + x1
        x = self.output_sublayer(x, x)
        return self.dropout(x)
    
class Transformer(nn.Module) :
    
    def __init__(self, config) :
        super().__init__()
        dimension = config['dimension']
        attn_heads = 4
        vocab_size = 23
        dropout = config['Dropout']
        self.embeddings = TransformerEmbedding(vocab_size, dimension, dropout)
        self.enc_1 = TransformerBlock(dimension, attn_heads, config['KANLayers'], config["KANSize1"], config["KANSize2"], config["KANSize3"], config["KANMHA"], config['GridSize'], config['SplineOrder'], dropout)
        self.enc_2 = TransformerBlock(dimension, attn_heads, config['KANLayers'], config["KANSize1"], config["KANSize2"], config["KANSize3"], config["KANMHA"], config['GridSize'], config['SplineOrder'], dropout)
        self.enc_3 = TransformerBlock(dimension, attn_heads, config['KANLayers'], config["KANSize1"], config["KANSize2"], config["KANSize3"], config["KANMHA"], config['GridSize'], config['SplineOrder'], dropout)
        self.enc_4 = TransformerBlock(dimension, attn_heads, config['KANLayers'], config["KANSize1"], config["KANSize2"], config["KANSize3"], config["KANMHA"], config['GridSize'], config['SplineOrder'], dropout)
        self.KAN = KAN([dimension, 229])
        
    def forward(self, x):
        mask = None
        x = self.embeddings(x)
        x = self.enc_1(x, mask)
        x = self.enc_2(x, mask)
        x = self.enc_3(x, mask)
        x = self.enc_4(x, mask)
        x = self.KAN(x[:,0,:])
        return x 
    
config_128 = {'KANLayers': 0, 'Dropout': 0.05283792466895528, 'Lr': 0.0006103489168332225, 'GridSize': 5, 'SplineOrder': 1, 'KANSize1': -1, 'KANSize2': -1, 'KANSize3': -1, 'KANMHA': 0, 'Dimension': 128}

config_256 = {'KANLayers': 0, 'Dropout': 0.06484167717898467, 'Lr': 0.00023036437005511046, 'GridSize': 3, 'SplineOrder': 3, 'KANSize1': -1, 'KANSize2': -1, 'KANSize3': -1, 'KANMHA': 0, 'Dimension': 256}