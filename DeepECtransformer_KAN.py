import math
import torch
import torch.nn as nn
from efficient_kan.src.efficient_kan import KAN  

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.max_len}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        return self.pos_embed(positions)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=128, dropout=0.1, pad_idx=0, max_len=1000):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
    def forward(self, input_ids):
        x = self.token(input_ids) + self.position(input_ids)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h=8, d_model=128, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None
    def _split_heads(self, x):
        b, l, d = x.size()
        x = x.view(b, l, self.h, self.d_k).permute(0, 2, 1, 3)
        return x
    def forward(self, x, mask=None):
        x = self.norm(x)
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        p_attn = self.softmax(scores)
        p_attn = self.attn_dropout(p_attn)
        self.last_attn = p_attn
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)
        x = self.out_dropout(x)
        return x
    def get_attn(self):
        return self.last_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.l1 = KAN([dimension, 512], grid_size=3, spline_order=3)
        self.l2 = KAN([512, 128], grid_size=3, spline_order=3)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class SublayerConnection(nn.Module):
    def __init__(self, size=128, dropout=0.1):
        super().__init__()
        self.feed_forward = PositionwiseFeedForward(size)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x1, x2):
        return x1 + self.dropout(self.feed_forward(self.norm(x2)))

class TransformerBlock(nn.Module):
    def __init__(self, hidden=128, attn_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        x1 = self.attention(x, mask)
        x = x + x1
        x = self.output_sublayer(x, x)
        return self.dropout(x)

class DeepECtransformer_KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dimension = 128
        self.attn_heads = 8
        self.vocab_size = 23
        self.pad_idx = 0
        self.max_len = 1000
        self.dropout_p = 0.1
        self.embeddings = TransformerEmbedding(self.vocab_size, self.dimension, self.dropout_p, pad_idx=self.pad_idx, max_len=self.max_len)
        self.enc_1 = TransformerBlock(self.dimension, self.attn_heads, self.dropout_p)
        self.enc_2 = TransformerBlock(self.dimension, self.attn_heads, self.dropout_p)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, self.dimension))
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1))
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=(self.max_len - 6, 1), stride=1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU()
        self.KAN = KAN([self.dimension, 1938], grid_size=3, spline_order=3)
    def forward(self, input_ids):
        mask = (input_ids != self.pad_idx)
        x = self.embeddings(input_ids)
        x = self.enc_1(x, mask)
        x = self.enc_2(x, mask)
        x = self.dropout(self.relu(self.bn1(self.conv1(x.unsqueeze(1)))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = x.view(-1, 128)
        x = self.KAN(x)
        return x
