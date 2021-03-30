import torch
from torch import nn
import torch.optim as optim

import numpy as np
import math
import random
import time


class PositionWiseFeedforwardLayer(nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionWiseFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        input x size = [batch_size, seq_len, hid_dim]
        '''
        x = self.dropout(torch.relu(self.fc_1(x))) # size = [batch_size, seq_len, pf_dim]
        
        x = self.fc_2(x) # size = [batch_size, seq_len, hid_dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query size = [batch size, query len, hid dim]
        # key size = [batch size, key len, hid dim]
        # value size = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        
        Q = self.fc_q(query) # Q size = [batch_size, query_len, hid dim]
        K = self.fc_k(key) # K size = [batch_size, key_len, hid_dim]
        V = self.fc_v(value) # V size = [batch_size, value_len, hid_dim]

        # size = [batch_size, n_heads, query_len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        # energy size = [batch_size, n_heads, query_len, head_dim]
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if not mask:
            energy = energy.masked_fill(mask==0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super(EncoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_ff = PositionWiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x size = [batch_size, src_len, hid_dim]
        # mask = [batch_size, 1, 1, src_len]

        # self attention
        x, _ = self.self_attention(x, x, x, mask)
        # dropout, res net and layer norm
        x = self.self_attn_layer_norm(x + self.dropout(x))

        # x size = [batch_size, src_len, hid_dim]
        # positionwise feedforward
        x = self.positionwise_ff(x)
        x = self.ff_layer_norm(x + self.dropout(x))

        return x
    

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim,
                n_layers, n_heads, pf_dim,
                dropout, device, max_lebgth=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_lebgth, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads,
                                        pf_dim, dropout, device) for _ in range(n_layers)])

        self.dropout = dropout
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # src = [batch_size, src_len, hid_dim]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)

        return src
    