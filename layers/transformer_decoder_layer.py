
# MIT License

# Copyright (c) 2024 D. Carpintero
# Modifications Copyright (c)  Da Saem Lee, 2026
import torch
from torch import nn
import math
from layers.multihead_diffattn import MultiheadDiffAttn

class AttentionHead(nn.Module):
    def __init__(self, n_embd, n_headd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, n_headd * 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        
        return torch.bmm(attn_probs, v)
    
    def forward(self, x, mask):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.scaled_dot_product_attention(q, k, v, mask=mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.output_linear = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask):
        return self.output_linear(torch.cat([head(x, mask) for head in self.heads], dim=-1))


class CrossAttentionHead(nn.Module):
    def __init__(self, n_embd, n_headd):
        super().__init__()
        self.q = nn.Linear(n_embd, n_headd)
        self.k = nn.Linear(n_embd, n_headd)
        self.v = nn.Linear(n_embd, n_headd)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0) 

        return torch.bmm(attn_probs, v)

    def forward(self, q, kv, mask):
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)
        
        return self.scaled_dot_product_attention(q, k, v, mask=mask)
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.heads = nn.ModuleList([CrossAttentionHead(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.output_linear = nn.Linear(n_embd, n_embd)

    def forward(self, q, kv, attn_mask):
        return self.output_linear(torch.cat([head(q, kv, attn_mask) for head in self.heads], dim=-1))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(n_embd, ff_dim), 
                                nn.GELU(), 
                                nn.Dropout(0.1),
                                nn.Linear(ff_dim, n_embd))

    def forward(self, x):
        return self.ff(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim, dropout=0.1):
        super().__init__()
        
        self.a2a_mha = MultiHeadAttention(n_embd, n_head)
        self.map2a = MultiHeadCrossAttention(n_embd, n_head)
        
        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.norm_3 = nn.LayerNorm(n_embd)
        
        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, map_enc, map_mask=None, mask=None):
        attn = self.map2a(x, map_enc, attn_mask=map_mask)
        x = self.norm_1(x + self.dropout(attn))
        x = self.norm_2(x + self.dropout(self.a2a_mha(x, mask)))

        return self.norm_3(x + self.feed_forward(x))


class TransformerDecoderLayerDiff(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim, layer_id, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.a2a_mha = MultiHeadAttention(n_embd, n_head)
        self.t2a_mha = MultiHeadAttention(n_embd, n_head)
        
        self.map2a = MultiheadDiffAttn(n_embd, layer_id, n_head)
        self.map2a_refinement = MultiheadDiffAttn(n_embd, layer_id, n_head)
        self.map2a_summary = MultiHeadCrossAttention(n_embd, n_head)
        
        
        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.norm_3 = nn.LayerNorm(n_embd)
        self.norm_4 = nn.LayerNorm(n_embd)
        self.norm_5 = nn.LayerNorm(n_embd)

        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, map_enc, map_mask=None, mask=None, map_summary=None):
        
        B, N, T, D = x.shape
        B, _, N_map = map_mask.shape
        
        map_mask = map_mask.view(B, 1, 1, N_map).expand(-1, self.n_head*2, N*T, -1)

        # attending summary first
        x = x.reshape(B, -1, D)
        m2a_summary = self.map2a_summary(self.norm_1(x), map_summary, attn_mask=None)
        x = x + self.dropout(m2a_summary)

        # t2a
        x = x.reshape(B, N, T, D).reshape(-1, T, D)
        x_t2a = self.t2a_mha(self.norm_2(x), None)
        x = x + self.dropout(x_t2a)
        
        #a2a
        x = x.reshape(B, N, T, D).permute(0,2,1,3).contiguous()
        x = x.reshape(B*T, N, D)
        x2a = self.a2a_mha( self.norm_3(x), mask)
        x = x + self.dropout(x2a)

        x = x.reshape(B, T, N, D).permute(0,2,1,3).contiguous()
        x = x.reshape(B, -1, D)    
        x2map, x2map_weights = self.map2a(self.norm_4(x), map_enc, attn_mask=map_mask)
        x = x + self.dropout(x2map) 
        x = x.reshape(B, N, T, D)

        return x + self.feed_forward(self.norm_5(x)), x2map_weights
