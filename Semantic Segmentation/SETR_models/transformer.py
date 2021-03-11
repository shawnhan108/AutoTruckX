import torch
from torch import nn, einsum

from einops import rearrange

class ResNorm(nn.Module):
    # "ADD & Norm" in the original paper
    # Apply residual and normalization.
    def __init__(self, dim, layer, dropout_rate=0):
        super().__init__()
        self.layer = layer 
        self.norm = nn.LayerNorm(dim)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout:
            return self.dropout(self.layer(self.norm(x))) + x
        return self.layer(self.norm(x)) + x


class MultiHeadAttention(nn.Module):
    # Transformer part A: Multi-Head Attention
    def __init__(self, dim, heads=8, qkv_bias=False, dropout_rate=0):
        super().__init__()
        self.heads_num = heads
        head_dim = dim // heads 
        inner_dim = head_dim * heads 

        self.in_proj = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.qk_scale = head_dim ** -0.5 # sqrt(d)
        self.dropout = nn.Dropout(dropout_rate)

        self.out_proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        # input x is of shape (batchsize, seqlen, embedding_size)
        # divide output to a tuple of 3 matrices
        # then dimension decomposition into number of heads * head_dim
        # each of the q,k,v has dimension batchsize, head_num, seqsize, querysize
        qkv = self.in_proj(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads_num), qkv)

        # omit masking, directly compute attention score qk using einsum, then softmax and dropout
        qk = einsum('b h i d, b h j d -> b h i j', q, k) * self.qk_scale
        qk = qk.softmax(dim=-1)
        qk = self.dropout(qk)

        # multiply with v=keys, reshape to (batchsize, seqlen, embedding_size)
        x = einsum('b h i j, b h j d -> b h i d', qk, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.out_proj(x)
        return x


class FeedForward(nn.Module):
    # Transformer part B: Feedforward module
    def __init__(self, dim, hidden_dim, dropout_rate=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    # full transformer module
    def __init__(self, dim, depth, heads, mlp_dim, ff_drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResNorm(dim, MultiHeadAttention(dim, heads=heads, dropout_rate=attn_drop_rate),dropout_rate=ff_drop_rate),
                ResNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout_rate=ff_drop_rate))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = ff(attn(x))
        return x


"""
t = Transformer(dim=32, depth=5, heads=8, mlp_dim=40)
print(t)
print(sum(p.numel() for p in t.parameters()))
x = torch.randn(2, 3, 224, 224)
print(x.size())
x = t(x)
print(x.size())
"""
