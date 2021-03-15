import torch
import torch.nn as nn, einsum

from resnet import ResNetV2

from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout_rate=0, use_vis=True):
        super(MultiHeadAttention, self).__init__()
        self.use_vis = use_vis
        self.head_num = heads
        self.head_dim = int(dim // heads)
        self.inner_dim = self.head_num * self.head_dim

        self.in_proj_q = nn.Linear(dim, inner_dim)
        self.in_proj_k = nn.Linear(dim, inner_dim)
        self.in_proj_v = nn.Linear(dim, inner_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def reshape(self, x):
        # (batchsize, seqlen, embedding_size) -> (batchsize, head_num, seqlen, querysize)
        x = x.view(x.size(0), x.size(1), self.head_num, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.reshape(self.in_proj_q(x))
        k = self.reshape(self.in_proj_k(x))
        v = self.reshape(self.in_proj_v(x))

        qk = torch.matmul(q, k.transpose(-1, -2))
        qk = x / (self.head_dim ** -0.5) # sqrt(d))
        qk = qk.softmax(dim=-1)
        weights = qk if self.use_vis else None
        qk = self.dropout(qk)

        x = einsum('b h i j, b h j d -> b h i d', qk, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.out_proj(x)
        return x, weights


class FeedForward(nn.Module):
    # Transformer part B: Feedforward module
    def __init__(self, dim, hidden_dim, dropout_rate=0):
        super(FeedForward, self).__init__()
        fc1 = nn.Linear(dim, hidden_dim)
        self.activ = nn.GELU()
        dropout1 = nn.Dropout(p=dropout_rate)
        fc2 = nn.Linear(hidden_dim, dim)
        dropout2 = nn.Dropout(p=dropout_rate)

        nn.init.xavier_uniform_(fc1.weight)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.normal_(fc1.bias, std=1e-6)
        nn.init.normal_(fc2.bias, std=1e-6)

        self.net = nn.Sequential(fc1, nn.GELU(), dropout1, fc2, dropout2)

    def forward(self, x):
        return self.net(x)

class PosEmbedding(nn.Module):
    # Embedding of the hybrid variant
    def __init__(self, img_dim, dim, res_layer_num, res_width_factor, dropout_rate, in_channels=3):
        super(Embedding, self).__init__()
        patch_size = (1, 1)
        patch_num = img_dim

        self.hybrid_model = ResNetV2(block_units=res_layer_num, width_factor=res_width_factor)

        in_channels = self.hybrid_model.width * 16
        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=dim,
                                        kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, patch_num, dim))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x, features = self.hybrid_model(x)  # N x dim x H/pd x W/pd
        x = self.patch_embedding(x)         # N x dim x H/pd x W/pd
        x = x.flatten(2).transpose(-1, -2)  # N x HW x dim

        x = x + self.pos_embedding
        x = self.dropout(x)
        return x, features

class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, use_vis=True,
                ff_drop_rate=0.1, attn_drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, heads=heads, dropout_rate=attn_drop_rate, use_vis=use_vis)
        self.ff = FeedForward(dim, hidden_dim=mlp_dim, dropout_rate=ff_drop_rate)
    
    def forward(self, x):
        res = x 
        x, weights = self.attn(self.attn_norm(x))
        x = x + res

        res = x 
        x = self.ff(self.ff_norm(x))
        x = x + res 
        return x, weights

class Encoder_No_Embed(nn.Module):
    def __init__(self, dim, heads, depth, use_vis=True,
                ff_drop_rate=0.1, attn_drop_rate=0.1):
        super(Encoder_No_Embed, self).__init__()
        self.use_vis = use_vis
        self.layers = nn.ModuleList()
        self.norm = nn.Linear(dim, eps=1e-6)

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, use_vis=use_vis,
                                                ff_drop_rate=ff_drop_rate, attn_drop_rate=attn_drop_rate))

    def forward(self, x):
        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x)
            if self.use_vis:
                attn_weights.append(weights)
        
        return self.norm(x), attn_weights

class Encoder(nn.Module):
    def __init__(self, dim, heads, depth, img_dim, res_layer_num, res_width_factor, use_vis=True,
                ff_drop_rate=0.1, attn_drop_rate=0.1):
        super(Encoder, self).__init__()
        self.embedding = PosEmbedding(img_dim=img_dim, dim=dim, res_layer_num=res_layer_num, 
                                    res_width_factor=res_width_factor, dropout_rate=ff_drop_rate)
        self.encoder = Encoder_No_Embed(dim=dim, heads=heads, depth=depth, use_vis=use_vis,
                                        ff_drop_rate=ff_drop_rate, attn_drop_rate=attn_drop_rate)
    
    def forward(self, x):
        x, features = self.embedding(x)
        x, attn_weights = self.encoder(x)
        return x, attn_weights, features
