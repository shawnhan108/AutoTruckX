import torch
import torch.nn as nn 

from transformer import Transformer
from res50fcn import ResNetV2
from position import LearnedPosEmbedding, FixedPosEmbedding

class SETR(nn.Module):
    def __init__(self, 
        img_dim, patch_dim, embedding_dim, hidden_dim, 
        channel_num, head_num, attn_depth, 
        ff_dropout_rate=0, attn_drop_rate=0, 
        conv_patch_extract=False, pos_encode="learned"):

        super(SETR, self).__init__()
        assert embedding_dim % head_num == 0

        # basic hyperparams
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.channel_num = channel_num
        self.head_num = head_num
        self.ff_dropout_rate = ff_dropout_rate
        self.attn_drop_rate = attn_drop_rate
        self.conv_patch_extract = conv_patch_extract # whether to use conv to extract img patches

        assert img_dim % patch_dim == 0
        self.patch_num = (int(img_dim // patch_dim)) ** 2       # HW/pd^2
        self.flat_dim = (patch_dim ** 2) * channel_num          # flat_dim = pd ^2 * c
        self.seqlen = self.patch_num                            # seqlen = HW/pd^2
        
        # layers
        self.in_proj = nn.Linear(self.flat_dim, self.embedding_dim)
        if pos_encode == "learned":
            self.pos_encode = LearnedPosEmbedding(self.embedding_dim, self.seqlen, self.seqlen)
        elif pos_encode == "fixed":
            self.pos_encode = FixedPosEmbedding(self.embedding_dim)
        self.pos_encode_drop = nn.Dropout(p=self.ff_dropout_rate)

        self.transformer = Transformer(dim=self.embedding_dim, depth=attn_depth, 
                                        heads=head_num, mlp_dim=hidden_dim, 
                                        ff_drop_rate=self.ff_dropout_rate, attn_drop_rate=self.attn_drop_rate)
        
        self.conv_patch = None
        if self.conv_patch_extract:
            self.conv_extract = nn.Conv2d(self.channel_num, self.embedding_dim, kernel_size=(self.patch_dim, self.patch_dim),
                                        stride=(self.patch_dim, self.patch_dim), padding=0)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)


    def encode(self, x):

        # transform input to N x seqlen x embed_dim
        n, c, h, w = x.shape                                                    # N, C, H, W
        if self.conv_patch_extract:
            x = self.conv_extract(x)                                            # N, embed_dim, H/pd, W/pd
            x = x.permute(0, 2, 3, 1).contiguous()                              # N, H/pd, W/pd, embed_dim
            x = x.view(x.size(0), -1, self.embedding_dim)                       # N, HW/pd^2(seqlen), embed_dim
        else:
            x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim).contiguous() 
                                                                                # N, C, H/pd, W/pd, pd, pd
            x = x.view(n, c, -1, self.patch_dim ** 2)                           # N, C, HW/pd^2, pd^2
            x = x.permute(0, 2, 3, 1).contiguous()                              # N, HW/pd^2(seqlen), pd^2, C 
            x = x.view(x.size(0), -1, self.flat_dim)                            # N, seqlen, flat_dim(pd^2 * c)
            x = self.in_proj(x)                                                 # N, seqlen, embed_dim
        
        # positional encode, then pass to transformer
        x = self.pos_encode_drop(self.pos_encode(x))
        x, zs = self.transformer(x)
        x = self.layer_norm(x)

        return x, zs

    def forward(self, x, intermediate_layers_to_extract=None):
        x, zs = self.encode(x)
        x = self.decode(x, zs, intermediate_layers_to_extract)

        if intermediate_layers_to_extract is not None:
            feature_dict = {}
            for idx in intermediate_layers_to_extract:
                feature_dict['Z{0}'.format(idx)] = zs[str(2 * idx - 1)]
            
            return x, feature_dict

        return x

    def decode(self, x):
        pass
    
    def _init_decode(self):
        pass

