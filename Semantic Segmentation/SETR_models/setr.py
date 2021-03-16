import torch
import torch.nn as nn 

from SETR_models.transformer import Transformer, InterSeq
from SETR_models.position import LearnedPosEmbedding, FixedPosEmbedding

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
        x = self.pos_encode_drop(self.pos_encode(x))                            # N, seqlen, embed_dim
        x, zs = self.transformer(x)                                             # N, seqlen, embed_dim
        print('out of transformer')
        x = self.layer_norm(x)

        return x, zs

    def forward(self, x, intermediate_layers_to_extract=None):
        x, zs = self.encode(x)                                                  # N, C, H, W -> N, seqlen, embed_dim
        x = self.decode(x, zs, intermediate_layers_to_extract)                  # N, seqlen, embed_dim -> N, class_num, H, W
        
        if intermediate_layers_to_extract is not None:
            feature_dict = {}
            for idx in intermediate_layers_to_extract:
                feature_dict['Z{0}'.format(idx)] = zs[str(2 * idx - 1)]
            
            return x, feature_dict
        return x
    
    def reshape_for_decoder(self, x):
        # x of shape N, seq_len=HW/pd^2, embed_dim 
        x = x.view(x.size(0), int(self.img_dim / self.patch_dim), int(self.img_dim / self.patch_dim), self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous() # x of shape N, embed_dim, H/pd, W/pd

    def decode(self, x):
        pass
    
    def _init_decode(self):
        pass


###########################################
# The variants of SETR, with decoders

class SETR_PUP(SETR):
    def __init__(self, 
        img_dim, patch_dim, embedding_dim, hidden_dim, 
        channel_num, head_num, class_num, attn_depth, 
        ff_dropout_rate=0, attn_drop_rate=0, 
        conv_patch_extract=False, pos_encode="learned"):
        super(SETR_PUP, self).__init__(
            img_dim=img_dim, patch_dim=patch_dim, embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, channel_num=channel_num, head_num=head_num, 
            attn_depth=attn_depth, ff_dropout_rate=ff_dropout_rate, 
            attn_drop_rate=attn_drop_rate, conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode)

        self.class_num = class_num
        self._init_decode()
    
    def _init_decode(self):
        reduce_dim = int(self.embedding_dim / 4)
        in_out_pairs = [
            (self.embedding_dim, reduce_dim),
            (reduce_dim, reduce_dim),
            (reduce_dim, reduce_dim),
            (reduce_dim, reduce_dim),
            (reduce_dim, self.class_num)
        ]

        layers = []
        for i, (in_channel, out_channel) in enumerate(in_out_pairs):
            layers.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            if i != 4:
                layers.append(nn.Upsample(scale_factor=int(self.patch_dim/4), mode='bilinear'))
        
        self.decoder = InterSeq(*layers, inter_results=False)
    
    def decode(self, x, inter_results, intmd_layers=None):
        # x of shape N, seq_len=HW/pd^2, embed_dim 
        x = self.reshape_for_decoder(x)         # x of shape N, embed_dim, H/pd, W/pd
        x = self.decoder(x)                     # x of shape N, class_num, H, W
        return x


class SETR_MLA(SETR):
    def __init__(self, 
        img_dim, patch_dim, embedding_dim, hidden_dim, 
        channel_num, head_num, class_num, attn_depth, 
        ff_dropout_rate=0, attn_drop_rate=0, 
        conv_patch_extract=False, pos_encode="learned"):
        super(SETR_MLA, self).__init__(
            img_dim=img_dim, patch_dim=patch_dim, embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, channel_num=channel_num, head_num=head_num, 
            attn_depth=attn_depth, ff_dropout_rate=ff_dropout_rate, 
            attn_drop_rate=attn_drop_rate, conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode)

        self.class_num = class_num
        self._init_decode()
    
    def get_aggregation_net(self): 
        model_in = InterSeq(inter_results=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(self.embedding_dim, int(self.embedding_dim/2), kernel_size=1)
        )

        model_mid = InterSeq(inter_results=False)
        model_mid.add_module(
            "layer_mid",
            nn.Conv2d(int(self.embedding_dim/2), int(self.embedding_dim/2), kernel_size=3, padding=(1,1))
        )

        model_out = InterSeq(inter_results=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(int(self.embedding_dim/2), int(self.embedding_dim/2), kernel_size=3, padding=(1,1))
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(int(self.embedding_dim/2), int(self.embedding_dim/4), kernel_size=3, padding=(1,1))
        )
        model_out.add_module(
            "upsample",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_mid, model_out # ed->ed/2, ed/2->ed/2, ed/2->ed/4 + up4
    
    def _init_decode(self):
        self.stream1_in, self.stream1_mid, self.stream1_out = self.get_aggregation_net()
        self.stream2_in, self.stream2_mid, self.stream2_out = self.get_aggregation_net()
        self.stream3_in, self.stream3_mid, self.stream3_out = self.get_aggregation_net()
        self.stream4_in, self.stream4_mid, self.stream4_out = self.get_aggregation_net()

        self.out_net = InterSeq(inter_results=False)
        self.out_net.add_module(
            "conv_1",
            nn.Conv2d(in_channels=self.embedding_dim, out_channels=self.class_num, kernel_size=1)
        )
        self.out_net.add_module(
            "upsample_1",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )

    def decode(self, x, inter_results, intmd_layers=None):
        assert intmd_layers is not None

        # get layer features: Z24, Z18, Z12, Z6
        zs = {}
        z_indices = []
        for i in intmd_layers:
            z_index = 'Z' + str(i)
            z_indices.append(z_index)
            zs[z_index] = inter_results[str(2 * i - 1)]
        
        z_indices.reverse()

        # Z24 
        z24_in = zs[z_indices[0]]
        z24_in = self.reshape_for_decoder(z24_in)
        z24_mid = self.stream1_in(z24_in)
        z24_out = self.stream1_out(z24_mid)

        # z18
        z18_in = zs[z_indices[1]]
        z18_in = self.reshape_for_decoder(z18_in)
        z18_mid = self.stream2_in(z18_in)
        z18_mid = z24_mid + z18_mid
        z18_out = self.stream2_mid(z18_mid)
        z18_out = self.stream2_out(z18_out)

        # z12 
        z12_in = zs[z_indices[2]]
        z12_in = self.reshape_for_decoder(z12_in)
        z12_mid = self.stream3_in(z12_in)
        z12_mid = z12_mid + z18_mid
        z12_out = self.stream3_mid(z12_mid)
        z12_out = self.stream3_out(z12_out)

        # z6 
        z6_in = zs[z_indices[2]]
        z6_in = self.reshape_for_decoder(z6_in)
        z6_mid = self.stream4_in(z6_in)
        z6_mid = z6_mid + z12_mid
        z6_out = self.stream4_mid(z6_mid)
        z6_out = self.stream4_out(z6_out)

        # aggregate and output 
        out = torch.cat((z24_out, z18_out, z12_out, z6_out), dim=1)
        out = self.out_net(out)

        return out

###########################################
# configure SETR variants with layer depths and dataset specifications

def get_SETR_PUP(dataset='cityscapes', size='s', conv_patch_extract=False, pos_encode="learned"):
    if dataset == 'cityscapes':
        img_dim = 768
        class_num = 19

    size = size.upper()
    assert size in ['S', 'L']    
    if size == 'S':
        aux_layers = None
        model = SETR_PUP(
            img_dim=img_dim, 
            patch_dim=16, 
            embedding_dim=768, 
            hidden_dim=3072, 
            channel_num=3, 
            head_num=12, 
            class_num=class_num, 
            attn_depth=12, 
            ff_dropout_rate=0.1, 
            attn_drop_rate=0.1, 
            conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode
        )
    elif size == 'L':
        aux_layers = [10, 15, 20, 24]
        model = SETR_PUP(
            img_dim=img_dim, 
            patch_dim=16, 
            embedding_dim=1024, 
            hidden_dim=4096, 
            channel_num=3, 
            head_num=16, 
            class_num=class_num, 
            attn_depth=24, 
            ff_dropout_rate=0.1, 
            attn_drop_rate=0.1, 
            conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode
        )

    return aux_layers, model

def get_SETR_MLA(dataset='cityscapes', size='s', conv_patch_extract=False, pos_encode="learned"):
    if dataset == 'cityscapes':
        img_dim = 768
        class_num = 19
    
    size = size.upper()
    assert size in ['S', 'L']    
    if size == 'S':
        aux_layers = None
        model = SETR_MLA(
            img_dim=img_dim, 
            patch_dim=16, 
            embedding_dim=768, 
            hidden_dim=3072, 
            channel_num=3, 
            head_num=12, 
            class_num=class_num, 
            attn_depth=12, 
            ff_dropout_rate=0.1, 
            attn_drop_rate=0.1, 
            conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode
        )
    elif size == 'L':
        aux_layers = [6, 12, 18, 24]
        model = SETR_MLA(
            img_dim=img_dim, 
            patch_dim=16, 
            embedding_dim=1024, 
            hidden_dim=4096, 
            channel_num=3, 
            head_num=16, 
            class_num=class_num, 
            attn_depth=24, 
            ff_dropout_rate=0.1, 
            attn_drop_rate=0.1, 
            conv_patch_extract=conv_patch_extract, 
            pos_encode=pos_encode
        )

    return aux_layers, model


"""
_, t = get_SETR_PUP()
print(t)
print(sum(p.numel() for p in t.parameters()))
x = torch.randn(16, 3, 768, 768)
print(x.size())
x = t(x)
print(x.size())
"""
