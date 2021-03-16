import torch
import torch.nn as nn 

from encoder import Encoder
from decoder import Decoder 

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling_scale=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling_scale > 1 else nn.Identity()
        )

class TransUNet(nn.Module):
    def __init__(self, dim, heads, mlp_dim, depth, img_dim, grid_dim, res_layer_num, res_width_factor, 
                decoder_channels, skip_num, skip_channels, class_num=19,
                use_vis=True, ff_drop_rate=0.1, attn_drop_rate=0.1):
        super(TransUNet, self).__init__()
        
        self.encoder = Encoder(dim, heads, mlp_dim, depth, img_dim, grid_dim, res_layer_num, res_width_factor, 
                                use_vis=use_vis, ff_drop_rate=ff_drop_rate, attn_drop_rate=attn_drop_rate)
        self.decoder = Decoder(dim, decoder_channels, skip_num, skip_channels)
        self.seg_head = SegmentationHead(decoder_channels[-1], class_num, kernel_size=3)

    def forward(self, x):
        x, attn_weights, features = self.encoder(x)
        x = self.decoder(x, features=features)
        x = self.seg_head(x)

        return x

###########################################
# configure TransUNet variants

# for cityscape dataset
CLASS_NUM = 19

def get_TransUNet_small(img_dim=256, class_num=CLASS_NUM):
    # 105M parameters including res50v2
    model = TransUNet(
        dim=768, 
        heads=12, 
        mlp_dim=3072,
        depth=12, 
        img_dim=img_dim, 
        grid_dim=16, 
        res_layer_num=(3, 4, 9), 
        res_width_factor=1, 
        decoder_channels=(256, 128, 64, 16), 
        skip_num=3, 
        skip_channels=[512, 256, 64, 16], 
        class_num=class_num,
        use_vis=False, 
        ff_drop_rate=0.1, 
        attn_drop_rate=0.0
    )

    return model

def get_TransUNet_large(img_dim=256, class_num=CLASS_NUM):
    # 324M parameters including res50v2
    model = TransUNet(
        dim=1024, 
        heads=16, 
        mlp_dim=4096,
        depth=24, 
        img_dim=img_dim, 
        grid_dim=16, 
        res_layer_num=(3, 4, 9), 
        res_width_factor=1, 
        decoder_channels=(256, 128, 64, 16), 
        skip_num=3, 
        skip_channels=[512, 256, 64, 16], 
        class_num=class_num,
        use_vis=False, 
        ff_drop_rate=0.1, 
        attn_drop_rate=0.0
    )

    return model


"""
t = get_TransUNet_large()
print(t)
print(sum(p.numel() for p in t.parameters()))
x = torch.randn(16, 3, 256, 256)
print(x.size())
x = t(x)
print(x.size())
"""
