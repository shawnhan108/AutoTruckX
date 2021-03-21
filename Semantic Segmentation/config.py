import torch

# Basic configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = "SETR-PUP" 
# net = "SETR-MLA" 
# net = "TransUNet-Base" 
# net = "TransUNet-Large" 
net = "UNet"

# data 
data_dir = "./data/cityscapes"
IMG_DIM = 256
CLASS_NUM = 13

# training 
use_dice_loss = False # True
lrate = 0.001
momentum = 0.9
print_freq = 50
tensorboard_freq = 20
wdecay = 1e-4
fine_tune_ratio = 0.8
early_stop_tolerance = 10 #4
is_continue = False
batch_size = 16
ckpt_src = "./checkpoints/{0}/best_ckpt.pth".format(net)
iteration_num = 80000
epoch_num = 40
# epochs num is determined based on number of iterations and dataloader length.

# inference
best_ckpt_src = "./checkpoints/{0}/U-Net2.pth".format(net)
inf_img_src = "./data/inference/input/test3.jpeg"
inf_vid_src = "./data/inference/input/test.mp4"
inf_out_img_src = "./data/inference/output/output3_3.jpg"
inf_out_vid_src = "./data/inference/output/output.avi"
