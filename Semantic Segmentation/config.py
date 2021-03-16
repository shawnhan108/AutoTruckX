import torch

# Basic configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = "SETR-PUP" 
# net = "SETR-MLA" 
# net = "TransUNet-Base" 
# net = "TransUNet-Large" 

# training 
lrate = 0.01
momentum = 0.9
wdecay = 1e-4
fine_tune_ratio = 0.8
is_continue = False
epochs = 25 ## TODO: determine epochs based on number of iterations and dataloader length.

# inference
best_ckpt_src = "./checkpoints/{0}/best_ckpt_1.pth".format(net)
