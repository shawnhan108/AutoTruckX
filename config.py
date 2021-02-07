import torch
from datetime import datetime

# data
train_img_dir = "./data/train"
valid_img_dir = "./data/valid"
train_ang = "./data/train_record.csv"
valid_ang = "./data/valid_record.csv"

# target network
net = "TruckNN" # "TruckRNN", "TruckInception"

# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
print_freq = 50
tensorboard_freq = 50
epochs = 20
lrate = 1e-4
wdecay = 1e-4
getLoss = torch.nn.MSELoss()

print_freq = 100
tensorboard_freq = 200

curtime = str(datetime.now())
ckpt_name = "./checkpoints/{1}/ckpt_{0}.pth".format(curtime.split(" ")[0] + "_" + 
            curtime.split(" ")[1][0:2] + "_" + curtime.split(" ")[1][3:5], net)
