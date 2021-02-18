import torch
from datetime import datetime

# data
img_dir = "./data/IMG"
csv_src = "./data/record.csv"

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
ckpt_src = "./checkpoints/{1}/ckpt_{0}.pth".format(curtime.split(" ")[0] + "_" + 
            curtime.split(" ")[1][0:2] + "_" + curtime.split(" ")[1][3:5], net)

# inference
best_ckpt_src = "./checkpoints/{0}/best_ckpt.pth".format(net)
inf_img_src = "./data/inference/input/test.jpeg"
inf_vid_src = "./data/inference/input/test.mp4"
inf_out_src = "./data/inference/output/output.txt"
inf_out_img_src = "./data/inference/output/output.jpg"
inf_out_vid_src = "./data/inference/output/output.avi"
