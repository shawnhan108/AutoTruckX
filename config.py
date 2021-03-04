import torch
from datetime import datetime

# data
img_dir = "./data/IMG"
csv_src = "./data/record.csv"

# target network
# net = "TruckNN" 
# net = "TruckRNN"
net = "TruckResnet50"

# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32 # 8 for LSTM
seq_len = 15 # for LSTM
print_freq = 50
tensorboard_freq = 50
epochs = 40 #20
lrate = 1e-4
wdecay = 1e-4
getLoss = torch.nn.MSELoss()
train_test_split_ratio = 0.8
early_stop_tolerance = 10 #4
fine_tune_ratio = 0.8
is_continue = True

print_freq = 100
tensorboard_freq = 200

curtime = str(datetime.now())
ckpt_src = "./checkpoints/{1}/ckpt_{0}.pth".format(curtime.split(" ")[0] + "_" + 
            curtime.split(" ")[1][0:2] + "_" + curtime.split(" ")[1][3:5], net)
ckpt_src = "./checkpoints/{0}/best_ckpt.pth".format(net)

# inference
best_ckpt_src = "./checkpoints/{0}/best_ckpt_1.pth".format(net)
inf_img_src = "./data/inference/input/test.jpeg"
inf_vid_src = "./data/inference/input/test.mp4"
inf_out_src = "./data/inference/output/output.txt"
inf_out_img_src = "./data/inference/output/output.jpg"
inf_out_vid_src = "./data/inference/output/output.avi"

# visualization
vis_out_src = "./data/inference/vis/out_test.png"
target_layer_name = "layer4"
