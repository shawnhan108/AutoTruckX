import logging 
import os

from config import device, IMG_DIM, CLASS_NUM
from SETR_models.setr import get_SETR_PUP, get_SETR_MLA
from TransUNet_models.transunet import get_TransUNet_base, get_TransUNet_large
from unet_model import UNet

from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torchvision import transforms

def get_logger():
    # Initiate a logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def load_ckpt_continue_training(ck_path, model, optimizer, logger):
    model = model.to(device)

    checkpoint = torch.load(ck_path, map_location=torch.device(device))
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = nn.DataParallel(model)
    
    logger.info("Continue training mode, from epoch {0}. Checkpoint loaded.".format(checkpoint['epoch']))

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

class LossMeter(object):
    # To keep track of most recent, average, sum, and count of a loss metric.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def split_img(np_img, img_dim):
    return np_img[:, :img_dim, :], np_img[:, img_dim:, :]

def get_clustering_model(img_dir_src, logger):
    logger.info("Clustering segmentation classes ...")

    colors = []
    img_srcs = os.listdir(img_dir_src)[:15]

    for img_src in img_srcs:
        img, mask = split_img(np.array(Image.open(os.path.join(img_dir_src, img_src)).convert('RGB')), IMG_DIM)
        colors.append(mask.reshape(-1, 3))
    colors = np.array(colors)
    colors = colors.reshape((-1, 3))

    cluster_model = KMeans(CLASS_NUM)
    cluster_model.fit(colors)

    logger.info("Segmentation classes clustering has finished.")
    return cluster_model

class DiceLoss(nn.Module):
    # Dice loss, retrieved from :
    # https://github.com/Beckschen/TransUNet/blob/86d7baffad9e952a90f2901599b35ed0ca1ffa72/utils.py#L9
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def select_model(model_name, init_msg):
    logger = get_logger()
    logger.info(init_msg)
    if model_name == "SETR-PUP":
        _, model = get_SETR_PUP()
    elif model_name == "SETR-MLA":
        _, model = get_SETR_MLA()
    elif model_name == "TransUNet-Base":
        model = get_TransUNet_base()
    elif model_name == "TransUNet-Large":
        model = get_TransUNet_large()
    elif model_name == "UNet":
        model = UNet(CLASS_NUM)
    model = model.to(device)
    
    return logger, model

def load_weights(model, ckpt_src, logger):
    state = torch.load(ckpt_src, map_location=torch.device(device))['model_state_dict']
    for key in list(state.keys()):
        state[key.replace('module.', '')] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("(2) Model Loaded ... ")

def preprocess_img(img):
    # input img : np array, returns a tensor of 1, C, H, W

    img = cv2.resize(img, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    img = img[np.newaxis, :]

    return img/255