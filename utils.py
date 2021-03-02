import logging

import torch
from torchvision import transforms 
import numpy as np

from config import device
from models import TruckNN, TruckResnet50, TruckRNN

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

def group_move_to_device(lst):
    # Accept input as a list of tensors, return a list of all tensors moved to device.
    for (i, item) in enumerate(lst):
        lst[i] = item.float().to(device)
    return lst

def get_logger():
    # Initiate a logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def select_model(model_name, init_msg):
    logger = get_logger()
    logger.info(init_msg)
    if model_name == "TruckNN":
        model = TruckNN()
    elif model_name == "TruckResnet50":
        model = TruckResnet50()
    elif model_name == "TruckRNN":
        model = TruckRNN()
    model = model.to(device)
    
    return logger, model

def load_weights(model, ckpt_src, logger):
    state = torch.load(ckpt_src, map_location=torch.device(device))['model_state_dict']
    for key in list(state.keys()):
        state[key.replace('module.', '')] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("(2) Model Loaded ... ")

def preprocess_img(img, model_name):
    # input img : np array, returns a tensor of 1, C, H, W
    img = torch.from_numpy(img).permute(2, 0, 1) # D, H, W

    if model_name == "TruckNN":
        size = (80, 240)
    elif model_name == "TruckResnet50":
        size = (224, 224)
    elif model_name == "TruckRNN":
        size = (80, 240)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Lambda(lambda x: (x / 127.5) - 1),
    ])
    img = transform(img)
    img = img[np.newaxis, :]

    return img

def load_ckpt_continue_training(ck_path, model, optimizer, logger):
    checkpoint = torch.load(ck_path, map_location=torch.device(device))
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info("Continue training mode, from epoch {0}. Checkpoint loaded.".format(checkpoint['epoch']))

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']