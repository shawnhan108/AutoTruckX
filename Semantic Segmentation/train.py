import warnings

import torch 
import torch.nn as nn 
from torch.optim import SGD, lr_scheduler

from tensorboardX import SummaryWriter

from SETR_models.setr import get_SETR_PUP, get_SETR_MLA
from TransUNet_models.transunet import get_TransUNet_base, get_TransUNet_large
from utils import get_logger, load_ckpt_continue_training, LossMeter
from config import device, net, lrate, momentum, wdecay, fine_tune_ratio, best_ckpt_src, \
                    is_continue, epochs

def train(cont=False):

    # for tensorboard tracking
    logger = get_logger()
    logger.info("(1) Initiating Training ... ")
    logger.info("Training on device: {}".format(device))
    writer = SummaryWriter()

    # init model 
    if net == "SETR-PUP":
        _, model = get_SETR_PUP()
    elif net == "SETR-MLA":
        _, model = get_SETR_MLA()
    elif net == "TransUNet-Base":
        model = get_TransUNet_base()
    elif net == "TransUNet-Large":
        model = get_TransUNet_large()
    
    # optimizer
    optim = SGD(model.parameters(), lr=lrate, momentum=momentum, weight_decay=wdecay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[int(epochs * fine_tune_ratio)], gamma=0.1)

    cur_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # for continue training
    if cont:
        model, optim, cur_epoch, best_loss = load_ckpt_continue_training(best_ckpt_src, model, optim, logger)
        logger.info("Current best loss: {0}".format(best_loss))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(cur_epoch):
                scheduler.step()
    else:
        model = nn.DataParallel(model)
        model = model.to(device)
    
    logger.info("(2) Model Initiated ... ")
    logger.info("Training model: {}".format(net))

if __name__ == "__main__":
    train(cont=is_continue)
