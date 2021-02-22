import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tensorboardX import SummaryWriter

from config import device, epochs, lrate, wdecay, batch_size, getLoss, print_freq, tensorboard_freq, ckpt_src, net, \
                    img_dir, csv_src
from utils import group_move_to_device, LossMeter, get_logger
from models import TruckNN, TruckResnet50, TruckRNN
from data import TruckDataset

"""
Input Dimension Validation: 

TruckNN: N x 3 x 80 x 240 -> N x 1
TruckRNN: N x 3 x 15 x 80 x 240 -> N x 5
TruckInception: N x 3 x 299 x 299 -> N x 1
"""

def train():
    def loadData():
        data = pd.read_csv(csv_src)
        X = data[data.columns[0]].values
        y = data[data.columns[3]].values

        return X, y

    # For tensorboard tracking
    logger = get_logger()
    logger.info("(1) Initiating Training ... ")
    logger.info("Training on device: {}".format(device))
    writer = SummaryWriter()

    # Init model
    if net == "TruckNN":
        model = TruckNN()
    elif net == "TruckResnet50":
        model = TruckResnet50()
    model = nn.DataParallel(model)
    model = model.to(device)
    logger.info("(2) Model Initiated ... ")
    logger.info("Training model: {}".format(net))

    # Schedule learning rate. Fine-tune after 25th epoch for 5 more epochs.
    optim = Adam(model.parameters(), lr=lrate, weight_decay=wdecay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[17], gamma=0.1)

    # Dataset and DataLoaders
    img_src_lst, angles = loadData()
    X_train, X_valid, y_train, y_valid = train_test_split(img_src_lst, angles, test_size=0.25, random_state=0, shuffle=True)
    train_dataset = TruckDataset(X=X_train, y=y_train)
    valid_dataset = TruckDataset(X=X_valid, y=y_valid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("(3) Dataset Initiated. Training Started. ")

    best_mse = float('inf')
    epochs_since_improvement = 0

    # loop over epochs
    epoch_bar = tqdm.tqdm(total=epochs, desc="Epoch", position=0, leave=True)
    for epoch in range(epochs):

        # Training.
        model.train()
        trainLossMeter = LossMeter()
        train_batch_bar = tqdm.tqdm(total=len(train_loader), desc="TrainBatch", position=0, leave=True)
        for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(train_loader):

            leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

            optim.zero_grad()
            for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:

                y_pred = model(img)
                y_train = y_train.unsqueeze(1) # of shape N x 1
                loss = getLoss(y_pred, y_train)

                # Backward Propagation, Update weight and metrics
                loss.backward()
                optim.step()

                # Update loss
                trainLossMeter.update(loss.item())

            # print status
            if (batch_num+1) % print_freq == 0:
                status = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, batch_num+1, len(train_loader), loss=trainLossMeter)
                logger.info(status)

            # log loss to tensorboard 
            if (batch_num+1) % tensorboard_freq == 0:
                writer.add_scalar('Train_Loss_{0}'.format(tensorboard_freq), 
                                trainLossMeter.avg, 
                                epoch * (len(train_loader) / tensorboard_freq) + (batch_num+1) / tensorboard_freq)
            train_batch_bar.update(1)

        writer.add_scalar('Train_Loss_epoch', trainLossMeter.avg, epoch)

        # Validation.
        model.eval()
        validLossMeter = LossMeter()
        valid_batch_bar = tqdm.tqdm(total=len(valid_loader), desc="ValidBatch", position=0, leave=True)
        with torch.no_grad():
            for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(valid_loader):

                leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

                for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:

                    y_pred = model(img)
                    y_train = y_train.unsqueeze(1) # of shape N x 1
                    loss = getLoss(y_pred, y_train)

                    # Update loss
                    validLossMeter.update(loss.item())

                # print status
                if (batch_num+1) % print_freq == 0:
                    status = 'Validation: [{0}][{1}/{2}]\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, batch_num+1, len(valid_loader), loss=validLossMeter)
                    logger.info(status)

                # log loss to tensorboard 
                if (batch_num+1) % tensorboard_freq == 0:
                    writer.add_scalar('Valid_Loss_{0}'.format(tensorboard_freq), 
                                    validLossMeter.avg, 
                                    epoch * (len(valid_loader) / tensorboard_freq) + (batch_num+1) / tensorboard_freq)
                valid_batch_bar.update(1)
        valid_loss = validLossMeter.avg
        writer.add_scalar('Valid_Loss_epoch', valid_loss, epoch)
        logger.info("Validation Loss of epoch [{0}/{1}]: {2}\n".format(epoch+1, epochs, valid_loss))    
    
        # update optim scheduler
        scheduler.step()

        # save checkpoint 
        is_best = valid_loss < best_mse
        best_loss = min(valid_loss, best_mse)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            state = {
                'epoch': epoch,
                'loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }
            torch.save(state, ckpt_src)
            logger.info("Checkpoint updated.")

        epoch_bar.update(1)
    writer.close()

if __name__ == "__main__":
    train()
