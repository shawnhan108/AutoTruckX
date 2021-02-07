import torch
import torch.nn as nn 
from torch.optim import Adam, lr_scheduler
from config import device, epochs, lrate, wdecay, batch_size, getLoss, print_freq, tensorboard_freq, ckpt_name
from utils import group_move_to_device, LossMeter, get_logger
from tensorboardX import SummaryWriter

from TruckNN import TruckNN, TruckInception, TruckRNN

"""
Input Dimension Validation: 

TruckNN: N x 3 x 80 x 240 -> N x 1
TruckRNN: N x 3 x 15 x 80 x 240 -> N x 5
TruckInception: N x 3 x 299 x 299 -> N x 1
"""

def train():  

    # For tensorboard tracking
    logger = get_logger()
    logger.info("(1) Initiating training ... ")
    writer = SummaryWriter()

    # Init model
    model = TruckNN()
    model = nn.DataParallel(model)
    model = model.to(device)
    logger.info("(2) Model Initiated ... ")

    # Schedule learning rate. Fine-tune after 25th epoch for 5 more epochs.
    optim = Adam(model.parameters(), lr=lrate, weight_decay=wdecay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[17], gamma=0.1)

    # TODO: Dataset and DataLoaders
    # (1) must cast from np to tensor: torch.from_numpy(x).permute(2, 0, 1) # D, H, W
    # (2) transformations: resize, totensor, normalize, augmentation?
    # (3) must return leftImg, centerImg, rightImg, leftAng, centerAng, rightAng, all of shape N x Img
    
    logger.info("(3) Dataset Initiated. Training Started. ")

    best_mse = float('inf')
    epochs_since_improvement = 0

    # loop over epochs
    for epoch in range(epochs):

        # Training.
        model.train()
        trainLossMeter = LossMeter()

        for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(train_loader):

            leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

            optim.zero_grad()
            for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:

                y_pred = model(img)
                y_pred = y_pred.unsqueeze(1) # of shape N x 1
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
                                epoch * (len(train_loader) / tensorboard_freq) + (i+1) / tensorboard_freq)

        writer.add_scalar('Train_Loss_epoch', trainLossMeter.avg, epoch)

        # Validation.
        model.eval()
        validLossMeter = LossMeter()

        with torch.no_grad():
            for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(valid_loader):

                leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

                for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:

                    y_pred = model(img)
                    y_pred = y_pred.unsqueeze(1) # of shape N x 1
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
                                    epoch * (len(valid_loader) / tensorboard_freq) + (i+1) / tensorboard_freq)

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
            torch.save(state, ckpt_name)
            logger.info("Checkpoint updated.")

    writer.close()
