import warnings
import os

import torch 
import torch.nn as nn 
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from tensorboardX import SummaryWriter
import tqdm

from SETR_models.setr import get_SETR_PUP, get_SETR_MLA
from TransUNet_models.transunet import get_TransUNet_base, get_TransUNet_large
from unet_model import UNet

from data import CityscapeDataset
from utils import get_logger, load_ckpt_continue_training, LossMeter, get_clustering_model, DiceLoss
from config import device, net, lrate, momentum, wdecay, fine_tune_ratio, best_ckpt_src, \
                    is_continue, iteration_num, IMG_DIM, data_dir, batch_size, print_freq, \
                    tensorboard_freq, CLASS_NUM, ckpt_src, early_stop_tolerance, epoch_num

def train(cont=False):

    # for tensorboard tracking
    logger = get_logger()
    logger.info("(1) Initiating Training ... ")
    logger.info("Training on device: {}".format(device))
    writer = SummaryWriter()

    # init model 
    aux_layers = None
    if net == "SETR-PUP":
        aux_layers, model = get_SETR_PUP()
    elif net == "SETR-MLA":
        aux_layers, model = get_SETR_MLA()
    elif net == "TransUNet-Base":
        model = get_TransUNet_base()
    elif net == "TransUNet-Large":
        model = get_TransUNet_large()
    elif net == "UNet":
        model = UNet(CLASS_NUM)
    
    # prepare dataset 
    cluster_model = get_clustering_model(os.path.join(data_dir, "train"), logger)
    train_dataset = CityscapeDataset(img_dir=data_dir, img_dim=IMG_DIM, mode="train", cluster_model=cluster_model)
    valid_dataset = CityscapeDataset(img_dir=data_dir, img_dim=IMG_DIM, mode="val", cluster_model=cluster_model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    logger.info("(2) Dataset Initiated. ")

    # optimizer
    epochs = epoch_num if epoch_num > 0 else iteration_num // len(train_loader) + 1
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
    
    logger.info("(3) Model Initiated ... ")
    logger.info("Training model: {}".format(net) + ". Training Started.")

    # loss 
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(CLASS_NUM)

    # loop over epochs
    iter_count = 0
    epoch_bar = tqdm.tqdm(total=epochs, desc="Epoch", position=cur_epoch, leave=True)
    logger.info("Total epochs: {0}. Starting from epoch {1}.".format(epochs, cur_epoch+1))

    for e in range(epochs - cur_epoch):
        epoch = e + cur_epoch

        # Training.
        model.train()
        trainLossMeter = LossMeter()
        train_batch_bar = tqdm.tqdm(total=len(train_loader), desc="TrainBatch", position=0, leave=True)

        for batch_num, (orig_img, mask_img) in enumerate(train_loader):
            orig_img, mask_img = orig_img.float().to(device), mask_img.float().to(device)

            if net == "TransUNet-Base" or net == "TransUNet-Large":
                pred = model(orig_img)
            elif net == "SETR-PUP" or net == "SETR-MLA":
                if aux_layers is not None:
                    pred, _ = model(orig_img)
                else:
                    pred = model(orig_img)
            elif net == "UNet":
                pred = model(orig_img)

            loss_ce = ce_loss(pred, mask_img[:].long())
            loss_dice = dice_loss(pred, mask_img, softmax=True)
            loss = 0.5 * (loss_ce + loss_dice)

            # Backward Propagation, Update weight and metrics
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update learning rate
            for param_group in optim.param_groups:
                orig_lr = param_group['lr']
                param_group['lr'] = orig_lr * (1.0 - iter_count / iteration_num) ** 0.9
            iter_count += 1

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
            for batch_num, (orig_img, mask_img) in enumerate(valid_loader):
                orig_img, mask_img = orig_img.float().to(device), mask_img.float().to(device)

                if net == "TransUNet-Base" or net == "TransUNet-Large":
                    pred = model(orig_img)
                elif net == "SETR-PUP" or net == "SETR-MLA":
                    if aux_layers is not None:
                        pred, _ = model(orig_img)
                    else:
                        pred = model(orig_img)
                elif net == "UNet":
                    pred = model(orig_img)

                loss_ce = ce_loss(pred, mask_img[:].long())
                loss_dice = dice_loss(pred, mask_img, softmax=True)
                loss = 0.5 * (loss_ce + loss_dice)

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
        is_best = valid_loss < best_loss
        best_loss_tmp = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
            if epochs_since_improvement == early_stop_tolerance:
                break # early stopping.
        else:
            epochs_since_improvement = 0
            state = {
                'epoch': epoch,
                'loss': best_loss_tmp,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }
            torch.save(state, ckpt_src)
            logger.info("Checkpoint updated.")
            best_loss = best_loss_tmp
        epoch_bar.update(1)
    writer.close()


if __name__ == "__main__":
    train(cont=is_continue)
