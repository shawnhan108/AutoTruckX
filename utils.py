import logging

import torch 
from config import device

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
