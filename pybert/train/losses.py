#encoding:utf-8
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MultiLabelSoftMarginLoss
import torch
import pdb

class MultiLabelSoftMarginLoss(object):
    def __init__(self):
        self.loss_fn = MultiLabelSoftMarginLoss()
    def __call__(self, output, target):
        loss = self.loss_fn(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        loss = self.loss_fn(input = output,target = target)
        return loss
