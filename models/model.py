import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.data_utils import PHNS
from settings.hparam import hparam as hp
from data.data_loader import TrainVoiceDataset, TestVoiceDataset


class Model(nn.Module):
    """
    Bsae Class to build SR model
    """
    def __init__(self):
        super(Model, self).__init__()

    @abc.abstractmethod
    def loss(*args, **kwargs):
        pass

    @staticmethod
    def data_loader(mode):
        if mode == 'train':
            dataset = TrainVoiceDataset()
        elif mode == 'test':
            dataset = TestVoiceDataset()
        else:
            raise NotImplementedError('%s mode is not implemented ! ' % mode)

        data_loader = DataLoader(dataset, batch_size=hp.train.batch_size,
                                 shuffle=(mode == 'train'), num_workers=hp.num_workers)
        return data_loader


    @staticmethod
    def calc_output(net):
        ppgs = F.softmax(net / hp.train.t, dim=-1)
        _, preds_ppg = torch.max(net, dim=-1)  # return arg_max
        return ppgs, preds_ppg

    @staticmethod
    def loss(mfcc, logits_ppg, y_ppg):
        is_target = torch.sign(torch.abs(torch.sum(mfcc, dim=-1)))  # indicator: (N, T)
        # flatten
        logits_ppg = logits_ppg.view(-1, len(PHNS))
        y_ppg = y_ppg.view(-1)
        loss = F.cross_entropy(logits_ppg, y_ppg, reduce=False)
        loss = loss.view(is_target.size()[0], -1)
        loss *= is_target
        return torch.mean(loss)

    @staticmethod
    def accuracy(pred_ppg, y_ppg):
        target = torch.sign(torch.abs(y_ppg))
        target = target.data.cpu().numpy()
        num_hits = torch.eq(pred_ppg, y_ppg).data.cpu().numpy()
        num_hits = np.sum(num_hits * target)
        num_targets = np.sum(target)
        return float(num_hits / num_targets), num_hits, num_targets

    @staticmethod
    def topk_accuracy(logit, y, topk=3):
        """
        calculate topk accuracy
        :param logit: shape (N, Time Steps, The Number of Phonemes)
        :param y: shape (N, Time Steps)
        :return: topk accuracy only
        """
        _, topk_var = logit.topk(topk, dim=-1)
        topk_arr = topk_var.data.cpu().numpy()
        y_arr = y.data.cpu().numpy()

        target = torch.sign(torch.abs(y))
        target = target.data.cpu().numpy()
        cor = 0.
        numb = np.sum(target)

        for b in range(y_arr.shape[0]):  # batch size
            for j, (pred, label) in enumerate(zip(topk_arr[b], y_arr[b])):
                if not target[b, j]:
                    continue
                if label in pred:
                    cor += 1
        return cor / numb, cor, numb

    @staticmethod
    def confusion_matrix(logit_ppg, y_ppg):
        """
        calculate and get confusion matrix as numpy array
        :param pred_ppg: shape (N, Time Steps, The Number of Phonemes)
        :param y_ppg: shape(N, Time Steps)
        :return: numpy array (NP, NP)
        """
        _, preds_ppg = torch.max(logit_ppg, dim=-1)
        pred_arr = preds_ppg.data.cpu().numpy()
        y_ppg = y_ppg.data.cpu().numpy()
        n = len(PHNS)
        confusion_matrix = np.zeros((n, n))
        for b in range(y_ppg.shape[0]):  # batch size
            for pred, label in zip(pred_arr[b], y_ppg[b]):
                confusion_matrix[pred, label] += 1
        return confusion_matrix
