import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.data_utils import PHNS
from data.data_loader import VoiceDataLoader
from models.model import Model
from models.modules import Prenet, CBHG, SeqLinear
from settings.hparam import hparam as hp


class CBHGNet(Model):

    def __init__(self):
        super(CBHGNet, self).__init__()
        self.prenet = Prenet(hp.default.n_mfcc, hp.train.hidden_units,
                             hp.train.hidden_units // 2, dropout_rate=hp.train.dropout_rate)
        self.cbhg = CBHG(
            K=hp.train.num_banks,
            hidden_size=hp.train.hidden_units // 2,
            num_highway_blocks=hp.train.num_highway_blocks,
            num_gru_layers=1
        )
        self.output = SeqLinear(hp.train.hidden_units, len(PHNS))
        self.ppg_output = nn.Softmax()

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        net = self.prenet(x)
        net, _ = self.cbhg(net)
        logits_ppg = self.output(net)
        return logits_ppg

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
    def data_loader(mode, split=-1.0):
        return VoiceDataLoader(mode=mode, data_split=split)
