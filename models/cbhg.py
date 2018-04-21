import torch.nn as nn

from data.data_utils import PHNS
from models.model import Model
from models.modules import Prenet, CBHG, SeqLinear
from settings.hparam import hparam as hp


class CBHGNet(Model):

    def __init__(self):
        super().__init__()
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
