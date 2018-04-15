import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import settings.hparam as hp


class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """
    def __init__(self, input_size, output_size, time_dim=1):
        """
        :param input_size: dimension of input
        :param output_size: dimension of output
        :param time_dim: index of time dimension
        """
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        """
        :param input_: sequences
        :return: outputs
        """
        batch_size = input_.size()[0]
        if self.time_dim == 2:
            input_ = input_.transpose(1, 2)
        input_ = input_.contiguous()
        input_ = input_.view(-1, self.input_size)

        out = self.linear(input_).view(batch_size, -1, self.output_size)

        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)

        return out


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, time_dim=2):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', SeqLinear(self.input_size, self.hidden_size, time_dim=time_dim)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(dropout_rate)),
             ('fc2', SeqLinear(self.hidden_size, self.output_size, time_dim=time_dim)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(dropout_rate)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out


class CBHG(nn.Module):
    """
    CBHG Module
    """
    def __init__(self, hidden_size, K=16, projection_size=256, num_highway_blocks=4, num_gru_layers=1, max_pool_kernel_size=2):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K

        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                         out_channels=hidden_size * 2,
                                         kernel_size=3,
                                         padding=int(np.floor(3/2)))

        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                                           out_channels=hidden_size,
                                           kernel_size=3,
                                           padding=int(np.floor(3/2)))

        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)

        self.batchnorm_proj_2 = nn.BatchNorm1d(hidden_size)

        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)

        self.highway = Highwaynet(self.hidden_size, num_layers=num_highway_blocks)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):

        input_ = input_.contiguous()
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = batchnorm(F.relu(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]

        # Projection
        style_feature = self.batchnorm_proj_1(F.relu(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_proj = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(style_feature))) + input_

        # Highway networks
        highway = self.highway.forward(conv_proj)
        highway = torch.transpose(highway, 1, 2)

        # Bidirectional GRU
        if torch.cuda.is_available():
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size)).cuda()
        else:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size))

        self.gru.flatten_parameters()

        content_feature, _ = self.gru(highway, init_gru)

        return content_feature, style_feature


class Highwaynet(nn.Module):
    """
    Highway network
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units, time_dim=2))
            self.gates.append(SeqLinear(num_units, num_units, time_dim=2))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c

        return out


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, norm_fn=None, dropout=False, activation_fn=F.relu):
        super(Conv1d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding
                                     )
        self.norm = norm_fn
        self.dropout = dropout
        self.activation_fn = activation_fn
        if norm_fn is not None:
            self.norm_fn = norm_fn(hp.hidden_size)

        if dropout:
            self.drop = nn.Dropout(p=0.25)

    def forward(self, input_):
        conv = self.activation_fn(F.conv1d(input_, self.weight, self.bias, self.stride,
                 self.padding, self.dilation, self.groups))

        if self.norm is not None:
            conv = self.norm_fn(conv)

        if self.dropout:
            conv = self.drop(conv)

        return conv