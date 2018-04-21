import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import settings.hparam as hp
from torch.autograd import Variable
from collections import OrderedDict


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


class MinimalGRU(nn.Module):
    """
    Implementation Revising GRU
    Reference : https://arxiv.org/abs/1710.00641
    Reference Source : https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
    Differences with original GRU
    1. No reset gate
    """

    def __init__(self, input_size, hidden_size, num_layers=1, is_bidirection=False,
                 bias=True, dropout=0, nonlinearity='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_directions = 2 if is_bidirection else 1
        self.bias = bias
        # setup nonlinearity
        if nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError('%s nonlinearity is not implemented !!' % nonlinearity)

        gate_size = 2 * hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

                w_ih = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = nn.Parameter(torch.Tensor(gate_size))
                b_hh = nn.Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self.reset_parameters()

    def reset_parameters(self):
        """
        https://github.com/pytorch/pytorch/blob/7b6b7d4575832a9af4257ba341f3da9e7a2a2b57/torch/nn/modules/rnn.py#L115
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx, time_dim=1):
        """
        Custom GRU Layers with gru cell in source
        :param x: sequence data
        :param hx: initial hidden status
        :param time_dim: the time axis on input data (default is 1)
        :return:
        """
        assert time_dim == 1

        t_len = x.size()[time_dim]

        for layer in range(self.num_layers):

            layer_outputs = []

            for direction in range(self.num_directions):

                # get attrs
                weight_attr_names = self._all_weights[layer * self.num_directions + direction]
                attrs = [getattr(self, name) for name in weight_attr_names]
                if self.bias:
                    w_ih, w_hh, b_ih, b_hh = attrs
                else:
                    w_ih, w_hh, b_ih, b_hh = attrs + [None, None]

                hx_outputs = []

                hx_ = hx[layer * self.num_directions + direction]

                # access on sequence
                for t in range(t_len):

                    input = x[:, t, :]
                    # GRU Cell Part
                    # make gates
                    gates = F.linear(input, w_ih, b_ih) + F.linear(hx_, w_hh, b_hh)
                    ug, og = gates.chunk(2, 1)

                    # calc
                    ug = F.sigmoid(ug)
                    og = self.act(og)
                    hx_ = ug * hx_ + (1 - ug) * og

                    hx_outputs.append(hx_)

                layer_outputs.append(hx_outputs)

            assert len(layer_outputs) in [1, 2]

            # make output or next input
            # bi-direction
            if len(layer_outputs) == 2:
                x = []
                for f, b in zip(layer_outputs[0], layer_outputs[1][::-1]):
                    x.append(torch.cat([f, b], dim=1).unsqueeze_(1))
                x = torch.cat(x, dim=1)
            # single direction
            else:
                x = torch.cat([item.unsqueeze_(1) for item in layer_outputs[0]], 1)

        return x
