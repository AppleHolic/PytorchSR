import abc
import torch.nn as nn


class Model(nn.Module):
    """
    Bsae Class to build SR model
    """
    def __init__(self):
        super(Model, self).__init__()

    @abc.abstractmethod
    def loss(*args, **kwargs):
        pass

    @abc.abstractmethod
    def data_loader(*args, **kwargs):
        pass
