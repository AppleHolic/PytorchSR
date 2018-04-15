import abc

from torch.utils.trainer import trainer


class CustomTrainer(trainer.Trainer):
    """
    Customizing Trainer
    link : https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py
    """

    @abc.abstractmethod
    def train(self):
        pass


class CBHGTrainer(CustomTrainer):
    """
    Custom Trainer for CBHG based phoneme classification model
    """

    def __init__(self):
        pass

    def train(self):
        pass
