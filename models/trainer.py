import abc

from torch.utils.trainer import trainer


class CustomTrainer(trainer.Trainer):
    """
    Customizing Trainer
    link : https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py
    """
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
        super(CustomTrainer, self).__init__(model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

    @abc.abstractmethod
    def train(self):
        pass


class CBHGTrainer(CustomTrainer):
    """
    Custom Trainer for CBHG based phoneme classification model
    """

    def __init__(self, model, optimizer, data_split=-1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = model.loss
        self.dataset = model.data_loader(mode='train', split=data_split)

    def train(self):
        pass
