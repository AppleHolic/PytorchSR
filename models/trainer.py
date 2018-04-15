import abc

from torch.autograd import Variable
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

    @staticmethod
    def to_variable(*args, is_cuda=True):
        result = [Variable(item) for item in args]
        if is_cuda:
            result = [item.cuda() for item in result]
        return result


class CBHGTrainer(CustomTrainer):
    """
    Custom Trainer for CBHG based phoneme classification model
    """

    def __init__(self, model, optimizer, data_split=-1.0, is_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = model.loss
        self.dataset = model.data_loader(mode='train', split=data_split)
        self.is_cuda = is_cuda

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            # batch_input, batch_target = data
            wav, mfcc, phn = self.to_variable(data, self.is_cuda)
            self.call_plugins('batch', i, wav, phn)

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(wav)
                loss = self.criterion(mfcc, batch_output, phn)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, wav, phn,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i