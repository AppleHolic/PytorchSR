import abc
import heapq

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.trainer import trainer


class CustomTrainer(trainer.Trainer):
    """
    Customizing Trainer
    link : https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py
    """
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
        super(CustomTrainer, self).__init__(model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train()
            self.call_plugins('epoch', i)

    @abc.abstractmethod
    def train(self):
        pass

    @staticmethod
    def to_variable(*args, is_cuda=True):
        result = [Variable(item, requires_grad=False) for item in args]
        if is_cuda:
            result = [item.cuda() for item in result]
        return result


class CBHGTrainer(CustomTrainer):
    """
    Custom Trainer for CBHG based phoneme classification model
    """

    def __init__(self, model, dataset, optimizer, is_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = model.loss
        self.dataset = dataset
        self.is_cuda = is_cuda
        super(CBHGTrainer, self).__init__(self.model, self.criterion, self.optimizer, self.dataset)

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            # batch_input, batch_target = data
            mfcc, phn = self.to_variable(*data, is_cuda=self.is_cuda)
            self.call_plugins('batch', i, mfcc, phn)

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(mfcc)
                loss = self.criterion(mfcc, batch_output, phn)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, mfcc, phn,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i