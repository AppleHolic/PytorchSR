import abc
import logging


# setup logger / cannot import from utils
trainer_logger = logging.getLogger('trainer')
trainer_logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
trainer_logger.addHandler(ch)


class Trainer:
    """
    Super class of trainer classes. Trainer should build up with extend this class
    """

    def __init__(self, model, optimizer, train_dataset, test_dataset, is_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.is_cuda = is_cuda
        self.status = {}

    def run(self, epochs=1):
        """
        Template Method to describe training
        :param epochs: the number of epochs
        :return: None
        """
        try:
            for i in range(epochs):
                self.model.train()
                self.train(i)
                self.model.eval()
                self.test(i)
                self.do_end_of_epoch(i)
        except KeyboardInterrupt:
            trainer_logger.info('Train is canceled !!')
        finally:
            self.finalize(i)

    @abc.abstractmethod
    def train(self, epoch):
        raise NotImplementedError()

    @abc.abstractmethod
    def test(self, epoch):
        raise NotImplementedError()

    @abc.abstractmethod
    def do_end_of_epoch(self, epoch):
        raise NotImplementedError()

    @abc.abstractmethod
    def finalize(self, epoch):
        raise NotImplementedError()


class ModelInferencer:

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError()
