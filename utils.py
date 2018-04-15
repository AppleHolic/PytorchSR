import logging

from torch.autograd import Variable
from torch.utils.trainer.plugins.monitor import Monitor

from models.cbhg import CBHGNet


def get_logger(name):
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_loadable_checkpoint(checkpoint):
    """
    If model is saved with DataParallel, checkpoint keys is started with 'module.' remove it and return new state dict
    :param checkpoint:
    :return: new checkpoint
    """
    new_checkpoint = {}
    for key, val in checkpoint.items():
        new_key = key.replace('module.', '')
        new_checkpoint[new_key] = val
    return new_checkpoint


class CBHGAccuracyMonitor(Monitor):
    stat_name = 'accuracy'

    def __init__(self, *args, is_cuda=True, **kwargs):
        kwargs.setdefault('unit', '%')
        kwargs.setdefault('precision', 2)
        self.is_cuda = is_cuda
        super(CBHGAccuracyMonitor, self).__init__(*args, **kwargs)

    def _get_value(self, iteration, input, target, output, loss):
        mfcc, phn = input, target
        output = Variable(output)
        if self.is_cuda:
            output = output.cuda()
        ppgs, preds_ppg = CBHGNet.calc_output(output)
        accuracy, cor, nb = CBHGNet.accuracy(preds_ppg, phn)
        return accuracy