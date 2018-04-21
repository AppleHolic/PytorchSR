import logging
import torch
from torch.autograd import Variable
from models.cbhg import CBHGNet
from models.mgru import MinimalGRUNet
from run import Runner
from trainers.timit import TIMITTrainer


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


def to_variable(tensor, is_cuda=True):
    result = Variable(tensor)
    if is_cuda:
        return result.cuda()
    else:
        return result


def get_trainer(name='cbhg'):
    if name not in Runner.IMPLEMENTED_MODELS:
        raise NotImplementedError('Trainer for %s is not implemented !! ' % name)

    if name == 'cbhg':
        return TIMITTrainer
    else:
        return None


def get_networks(name='cbhg', checkpoint_path='', is_cuda=True, is_multi_gpu=True):
    """

    :param name: the name of network
    :param checkpoint_path: checkpoint path if you want to load checkpoint
    :param is_cuda: usage of cuda
    :param is_multi_gpu: check multi gpu
    :return: network, pretrained step
    """

    if name == 'cbhg':
        network = CBHGNet()
    elif name == 'mgru':
        network = MinimalGRUNet()
    else:
        raise NotImplementedError('Network %s is not implemented !! ' % name)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(get_loadable_checkpoint(checkpoint['net']))

    if is_cuda:
        network = network.cuda()

    if is_multi_gpu:
        network = torch.nn.DataParallel(network)

    return network
