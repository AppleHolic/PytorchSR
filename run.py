import fire
from torch.utils.data import DataLoader

import utils
import torch.utils.trainer.plugins as plugins
from torch import optim
from models.cbhg import CBHGNet
from models.trainer import CBHGTrainer
from settings.hparam import hparam as hp


class Runner:

    def train(self, model, data_split=-1.0, is_cuda=True):
        mode = 'train'
        logger = utils.get_logger(mode)
        # initialize hyperparameters
        hp.set_hparam_yaml(mode)
        logger.info('Setup mode as %s, model : %s' % (mode, model))

        # get network and trainer
        net, custom_trainer = Runner.get_net_and_trainer(model, is_cuda=is_cuda)

        # setup dataset
        dataset = net.data_loader(mode='train', split=data_split)
        dataset = DataLoader(dataset, batch_size=hp.train.batch_size, shuffle=True, num_workers=16)

        # setup optimizer
        if mode == 'train':
            parameters = net.parameters()
            logger.info(net)
        lr = getattr(hp, mode).lr
        optimizer = optim.Adam(parameters, lr=lr)

        # pass model, loss, optimizer and dataset to the trainer
        t = custom_trainer(net, dataset, optimizer, is_cuda=is_cuda)

        # register some monitoring plugins
        t.register_plugin(plugins.ProgressMonitor())
        t.register_plugin(utils.CBHGAccuracyMonitor())
        t.register_plugin(plugins.LossMonitor())
        t.register_plugin(plugins.TimeMonitor())
        t.register_plugin(plugins.Logger(['progress', 'accuracy', 'loss', 'time']))

        # train!
        t.run(hp.train.num_epochs)

    def test(self):
        raise NotImplementedError('Test mode is not implemented!')

    def eval(self):
        raise NotImplementedError('Evaluation mode is not implemented!')

    @staticmethod
    def get_net_and_trainer(model, is_cuda=True):
        if model == 'cbhg':
            net, trainer = CBHGNet(), CBHGTrainer
            if is_cuda:
                net = net.cuda()
            return net, trainer
        else:
            raise NotImplementedError('The model %s is not implemented yet!' % model)


if __name__ == '__main__':
    fire.Fire(Runner)
