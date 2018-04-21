import time
import torch
import utils
import os
import numpy as np
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from collections import defaultdict
from settings.hparam import hparam as hp
from data.data_utils import PHNS
from trainers.trainer import Trainer, trainer_logger, ModelInferencer


class TIMITTrainer(Trainer):
    """
    Customizing Trainer for phoneme classification
    """
    def __init__(self, model, optimizer, train_dataset, test_dataset, is_cuda=True, logdir='', savedir='', topk=3):
        super().__init__(model, optimizer, train_dataset, test_dataset, is_cuda=is_cuda)
        self.logdir = logdir
        self.savedir = savedir
        self.is_cuda = is_cuda
        self.topk = topk
        self.best_status = defaultdict(float)
        if self.logdir:
            self.writer = SummaryWriter(log_dir=logdir)

    def train(self, epoch):
        trainer_logger.info('------------- TRAIN Epoch : %d -------------' % epoch)
        nb_batch = len(self.train_dataset)
        self.status['train'] = defaultdict(float)

        for i, ds in enumerate(self.train_dataset):
            current_step = epoch * nb_batch + i + 1
            loss, cor, nb = TIMITModelInferencer.train(
                epoch, i, len(self.train_dataset), self.model, self.optimizer, ds, current_step, self.is_cuda
            )
            self.status['train']['loss'] += loss
            self.status['train']['cor'] += cor
            self.status['train']['nb'] += nb

        # calc train metrics
        self.status['train']['loss'] /= len(self.train_dataset)
        self.status['train']['acc'] = float(self.status['train']['cor'] / self.status['train']['nb'])

        # train log
        self.__train_log(epoch)

    def test(self, epoch):
        trainer_logger.info('------------- TEST Epoch : %d -------------' % epoch)
        nb_batch = len(self.test_dataset)
        self.status['test'] = defaultdict(float)
        self.status['test']['confusion_mat'] = np.zeros((len(PHNS), len(PHNS)), dtype=np.int32)

        for i, ds in enumerate(self.test_dataset):
            loss, cor, nb, topk_cor, topk_nb, confusion_mat = TIMITModelInferencer.test(
                i, nb_batch, self.model, ds, is_cuda=self.is_cuda, topk=self.topk, is_confusion_mat=True
            )
            self.status['test']['loss'] += loss
            self.status['test']['cor'] += cor
            self.status['test']['nb'] += nb
            self.status['test']['topk_cor'] += topk_cor
            self.status['test']['topk_nb'] += topk_nb
            self.status['test']['confusion_mat'] += confusion_mat.astype(np.int32)

        # calc entire test metrics
        self.status['test']['loss'] /= len(self.test_dataset)
        self.status['test']['acc'] = float(self.status['test']['cor'] / self.status['test']['nb'])
        self.status['test']['topk_acc'] = float(topk_cor / topk_nb)

        # test log
        self.__test_log(epoch)

    def __train_log(self, epoch):
        metric_keys = ['cor', 'nb', 'acc', 'loss']

        cor, nb, acc, loss = [self.status['train'][key] for key in metric_keys]

        msg = 'Epoch %d / Total Train Loss : %.6f / Cor %d/%d / Accuracy %.6f' % (epoch, loss, cor, nb, acc)
        trainer_logger.info(msg)

        if self.logdir:
            self.writer.add_scalar('train/loss', loss, global_step=epoch)
            self.writer.add_scalar('train/accuracy', acc, global_step=epoch)

    def __test_log(self, epoch):
        metric_keys = ['cor', 'nb', 'acc', 'topk_cor', 'topk_nb', 'topk_acc', 'loss']

        cor, nb, acc, topk_cor, topk_nb, topk_acc, test_loss = [self.status['test'][key] for key in metric_keys]

        # make test log message
        msg = 'Epoch %d / Total Test Loss : %.6f' % (epoch, test_loss)
        msg = msg + ' / Cor %d/%d / Accuracy %.6f' % (cor, nb, acc)
        msg = msg + ' / Top 3 Accuracy %.4f' % topk_acc
        trainer_logger.info(msg)

        if self.logdir:
            self.writer.add_scalar('test/loss', test_loss, global_step=epoch)
            self.writer.add_scalar('test/accuracy', acc, global_step=epoch)
            self.writer.add_scalar('test/top3_accuracy', topk_acc, global_step=epoch)

    def do_end_of_epoch(self, epoch):
        # save best
        if 'test_acc' not in self.best_status:
            self.best_status['test_acc'] = 0.0

        if self.best_status['test_acc'] < self.status['test']['acc'] and self.savedir:
            # check and make directory
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)

            self.best_status = deepcopy(self.status)
            # save
            state_dict = {
                'optimizer': self.optimizer.state_dict(),
                'pretrained_step': epoch,
                'status': self.best_status
            }

            save_path = '%s/checkpoint_%d_loss_%.6f.pth.tar' % (self.savedir, epoch, self.best_status['test']['loss'])
            state_dict['net'] = self.model.state_dict()
            save_path = save_path.replace('.pth', 'acc_%.6f_topkacc_%.6f.pth' % (
                self.best_status['test']['acc'], self.best_status['test']['topk_acc']))
            torch.save(state_dict, save_path)

            trainer_logger.info('epoch %d ] save model at step %d ...' % (epoch, epoch))

    def finalize(self, epoch):
        pass


class TIMITModelInferencer(ModelInferencer):

    @staticmethod
    def train(epoch, batch_idx, nb_batch, network, optimizer, ds, current_step, is_cuda=True):

        start_time = time.time()
        if type(network) is DataParallel:
            module = network.module
        else:
            module = network

        optimizer.zero_grad()

        # setup data
        mfccs, phns = list(map(lambda x: utils.to_variable(x, is_cuda), ds))

        # Forward
        logits_ppg = network(mfccs)
        ppgs, preds_ppg = module.calc_output(logits_ppg)

        loss = module.loss(mfccs, logits_ppg, phns)

        accuracy, cor, nb = module.accuracy(preds_ppg, phns)

        loss.backward()

        optimizer.step()

        # time dist
        time_per_step = time.time() - start_time

        if batch_idx % hp.log_step == 0:
            msg = '%d epoch / %d/%d batch / %d iteration / Elapsed: %.2f sec / total loss: %.6f / Accuracy %.6f' \
                  % (epoch, batch_idx, nb_batch, current_step, time_per_step, loss.data[0], accuracy)
            trainer_logger.info(msg)

        return loss.data.cpu().numpy(), cor, nb

    @staticmethod
    def test(batch_idx, nb_batch, network, ds, topk=0, is_confusion_mat=False, is_cuda=True):
        start_time = time.time()
        if type(network) is DataParallel:
            module = network.module
        else:
            module = network

        # setup data
        mfccs, phns = list(map(lambda x: utils.to_variable(x, is_cuda), ds))

        # Forward
        logits_ppg = network(mfccs)
        ppgs, preds_ppg = module.calc_output(logits_ppg)
        loss = module.loss(mfccs, logits_ppg, phns)
        accuracy, cor, nb = module.accuracy(preds_ppg, phns)

        topk_cor, topk_nb = 0, 0
        confusion_mat = None

        if topk > 0:
            topk_acc, topk_cor, topk_nb = module.topk_accuracy(logits_ppg, phns)
        if is_confusion_mat:
            confusion_mat = module.confusion_matrix(logits_ppg, phns)

        if batch_idx % hp.log_step == 0:
            msg = '%d/%d batch / Elapsed: %.2f sec / Total loss: %.6f / Accuracy %.6f' \
                  % (batch_idx, nb_batch, time.time() - start_time, loss.data[0], accuracy)
            if topk > 0:
                msg = msg + ' / Top %d accuracy %.6f  ' % (topk, topk_acc)
            trainer_logger.info(msg)

        return loss.data.cpu().numpy(), cor, nb, topk_cor, topk_nb, confusion_mat
