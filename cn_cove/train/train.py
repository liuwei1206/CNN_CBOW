__author__ = "liuwei"

"""
train the CNN_W2V_Model
"""
import torch
import numpy as np
import torch.nn as nn
import re
import os
import time
import logging

from ..function.util import time_to_str, str_to_time, cal_perplexities, eval_nearest_neighbors
from ..common.checks import ConfigurationError

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler() # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)


class CCTrainer(object):
    def __init__(self, model, optimizer, dataset, dev_dataset, options, gpu=True):
        """
        Args:
             model: language model
             optimizer: optimizer
             dataset: train corpus dataset
             dev_dataset: dev corpus dataset
             options: train options
             {
                total_batch:
                batch_size:
                epoch:
                lr_decay: learning_rate decay rate
                path_save_model
                save_model_every_seconds:
                num_model_to_save:
                log_dir
             }

        """
        self._model = model
        if gpu:
            self._model = self._model.cuda()
        self._optim = optimizer
        self._dataset = dataset
        self._dev_dateset = dev_dataset

        self._lr = options['lr']
        self._lr_decay = options['lr_decay']
        self._clip_grad = options['clip_grad']
        self._total_batch = options['total_batch']
        self._batch_now = 0
        self._epoch = options['epoch']
        self._epoch_now = 1
        self._batch_size = options['batch_size']

        self._path_save_model = options['path_save_model']
        self._save_model_every_seconds = options['save_model_every_seconds']
        self._num_model_to_save = options['num_model_to_save']
        self._log_dir = options['log_dir']
        self._tensorboard = options['tensorboard']
        self._vocab = options['vocab']
        self._words = options['words']

        self._old_models_path = []

        if self._clip_grad <= 0:
            self._clip_grad = None

    def save_model(self, epoch_time):
        """
        Args:
            epoch_time: epoch and time str. eg: 1.2018-02-23_18:30:20
        """
        if self._path_save_model is not None:
            model_path = os.path.join(self._path_save_model, "model_state_epoch_{}.th".format(epoch_time))
            model_state = self._model.state_dict()
            torch.save(model_state, model_path)

            train_path = os.path.join(self._path_save_model, "training_state_epoch_{}.th".format(epoch_time))
            epoch = epoch_time.split('.')[0]
            training_state = {
                'epoch': epoch,
                'optimizer': self._optim.state_dict()
            }
            torch.save(training_state, train_path)

            # if need to delete old model
            if self._num_model_to_save is not None and self._num_model_to_save > 0:
                self._old_models_path.append([model_path, train_path])
                if len(self._old_models_path) > self._num_model_to_save:
                    # print(len(self._old_models_path), "++++++++++++++++++++++++++")
                    paths_to_remove = self._old_models_path.pop(0)
                    # print(paths_to_remove)

                    for fname in paths_to_remove:
                        os.remove(fname)

    def find_last_model(self):
        """
        find the path of latest saved model
        """
        if self._path_save_model is None:
            return None

        saved_models_path = os.listdir(self._path_save_model)
        if len(saved_models_path) == 0:
            return None
        saved_models_path = [x for x in saved_models_path if 'model_state_epoch' in x]
        print(saved_models_path)

        found_epochs = [
            re.search("model_state_epoch_([0-9\.\-\_\:]+).th", x).group(1)
            for x in saved_models_path
        ]
        int_epochs = [[int(pieces.split('.')[0]), pieces.split('.')[1]] for pieces in found_epochs]
        last_epochs = sorted(int_epochs, reverse=True)[0]
        epoch_to_load = '{}.{}'.format(last_epochs[0], last_epochs[1])

        model_path = os.path.join(self._path_save_model, "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._path_save_model, "training_state_epoch_{}.th".format(epoch_to_load))

        return model_path, training_state_path

    def restore_model(self):
        """
        restore a model from the latest checkpoint file
        """
        lastest_checkpoint = self.find_last_model()

        if lastest_checkpoint is None:
            return 1
        else:
            model_path, training_state_path = lastest_checkpoint

            model_state = torch.load(model_path)
            training_state = torch.load(training_state_path)
            self._model.load_state_dict(model_state)
            self._optim.load_state_dict(training_state['optimizer'])

            epoch = training_state['epoch']
            return int(epoch)

    def decay_lr(self):
        """
        decrease the learning rate as the model train
        """
        lr = self._lr * (1 - (self._batch_now * 1.0 / self._total_batch))
        if lr <= 0.00005:
            lr = 0.00005
        logger.info("batch: %d/%d, train rate: %f%%, lr: %f",
                    self._batch_now, self._total_batch, self._batch_now * 100.0 / self._total_batch,
                    lr)
        for param_group in self._optim.param_groups:
            param_group['lr'] = lr


    def evaludate(self, epoch):
        """
        use dev dataset evaluate the model
        no grad, set model.eval()

        Args:
            epoch: the number of this epoch
        """
        # empty train cache
        torch.cuda.empty_cache()

        validation_loss = 0.0
        validation_accu = 0.0

        batch_num = 0
        self._model.eval()

        for batch_data in self._dev_dateset.get_batch():
            batch_num += 1

            loss, accu = self._model(batch_data, True)
            del batch_data

            validation_loss += loss.item()
            validation_accu += accu.item()

        validation_loss = validation_loss / batch_num
        validation_accu = validation_accu / batch_num

        # cal_perplexities(validation_loss),
        self._tensorboard.add_validation_scalar("dev_perplexities",
                                                validation_loss,
                                                epoch)
        self._tensorboard.add_validation_scalar("dev_accu", validation_accu * 100, epoch)
        #logger.info("validation_loss: %f, validation_accu: %f%%", cal_perplexities(validation_loss), validation_accu * 100.0)
        logger.info("validation_loss: %f, validation_accu: %f%%", validation_loss, validation_accu * 100.0)

        eval_nearest_neighbors(self._vocab, self._model._embed, self._words)
        torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        """
        train one epoch
        Args:
            epoch: the number of now epoch
        """
        logger.info("Train Epoch %d/%d", epoch, self._epoch)

        train_loss = 0.0
        train_accu = 0.0

        batch_this_epoch = 0

        last_save_time = time.time()
        self._model.train()

        for batch_data in self._dataset.get_batch():
            batch_this_epoch += 1
            self._batch_now += 1
            batch_now_total = self._batch_now

            self._optim.zero_grad()

            loss, accu = self._model(batch_data, False)
            del batch_data

            loss.backward()
            # set clip_grad
            if self._clip_grad is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
            self._optim.step()

            train_loss += loss.item()
            train_accu += accu.item()

            del loss
            del accu
            if batch_now_total % 100 == 0:
                torch.cuda.empty_cache()

            if batch_now_total % 1000 == 0:
                self.decay_lr()
                #torch.cuda.empty_cache()

            # send loss and accu to tensorboard, and the total norm
            self._tensorboard.add_train_scalar("train_perplexities",
                                               train_loss / batch_this_epoch,
                                               self._batch_now)
            self._tensorboard.add_train_scalar("train_accu", train_accu * 100 / batch_this_epoch,
                                               self._batch_now)
            self._tensorboard.add_train_scalar('train_total_norm', total_norm, self._batch_now)

         
            # save the model
            if self._save_model_every_seconds is not None and (
                time.time() - last_save_time > self._save_model_every_seconds
            ):
                last_save_time = time.time()
                self.save_model('{}.{}'.format(epoch, time_to_str(last_save_time)))

        # validation
        self.evaludate(epoch)

    def train(self):
        """
        train the model, call the train_epoch
        """
        try:
            start_epoch = self.restore_model()
            self._batch_now = self._total_batch * (start_epoch - 1) * 1.0 / self._epoch
        except:
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        logger.info("Beginning trainning!")

        training_start_time = time.time()
        for epoch in range(start_epoch, self._epoch + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            epoch_end_time = time.time()
            logger.info("#################################################")
            logger.info('train time of epoch %d: %ds', epoch, (epoch_end_time - epoch_start_time))
            logger.info("#################################################")

            self.save_model('{}.{}'.format(epoch, time_to_str(int(epoch_end_time))))

        training_end_time = time.time()
        logger.info('train over, time: %ds', (training_end_time - training_start_time))

