__author__ = "liuwei"

import time
import os
import argparse
import torch
import numpy as np
from tensorboardX import SummaryWriter

#from cn_cove.data.genetor_sents import create_sents
from cn_cove.data.vocab import make_vocab_file, Vocabulary, get_vocab_size, get_data_rows, words_from_vocab, get_batch_num_by_words
from cn_cove.data.dataset import LM_Dataset
from cn_cove.data.dev_dataset import Validation_Dataset
from cn_cove.data.train_dataset import Train_Dataset
from cn_cove.module.feature import WordFeature, BiasFeature
from cn_cove.module.cnn_lm import CNN_LM
from cn_cove.function.f_neg_loss import get_neg_samples, get_neg_table
from cn_cove.function.dump_weight import dump_weights, load_weights, dump_embedding
from cn_cove.module.loss import Neg_Loss, Full_Loss
from cn_cove.model.cnn_w2v import CNN_W2V_Model
from cn_cove.module.tbx_writer import TensorboardWriter
from cn_cove.train.train import CCTrainer

## Corpus
parser = argparse.ArgumentParser(description='params of the model')
parser.add_argument('--raw_corpus_dir', help='the dir of raw corpus', default='data/raw_corpus')
parser.add_argument('--new_corpus_dir', help='the dir of corpus', default='data/corpus')
parser.add_argument('--dev_corpus_dir', help='the dir of validation corpus', default='data/dev')
parser.add_argument('--min_len', help='the min len of sent to resolve', default=10)

parser.add_argument('--min_cnt', help='the min frequence of the word', default=50)
parser.add_argument('--vocab_file', help='the path of vocab file', default='data/vocab.txt')
parser.add_argument('--batch_size', help='the batch size', default=32)
parser.add_argument('--vocab_size', help='the size of vocab', default=0)

##
parser.add_argument('--n_tokens', help="n_tokens word each row", default=120)
parser.add_argument('--word_dim', help='the size of word embedding', default=300)
parser.add_argument('--cnn_windows', help='the train cnn model windows', default=10)
parser.add_argument('--n_cnns', help='the num layer of cnn', default=1)
parser.add_argument('--cnn_dropout', help='the dropout rate of cnn model', default=0.5)
parser.add_argument('--n_highway', help='the number of highway layers', default=0)

parser.add_argument('--epoch', help='train epochs', default=20)
parser.add_argument('--lr', help='the learning rate', default=0.01)
parser.add_argument('--lr_decay', help='the decay rate of lr', default=0.1)
parser.add_argument('--weight-decay', help="do L2 norm for weights", default=1e-8)
parser.add_argument('--clip_norm', help="the max norm of grad", default=5)
parser.add_argument('--neg_size', help='the number of neg samples', default=1200)
parser.add_argument('--dev_batch_num', help='how many batch use to validation', default=10)
parser.add_argument('--total_batch', help='the total train batch, batch_one_epoch * epoch', default=1)
parser.add_argument('--path_save_model', help='the dir to save models', default='data/models')
parser.add_argument('--save_model_every_seconds', help='save model every num seconds', default=1800)
parser.add_argument('--num_models_to_save', help='save at most num models', default=20)
parser.add_argument('-log_dir', help='path to save log', default='data/log')


ops = parser.parse_args()
if __name__ == "__main__":
    # prepro of the raw corpus
    #create_sents(ops.raw_corpus_dir, ops.new_corpus_dir, ops.min_len)

    # create vocab file
    make_vocab_file(ops.new_corpus_dir, ops.min_cnt)


    torch.backends.cudnn.benchmark = True
    gpu = False
    if torch.cuda.is_available():
        gpu = True

    vocab_size = get_vocab_size(ops.vocab_file)
    # all_rows = get_data_rows(ops.new_corpus_dir)
    batch_num_per_epoch = get_batch_num_by_words(ops.new_corpus_dir, ops.batch_size * ops.n_tokens)
    batch_size = ops.batch_size
    ops.vocab_size = vocab_size
    # ops.total_batch = int(all_rows / batch_size) * ops.epoch
    ops.total_batch = batch_num_per_epoch * ops.epoch

    # data
    vocab = Vocabulary(ops.vocab_file)
    # dataset = LM_Dataset(ops.new_corpus_dir, batch_size, vocab)
    # dev_dataset = Validation_Dataset(ops.dev_corpus_dir, batch_size, vocab)
    dataset = Train_Dataset(ops.new_corpus_dir, batch_size, ops.n_tokens,
                            ops.cnn_windows, vocab, gpu)
    dev_dataset = Train_Dataset(ops.dev_corpus_dir, batch_size, ops.n_tokens,
                                ops.cnn_windows, vocab, gpu, ops.dev_batch_num)

    # embedding
    softmax_weight = WordFeature(ops.vocab_size, ops.word_dim, seed=1337)
    softmax_bias = BiasFeature(ops.vocab_size)

    word_embedding = WordFeature(ops.vocab_size, ops.word_dim, seed=1206)

    # cnn_lm
    cnn_lm = CNN_LM(ops.word_dim, ops.word_dim, ops.n_cnns, ops.cnn_windows,
                    ops.cnn_dropout, ops.n_highway, gpu=gpu)

    # loss, negative loss and full loss
    loss_f = torch.nn.CrossEntropyLoss()
    neg_table = get_neg_table(ops.vocab_size)
    loss = Neg_Loss(loss_f, ops.neg_size, ops.vocab_size, neg_table,
                    softmax_weight, softmax_bias, gpu=gpu)
    full_loss = Full_Loss(loss_f, softmax_weight, softmax_bias, gpu=gpu)

    # model
    model = CNN_W2V_Model(word_embedding, cnn_lm, loss, full_loss, gpu=gpu)
    # print(type(model).__name__)

    train_log = SummaryWriter(os.path.join(ops.log_dir, "train"))
    validation_log = SummaryWriter(os.path.join(ops.log_dir, "validation"))
    tensorboard = TensorboardWriter(train_log, validation_log)

    params = model.state_dict()
    # print('learning params:')
    # for k, v in params.items():
    #     print(k)
    words = ['中国', '秦始皇', '股票', '狗', '跳舞', '数学', '书法', '一', '奥巴马', '黄河', '北京', '和谐', '主席', '泰山', '巴黎']

    # train options
    train_options = {
        'total_batch': ops.total_batch,
        'batch_size': ops.batch_size,
        'epoch': ops.epoch,
        'lr_decay': ops.lr_decay,
        'lr': ops.lr,
        'clip_grad': ops.clip_norm,
        'path_save_model': ops.path_save_model,
        'save_model_every_seconds': ops.save_model_every_seconds,
        'num_model_to_save': ops.num_models_to_save,
        'log_dir': ops.log_dir,
        'gpu': gpu,
        'tensorboard': tensorboard,
        'vocab': vocab,
        'words': words
    }

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=ops.lr, weight_decay=ops.weight_decay)

    trainer = CCTrainer(model, optim, dataset, dev_dataset, options=train_options, gpu=gpu)

    # trainer.train()
    trainer.restore_model()
    # dump_weights(trainer._model, 'data/h5py/CNN_W2V.hdf5')
    #
    # load_weights('data/h5py/CNN_W2V.hdf5')

    words = words_from_vocab('data/vocab.txt')
    dump_embedding(trainer._model, words, 'data/embed.txt')






