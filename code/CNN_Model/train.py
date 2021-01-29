'''
Author: Kai Niu
Date: 2021-01-09 03:44:13
LastEditors: Kai Niu
LastEditTime: 2021-01-16 04:16:35
'''

import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import News20Dataset
from tqdm import tqdm
from CNN import CNN
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from utils import get_pretrained_weights, Tokenizor


def parse_args():
    parser = argparse.ArgumentParser("GloVe train")
    parser.add_argument(
        "--epochs", help="number of epochs", dest="epochs", type=int, default=5
    )
    parser.add_argument(
        "--batch-size", help="size of the batch", dest="batch_size", type=int, default=512
    )
    parser.add_argument(
        "--embedding-dim", help="size of word embedding", dest="embedding_dim", type=int, default=300
    )
    parser.add_argument(
        "--lr", help="initial learning rate", dest="lr", type=float, default=0.05
    )
    parser.add_argument(
        "--dropout-rate", help="dropout rate for the model", dest="dropout_rate", type=float, default=0.5
    )
    parser.add_argument(
        "--embedding_path", help="pretrained embedding path", dest="embedding_path", type=str, default=None
    )
    parser.add_argument(
        "--max-vocab",
        help="number of most frequent words to keep",
        dest="max_vocab",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--min-count",
        help="lower limit such that words which occur fewer than <int> times are discarded",
        dest="min_count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--filter-size", help="the number of filter for CNN", dest="filter_size", type=int, default=2
    )
    parser.add_argument(
        "--kernel-size", help="a list of kernel size", dest="kernel_size", type=str, default='1'
    )

    return parser.parse_args()


def model_eval(model, val_dl):
    model.eval()
    corr, total = 0, 0
    pre = []
    label = []
    for i, (xb, yb) in enumerate(val_dl):
        outputs = model(xb)
        pre += torch.argmax(outputs, dim=1).tolist()
        label += yb.tolist()
    pre = np.array(pre)
    label = np.array(label)
    precision = np.sum(pre == label)/len(pre)
    print('precision', precision)
    return precision


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def get_preprocess(dev):
    def preprocess(x, y):
        return x.to(torch.int64).to(dev), y.to(torch.int64).to(dev)

    return preprocess


def main(args: argparse.ArgumentParser):
    lr = args.lr
    dropout_rate = args.dropout_rate
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_path = args.embedding_path
    min_count = args.min_count
    maximum_vocab_size = args.max_vocab

    filter_size = args.filter_size
    kernel_size = list(map(int, args.kernel_size.split(',')))
    output_dir = '''output/lr_%f_dropout_rate_%f_filter_size_%d_kernel_size_%s_batch_size_%d_epochs_%d''' % (
        args.lr, args.dropout_rate, args.filter_size, args.kernel_size, args.batch_size, args.epochs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # the size of training dataset
    raw_data = fetch_20newsgroups(
        data_home='data/news20',
        subset='train',
        categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
        shuffle=False,
        remove=('headers', 'footers', 'quotes'))

    X, y = raw_data['data'], raw_data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    tokenizor = Tokenizor(X, maximum_vocab_size, min_freq=min_count)
    train_dataset = News20Dataset(
        X_train, y_train, tokenizor, 200)
    val_dataset = News20Dataset(
        X_test, y_test, tokenizor, 200)
    print('The size of training dataset is %d. The size of development dataset is %d' % (
        len(X_train), len(X_test)))

    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    preprocess = get_preprocess(dev)
    val_dl = DataLoader(val_dataset, batch_size=batch_size)
    train_dl = DataLoader(train_dataset, batch_size=32)

    val_dl = WrappedDataLoader(val_dl, preprocess)
    train_dl = WrappedDataLoader(train_dl, preprocess)

    # self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
    model = CNN(output_size=train_dataset.num_classes,
                vocab_size=train_dataset.vocab_size,
                embedding_length=embedding_dim,
                filter_size=filter_size,
                kernel_sizes=kernel_size,
                dropout=dropout_rate)
    if embedding_path:
        # glove_path: str, vocab: List, embed_dim, device
        pretrained_embed = get_pretrained_weights(
            embedding_path, tokenizor.token_to_id.keys(), embedding_dim, dev)
        model.word_embeddings.weight = nn.Parameter(pretrained_embed)

    model.train()
    model.to(dev)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_score = 0
    with open(os.path.join(output_dir, 'training.log'), 'w') as f:
        for epoch in tqdm(range(epochs)):
            f.write(
                '----------------------epoch %d--------------------------------\n' % (epoch))
            for i, (xb, yb) in enumerate(train_dl):
                model.train()
                outputs = model(xb)
                loss = F.cross_entropy(outputs, yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (i + len(train_dl) * epoch) % 10 == 0:
                    f.write('iterate %d loss %.2f\n' %
                            (i + len(train_dl) * epoch, loss))
            cur_score = model_eval(model, val_dl)
            if cur_score > best_score:
                best_score = cur_score
                torch.save(model.state_dict(), output_dir + '/model')
        f.write('The best score is %f\n' % (best_score))


if __name__ == "__main__":
    args = parse_args()
    main(args)
