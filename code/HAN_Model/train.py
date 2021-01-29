'''
Author: Kai Niu
Date: 2021-01-09 03:44:13
LastEditors: Kai Niu
LastEditTime: 2021-01-16 03:52:08
'''

import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import News20Dataset, MyDataLoader, collate_fn, WrappedDataLoader
from tqdm import tqdm
from model import HierarchicalAttentionNetwork
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5)

    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--word_gru_hidden_dim", type=int, default=300)
    parser.add_argument("--sent_gru_hidden_dim", type=int, default=300)
    parser.add_argument("--word_gru_num_layers", type=int, default=1)
    parser.add_argument("--sent_gru_num_layers", type=int, default=1)
    parser.add_argument("--word_att_dim", type=int, default=200)
    parser.add_argument("--sent_att_dim", type=int, default=200)

    parser.add_argument("--vocab_path", type=str,
                        default="../data/glove/glove.7B.300d.txt")
    parser.add_argument("--cache_data_dir", type=str, default="data/news20/")
    parser.add_argument("--output_dir", type=str, default="best_model")

    # NOTE MODIFICATION (EMBEDDING)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--freeze", type=bool, default=False)

    # NOTE MODIFICATION (FEATURES)
    parser.add_argument("--use_layer_norm", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    return parser.parse_args()


def model_eval(model, val_dl):
    model.eval()
    corr, total = 0, 0
    pre = []
    label = []
    with torch.no_grad():
        for i, (docs, labels, doc_lengths, sent_lengths) in enumerate(val_dl):

            scores, word_att_weights, sentence_att_weights = model(
                docs, doc_lengths, sent_lengths)

            pre += torch.argmax(scores, dim=1).tolist()
            label += labels.tolist()
        pre = np.array(pre)
        label = np.array(label)
        score = np.sum(pre == label) / len(pre)
        print('precision', score)
    return score


def get_preprocess(dev):
    def preprocess(*batch_data):
        return [x.to(torch.int64).to(dev) for x in batch_data]

    return preprocess


def main(args: argparse.ArgumentParser):
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    max_grad_norm = args.max_grad_norm

    embed_dim = args.embed_dim
    word_gru_hidden_dim = args.word_gru_hidden_dim
    sent_gru_hidden_dim = args.sent_gru_hidden_dim
    word_gru_num_layers = args.word_gru_num_layers
    sent_gru_num_layers = args.sent_gru_num_layers
    word_att_dim = args.word_att_dim
    sent_att_dim = args.sent_att_dim

    vocab_path = args.vocab_path
    cache_data_dir = args.cache_data_dir
    output_dir = args.output_dir

    pretrain = args.pretrain
    freeze = args.freeze

    use_layer_norm = args.use_layer_norm
    dropout = args.dropout

    min_count = args.min_count
    maximum_vocab_size = args.max_vocab

    output_dir = '''output/lr_%f_dropout_rate_%f_batch_size_%d_epochs_%d_
    embed_dim_%d_word_gru_hidden_dim_%d_sent_gru_hidden_dim_%d_
    word_gru_num_layers_%d_sent_gru_num_layers_%d_word_att_dim_%d_sent_att_dim_%d''' % (
        lr, dropout, batch_size, num_epochs, embed_dim, word_gru_hidden_dim,
        sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim)
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
    val_dl = MyDataLoader(val_dataset, batch_size=batch_size)
    train_dl = MyDataLoader(train_dataset, batch_size=batch_size)

    val_dl = WrappedDataLoader(val_dl, preprocess)
    train_dl = WrappedDataLoader(train_dl, preprocess)

    # self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
    model = HierarchicalAttentionNetwork(
        num_classes=train_dataset.num_classes,
        vocab_size=tokenizor.vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout).to(dev)

    if vocab_path:
        # glove_path: str, vocab: List, embed_dim, device
        pretrained_embed = get_pretrained_weights(
            vocab_path, tokenizor.token_to_id.keys(), embed_dim, dev)
        model.sent_attention.word_attention.embeddings.weight = nn.Parameter(
            pretrained_embed)

    model.train()
    model.to(dev)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_score = 0
    with open(os.path.join(output_dir, 'training.log'), 'w') as f:
        for epoch in tqdm(range(num_epochs)):
            f.write(
                '----------------------epoch %d--------------------------------\n' % (epoch))
            for i, (docs, labels, doc_lengths, sent_lengths) in enumerate(train_dl):
                model.train()
                scores, word_att_weights, sentence_att_weights = model(
                    docs, doc_lengths, sent_lengths)
                loss = F.cross_entropy(scores, labels)
                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                if (i + len(train_dl) * epoch) % 50 == 0:
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
