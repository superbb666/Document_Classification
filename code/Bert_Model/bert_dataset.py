'''
Author: Kai Niu
Date: 2020-12-16 00:28:01
LastEditors: Kai Niu
LastEditTime: 2021-01-09 19:53:43
'''
import os
import csv

import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.datasets import fetch_20newsgroups


class News20Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, X, y, word_map_path, max_length=150):
        """
        :param cache_data_path: folder where data files are stored
        :param word_map_path: path for vocab dict, used for embedding
        :param max_sent_length: maximum number of words in a sentence
        :param max_doc_length: maximum number of sentences in a document 
        :param is_train: true if TRAIN mode, false if TEST mode
        """
        self.max_length = max_length

        self.X = X
        self.y = y
        self.vocab = pd.read_csv(
            filepath_or_buffer=word_map_path,
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]

    def __getitem__(self, i):
        text = self.X[i]
        label = self.y[i]

        return text, label
#        # NOTE MODIFICATION (REFACTOR)
#        doc, num_sents, num_words = self.transform(text)
#
#        if num_sents == -1:
#            return None
#
#        return doc, label
#

    def __len__(self):
        return len(self.X)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_classes(self):
        return 4
        # return len(list(self.data.target_names))
