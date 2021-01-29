'''
Author: Kai Niu
Date: 2021-01-15 21:24:00
LastEditors: Kai Niu
LastEditTime: 2021-01-15 22:44:15
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

    def __init__(self, X, y, tokenizor, max_length=150):
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
        self.tokenizor = tokenizor

    # NOTE MODIFICATION (REFACTOR)
    def transform(self, text):
        # encode document
        doc = self.tokenizor.convert_tokens_to_ids(
            text.split())[:self.max_length]
        if len(doc) < self.max_length:
            doc += [self.tokenizor.convert_tokens_to_ids('<unk>')] * \
                (self.max_length - len(doc))
        return doc

    def __getitem__(self, i):
        text = self.X[i]
        label = self.y[i]

        text = self.transform(text)

        return torch.Tensor(text), label

    def __len__(self):
        return len(self.X)

    @property
    def vocab_size(self):
        return self.tokenizor.vocab_size

    @property
    def num_classes(self):
        return len(set(self.y))
        # return len(list(self.data.target_names))


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths = list(zip(*batch))

    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

    docs_tensor = torch.zeros(
        (bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()

    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(
            sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx,
                        :sent_length] = torch.LongTensor(sent)

    return docs_tensor, torch.LongTensor(labels), torch.LongTensor(doc_lengths), sent_lengths_tensor
