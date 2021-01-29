'''
Author: Kai Niu
Date: 2020-12-20 02:15:41
LastEditors: Kai Niu
LastEditTime: 2021-01-15 21:38:25
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Fasttext(nn.Module):
    def __init__(self, output_size: int, vocab_size: int,
                 embedding_length: int, dropout: float = 0.8):
        super(Fasttext, self).__init__()
        """
        Arguments
        ---------
        output_size : 2 = (pos, neg)
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        dropout : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        """
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.dropout = nn.Dropout(dropout)
        self.label = nn.Linear(embedding_length, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.word_embeddings(x)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.label(out)
        return out
