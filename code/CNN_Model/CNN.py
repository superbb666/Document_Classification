'''
Author: Kai Niu
Date: 2020-12-18 20:47:43
LastEditors: Kai Niu
LastEditTime: 2021-01-09 03:42:19
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List


class CNN(nn.Module):
    def __init__(self,
                 output_size: int,
                 vocab_size: int,
                 embedding_length: int,
                 kernel_sizes: List = [2, 3, 4],
                 filter_size: int = 2,
                 dropout: float = 0.8):
        super(CNN, self).__init__()

        """
		Arguments
		---------
		output_size : 2 = (pos, neg)
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		kernal_sizes:  The size of cnn filter
        filter_size: The size of CNN output channel
		dropout : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""

        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        # Initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        # self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.dropout = nn.Dropout(dropout)
        self.filter_size = filter_size
        self.kernel_sizes = kernel_sizes
        self.CNNs = nn.ModuleList(
            [nn.Conv2d(1, filter_size, (h, embedding_length)) for h in kernel_sizes])
        self.label = nn.Linear(filter_size * len(kernel_sizes), output_size)

    def forward(self, input_sentence: torch.Tensor):
        # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = self.word_embeddings(input_sentence)
        input = input.unsqueeze(1)
        cnn_outputs = [F.relu(cnn(input).squeeze(-1)) for cnn in self.CNNs]
        pool_outpus = [nn.functional.max_pool1d(
            cnn_output, cnn_output.size()[-1]).squeeze(-1) for cnn_output in cnn_outputs]
        merged_out = torch.cat(pool_outpus, dim=1)
        merged_out = self.dropout(merged_out)
        logits = self.label(merged_out)

        return logits
