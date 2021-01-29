'''
Author: Kai Niu
Date: 2020-12-20 01:44:15
LastEditors: Kai Niu
LastEditTime: 2021-01-15 23:33:48
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, output_size: int, vocab_size: int,
                 embedding_dim: int, lstm_hidden_size: int,
                 num_classes: int, lstm_num_layers: int, dropout: float = 0.8):
        super(Model, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_size, lstm_num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size * 2 +
                            embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.word_embeddings(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = torch.nn.functional.max_pool1d(
            out, kernel_size=out.size()[-1]).squeeze()
        out = self.fc(out)
        return out
