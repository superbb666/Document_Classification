'''
Author: Kai Niu
Date: 2020-12-20 02:15:41
LastEditors: Kai Niu
LastEditTime: 2021-01-16 00:16:42
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, output_size: int, vocab_size: int,
                 embedding_dim: int, lstm_hidden_size: int,
                 num_classes: int, lstm_num_layers: int,
                 hidden_size2: int, dropout: float = 0.8):
        super(Model, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_size, lstm_num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(lstm_hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(lstm_hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # [batch_size, seq_len, embeding]=[128, 32, 300]
        emb = self.word_embeddings(x)
        # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        H, _ = self.lstm(emb)

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w),
                          dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
