'''
Author: Kai Niu
Date: 2020-12-15 04:02:55
LastEditors: Kai Niu
LastEditTime: 2020-12-18 20:47:27
'''
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import HierarchicalAttentionNetwork
from bert_dataset import News20Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from transformers import BertConfig
from transformers import BertForSequenceClassification

from transformers import AdamW
from torch.utils.data import DataLoader

# the size of training dataset
raw_data = fetch_20newsgroups(
    data_home='data/news20',
    subset='train',
    categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
    shuffle=False,
    remove=('headers', 'footers', 'quotes'))

X, y = raw_data['data'], raw_data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42) 

train_dataset = News20Dataset(X_train, y_train, "data/glove/glove.6B.300d.txt", 200)
val_dataset = News20Dataset(X_test, y_test, "data/glove/glove.6B.300d.txt", 200) 


config = BertConfig()
config.num_labels = 4

model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path ='bert-base-uncased', num_labels=4)
model.train()


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
            


def get_preprocess(tokenizer,dev):
    def preprocess(x, y):
        encoding = tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        return encoding.to(dev), y.to(dev)
    
    return preprocess
preprocess = get_preprocess(tokenizer, dev)
train_dl = DataLoader(train_dataset, batch_size=5)
val_dl = DataLoader(val_dataset, batch_size=1)

val_dl = WrappedDataLoader(val_dl, preprocess)
train_dl = WrappedDataLoader(train_dl, preprocess)

model.to(dev)

def model_eval(model, val_dl):
    model.eval()
    corr, total = 0, 0
    for i, (xb, yb) in enumerate(val_dl):
        outputs = model(**xb)
        corr += torch.sum(torch.argmax(outputs.logits,dim=1) == yb).item()
        total += len(xb)
    print('precision', corr/total)
    model.train()

for epoch in range(5):
    for i, (xb, yb) in enumerate(train_dl):
        outputs = model(**xb, labels=yb)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 300==0:
            print('iterate %d loss %.2f' %(i, loss))
    model_eval(model, val_dl)