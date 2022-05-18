from matplotlib.pyplot import xlim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from glob import glob
from torchtext import data

import json
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(data.Dataset):
    def __init__(self, filename, fields):

        examples = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                text, label = line.strip().split('\t')
                examples.append(data.Example.fromlist([label, text], fields))

        super(MyDataset, self).__init__(examples, fields)


def data_iter(train_path, valid_path, test_path, batch_size=32, device=device):

    TEXT = data.Field(
        sequential=True, tokenize=lambda text: list(str(text)), lower=True, fix_length=128)
    LABEL = data.Field(sequential=False, use_vocab=False)
    fields = [('label', LABEL), ('text', TEXT)]
    train = MyDataset(train_path, fields)
    valid = MyDataset(valid_path, fields)
    test = MyDataset(test_path, fields)
    TEXT.build_vocab(train, vectors=None)

    vocab = TEXT.vocab.itos
    
    json_str = json.dumps(TEXT.vocab.stoi, ensure_ascii=False, indent=4)
    with open('stoi.json', 'w',encoding='utf-8') as json_file:
        json_file.write(json_str)
    

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        shuffle=True,
        repeat=False)

    return train_iter, valid_iter, test_iter, vocab


def main():

    train_iter, valid_iter, test_iter, vocab = data_iter(train_path=r"dataset/sentiment/sentiment.train.data",
                                                         valid_path=r"dataset/sentiment/sentiment.valid.data",
                                                         test_path=r"dataset/sentiment/sentiment.test.data")

    print("Model: LSTM ")

    model = BiLSTMWithAttention(len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("start traning...")

    best_acc = 0

    for epoch in range(10):
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)

        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

        if best_acc <= valid_acc:
            best_acc = valid_acc
            pth = model.state_dict()
            torch.save(pth, "best.pth")
        print(f'Epoch: {epoch+1:02}, Train Acc: {train_acc * 100:.2f}%, valid Acc: {valid_acc * 100:.2f}% , Best Acc: {best_acc * 100:.2f}%')

    # load model
    test_model = BiLSTMWithAttention(len(vocab)).to(device)
    test_model.load_state_dict(torch.load("best.pth"))
    test_loss, test_acc = evaluate(test_model, test_iter, criterion)
    print(f'Test Acc: {test_acc * 100:.2f}%')


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        batch.text = batch.text.permute(1, 0)
        pred = model(batch.text)

        loss = criterion(pred, batch.label)

        acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch.text = batch.text.permute(1, 0)
            pred = model(batch.text)
            # print(bias)
            loss = criterion(pred, batch.label)
            acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class Attention(torch.nn.Module):
    
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        """
        add your code
        """
        self.w = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x):  
        """
        add your code
        """
        A1 = torch.matmul(torch.tanh(x),self.w)
        A2 = F.softmax(A1,dim=1)
        x = torch.sum(x*A2.unsqueeze(-1),1)
        return x


class BiLSTMWithAttention(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding_dim = 100
        self.hidden_dim = 32
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(vocab_size,self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=1,bidirectional=True)
        self.attention = Attention(self.hidden_dim*2)
        self.fc = nn.Linear(self.hidden_dim*2,2)

    def forward(self, x):

        """
        add your code
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x,_ = self.rnn(x)
        x = self.attention(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    main()
