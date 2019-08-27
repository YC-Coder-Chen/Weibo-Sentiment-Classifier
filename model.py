#!/usr/bin/env python3
"The below code is partial taken or modified from the below course:"
"[Dive into deep learning]: https://d2l.ai/"

import collections
import numpy as np
from mxnet import autograd, nd, init
from mxnet.gluon import data as gdata, nn, rnn
from mxnet.contrib import text
import time
import matplotlib.pyplot as plt
import jieba
import pickle

"Prepare each text into [text, tag] format"
def handle_data(x):
    tag = x.split('\t')[0]
    words = ''.join(x.split('\t')[1].split(' '))
    return [words, tag]

"Load the data"
def load_data():
    with open('./data/corpus_train.txt', encoding="utf-8") as file:
        data = file.readlines()
    all_data = [handle_data(title) for title in data]
    return all_data

"Toeknize all the words and applied jieba to cut the words"
def tokenizer(text):
    return [tok for tok in jieba.cut(text)]

"Use the tokenizer to transform the data"
def get_token(data, tokenizer = tokenizer):
    return [tokenizer(review) for review, tag in data]

"Prepare the data can create class text.vocab.Vocabulary"
def get_vocab(data, get_token=get_token): 
    tokenized_data = get_token(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # remove words that have frequency below 5
    return text.vocab.Vocabulary(counter, min_freq=5)

"Since weibo text have different length, so we padding them into same length"
def padding(x, max_length=80):
    return x[:max_length] if len(x) > max_length else x + [0] * (max_length - len(x))

"Convert string label into float"
def labeling(x):
    if x == '0':
        return 0
    if x == '1':
        return 1      

"Use the padding function to preprocess the data"
def preprocess_data(data, vocab, max_length=80, padding=padding): 
    tokenized_data = get_token(data)
    features = nd.array([padding(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([labeling(score) for _, score in data])
    return features, labels

def prepare_data(train_data, test_data, max_length=80):
    vocab = get_vocab(train_data)
    print('# words in vocab:', str(len(vocab)))
    # save the vocal for furthur prediction
    with open(f"./vocab.pkl", "wb") as fp:
        pickle.dump(vocab, fp) 
    train_features, train_labels = preprocess_data(train_data, vocab, max_length)
    test_features, test_labels = preprocess_data(test_data, vocab, max_length)
    return vocab, train_features, train_labels, test_features, test_labels

"Define th Weibo RNN model"
class Weibo_RNN(nn.Block):
    def __init__(self, vocab_embed, embed_size, num_hiddens, num_layers, bidirectional, **kwargs):
        super(Weibo_RNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab_embed), embed_size)
        # embed_size is determined by fasttext
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=bidirectional, input_size=embed_size, dropout=0.2)
        # set up the dropout layer to prevent overfit
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        """
        inputs: (batch_size, max_length)
        embeddings: (max_length, batch_size, embed_size)
        codes:  (max_length, batch_size, 2* num_hidden) if bidirectional
        codes:  (max_length, batch_size, num_hidden) if not bidirectional
        encode_output = (batch_size, 2*2*num_hidden) if bidirectional
        encode_output = (batch_size, 2*num_hidden) if not bidirectional
        output = (batch_size, 2)
        
        """      
        embeddings = self.embedding(inputs.T) # LSTM model need to transpose the input first
        codes = self.encoder(embeddings)
        encode_output = nd.concat(codes[0], codes[-1])
        output = self.decoder(encode_output)
        return output

"Load the model and the pretrained fasttext to prevent overfit"
def load_net(fasttext, vocab, embed_size, num_hiddens, num_layers, ctx, bidirectional):
    if fasttext:
        fasttext_embedding = text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec', vocabulary=vocab)
    net = Weibo_RNN(vocab, embed_size, num_hiddens, num_layers, bidirectional)
    net.initialize(init.Xavier(), ctx=ctx)
    
    if fasttext:
        net.embedding.weight.set_data(fasttext_embedding.idx_to_vec)
        net.embedding.collect_params().setattr('grad_req', 'null')
    return net

"Create k-fold"    
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid

"Define the accuary matrix for each fold"
def evaluate_accuracy_fold(net, features, labels, batch_size, ctx):
    acc_sum, n = 0, 0   
    # create data loader to prevent GPU from running out of memory
    eval_iter = gdata.DataLoader(gdata.ArrayDataset(
        features, labels), batch_size, shuffle=True)
    for X, y in eval_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n   

"Define the trainer for each fold"
def train_model(net, train_features, train_labels, test_features, test_labels, num_epochs, trainer, loss, 
                learning_rate, batch_size, ctx):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(evaluate_accuracy_fold(net, train_features, train_labels, batch_size, ctx))
        if test_labels is not None:
            test_ls.append(evaluate_accuracy_fold(net, test_features, test_labels, batch_size, ctx))
    return train_ls, test_ls

"Start trainning on k-fold"    
def train_k_fold(net, k, X_train, y_train, num_epochs, learning_rate, batch_size, trainer, loss, ctx):
    # apply two dictionary to store the epoch data
    result_train = {}
    result_valid = {}
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train_model(net, *data, num_epochs, trainer, loss, learning_rate, batch_size, ctx)
        result_train[i] = train_ls
        result_valid[i] = valid_ls
        print(f"Fold {i+1} is finished")
    return result_train, result_valid

"Plot K-fold scores to determine which epoch have the best outcome"
def plot_k_fold(result_train, result_valid):
    acc_sum_train = np.zeros_like(result_train[0])
    for result in result_train.values():
        acc_sum_train += np.array(result)
    acc_sum_train = acc_sum_train/len(result_train)
    
    acc_sum_valid = np.zeros_like(result_valid[0])
    for result in result_valid.values():
        acc_sum_valid += np.array(result)
    acc_sum_valid = acc_sum_valid/len(result_valid)
    print(f'The best number of epoch is {np.argmax(acc_sum_valid) + 1}')

    plt.xlabel('Epoches')
    plt.ylabel('Score')
    plt.semilogy(range(1, len(acc_sum_train) + 1), acc_sum_train)
    plt.semilogy(range(1, len(acc_sum_valid) + 1), acc_sum_valid, linestyle=':')
    plt.legend('K-Fold Scores')
    plt.show()

"Define the accuracy matrix for the final model trained on all the training set"
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

"Train the model on all the train sample"
def train_on_all_data(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, batch_size, trainer, loss, ctx):
    train_set = gdata.ArrayDataset(train_features, train_labels)
    test_set = gdata.ArrayDataset(test_features, test_labels)
    train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, batch_size)
    print('training on', ctx)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
    test_acc = evaluate_accuracy(test_iter, net, ctx)
    print(f'loss {train_l_sum / n}, train acc {train_acc_sum / n}, test acc {test_acc}, time {time.time() - start}')
    net.save_parameters(f'./data/params_best')

"Predict final sentiment"
def predict_sentiment(net, vocab, sentence, ctx):
    sent_list = [tok for tok in jieba.cut(sentence)] 
    sentence = nd.array(vocab.to_indices(sent_list), ctx)
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    if label.asscalar() == 0:
        return 'Negative'
    else:
        return 'Positive'
