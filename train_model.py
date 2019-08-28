#!/usr/bin/env python3
import mxnet as mx
from model import load_data, prepare_data, load_LSTM_net, load_CNN_net, train_k_fold, plot_k_fold, train_on_all_data
from mxnet import gluon
from mxnet.gluon import loss as gloss

"Prepare the data"
all_data = load_data()
train_data = all_data[0:int(len(all_data)*0.75)]
test_data = all_data[int(len(all_data)*0.75):]
vocab, train_features, train_labels, test_features, test_labels = prepare_data(train_data, test_data, 80)

"Load the LSTM model, set model parameters"
embed_size, num_hiddens, num_layers, ctx, bidirectional, fasttext = 300, 60, 2, mx.gpu(), True, True
net = load_LSTM_net(fasttext, vocab, embed_size, num_hiddens, num_layers, ctx, bidirectional) # load the LSTM Model

# "Load the CNN model, set model parameters"
# embed_size, kernel_sizes, nums_channels, ctx, fasttext = 300, [2, 3, 4, 5], [65, 65, 65, 65], mx.gpu(), True
# net = load_CNN_net(fasttext, vocab, embed_size, kernel_sizes, nums_channels, ctx)

"Define training parameter"
learning_rate, num_epochs, K, batch_size = 0.005, 65, 5, 64
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
loss = gloss.SoftmaxCrossEntropyLoss()

"K-fold cross-validation"
result_train, result_valid = train_k_fold(net, K, train_features, train_labels, num_epochs, learning_rate, batch_size, trainer, loss, ctx)
plot_k_fold(result_train, result_valid)

"Train on all the dataset"
# for the LSTM Model
num_epochs = 15 # k-fold shows that 15 epoches is the best for LSTM
Label = "LSTM" 
net = load_LSTM_net(fasttext, vocab, embed_size, num_hiddens, num_layers, ctx, bidirectional)
train_on_all_data(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, batch_size, trainer, loss, ctx, Label)

# # for the CNN Model
# num_epochs = 12 # k-fold shows that 12 epoches is the best for CNN
# Label = "CNN" 
# net = load_CNN_net(fasttext, vocab, embed_size, kernel_sizes, nums_channels, ctx)
# train_on_all_data(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, batch_size, trainer, loss, ctx, Label)