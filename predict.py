#!/usr/bin/env python3

import mxnet as mx
import pickle
from model import load_net, predict_sentiment

with open('./data/vocab.pkl', "rb") as fp:
      vocab = pickle.load(fp) 
      
embed_size, num_hiddens, num_layers, ctx, bidirectional, fasttext = 300, 60, 2, mx.cpu(), True, True
net = load_net(fasttext, vocab, embed_size, num_hiddens, num_layers, ctx, bidirectional)
net.load_parameters('./data/params_best')

sentence = '我今天心情好'
predict_sentiment(net, vocab, sentence, ctx)
"return positive"

sentence = '我今天心情不好'
predict_sentiment(net, vocab, sentence, ctx)
"return negative"

sentence = '今天真操蛋'
predict_sentiment(net, vocab, sentence, ctx)
"return negative"

sentence = '祝大家新年快乐呀'
predict_sentiment(net, vocab, sentence, ctx)
"return positive"

sentence = '下地狱吧'
predict_sentiment(net, vocab, sentence, ctx)
"return negative"