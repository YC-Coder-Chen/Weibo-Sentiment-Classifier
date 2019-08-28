#!/usr/bin/env python3

import mxnet as mx
import pickle
from model import load_LSTM_net, load_CNN_net, predict_sentiment

with open('./data/vocab.pkl', "rb") as fp:
      vocab = pickle.load(fp) 

# load the LSTM Model      
embed_size, num_hiddens, num_layers, ctx, bidirectional, fasttext = 300, 60, 2, mx.cpu(), True, True
net = load_LSTM_net(fasttext, vocab, embed_size, num_hiddens, num_layers, ctx, bidirectional)
net.load_parameters('./data/params_best_LSTM')

# # load the CNN Model  
# embed_size, kernel_sizes, nums_channels, ctx, fasttext = 300, [2, 3, 4, 5], [65, 65, 65, 65], mx.gpu(), True
# net = load_CNN_net(fasttext, vocab, embed_size, kernel_sizes, nums_channels, ctx)
# net.load_parameters('./data/params_best_CNN')

# sample test, LSTM works well on shorter text, while CNN works well on longer text
sentence = '我今天心情好'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return positive"

sentence = '我今天心情不好'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return negative"

sentence = '今天真操蛋'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return negative"

sentence = '祝大家新年快乐呀'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return positive"

sentence = '下地狱吧'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return negative"

sentence = '你咋不上天呢'
predict_sentiment(net, vocab, sentence, ctx)
"LSTM return negative"

sentence = '今天出太阳啦，我今天特别心情好'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return positive"

sentence = '居然要下雨了，我没有带伞啊，今天真的是操蛋'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return negative"

sentence = '杨哥永远相信支持你，不要管那些流言蜚语，你要知道你是我们永远前进的动力，在长沙好好玩哦，湖南欢迎你，常来长沙哦'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return positive"

sentence = '孙杨:我的训练和生活受到了巨大困扰，已远远超出承受范围。 如果十年辉煌，十年热爱都无法证明自己的清白，那么坚持的意义到底又在哪里？脱下战神衣袍，他只是一个人，有血有肉，有苦有泪，会崩溃也会心碎。'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return negative"

sentence = '真的已经到了非要他本人一个字一个字为自己辩解的时候了吗，大众的舆论真的太可怕了。所以说，不是没有证据，也根本不是逃避什么，只是时候未到而已，真相和事实绝对不会被掩埋。求求那些媒体了，不要再往他身上插刀了，他真的无法承受了'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return negative"

sentence = '在激进示威者的持续扰乱下，降价、亏损、歇业，诸多行业遭受重大影响，香港经济增速创10年来新低，失业率攀升至2.9%。这些香港“冰与火”的反差，让人心痛！'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return negative"

sentence = '不知道大家有没有发现一个特别严肃的问题，这帮香港废青真的是奇丑无比，一个个尖嘴猴腮的，一点也不像我们传统中国人的容貌！真的是求求你们还是把狗嘴套子戴上吧，别出来恶心人了！'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return negative"

sentence = '这个颜色好好看啊 我觉得是挺好的 和极光色有点像呢'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return positive"

sentence = '哈哈哈哈哈听完彩虹的歌词真的又好笑又有意义[哈哈][哈哈][哈哈]，生活哲理啊都是[doge] #年龄是结婚的必要因素吗# 因为大学不懂恋爱晚了'
predict_sentiment(net, vocab, sentence, ctx)
"CNN return positive"