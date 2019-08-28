# Weibo-Sentiment-Classifier
============

Welcome, this project is a Chinese Weibo-Sentiment-Classifier trained on 10k Weibo data based on 2-layer bidirectional LSTM model using MXNET and Python 3.6. We also provide a trained model based on a multi-kernel 1d text-CNN model[1] using MXNET and Python 3.6. This project is inspired by the interactive deep learning course [Dive into Deep Learning](https://d2l.ai/) and some of the codes in the project are also taken from the above course. Our LSTM model achieved 85% accuracy on 5-fold cross-validation set and 83% accuracy on the test set, while the text-CNN model achieved 83% accuracy on 5-fold cross-validation set and 82% accuracy on the test set. The project might offer you some insights on how to apply the LSTM or text-CNN model in handling informal Chinese Weibo sentiment data.

Data
------------

The provided dataset came from [Baidu Senta](https://github.com/baidu/Senta). The provided trained model is based on all the training sample from in the project. Users can apply our modeling framework to other Chinese datasets.


Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your model. The default optimizer is "ADAM", users can also change the optimizer to "SGD" or other optimizers supported by MXNET in the [model.py](/model.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained classifier.

```
# for the LSTM Model
Classifier model parameters:
embed_size = 300  # determined by the pretrained fasttext model 
num_hiddens = 60
num_layers = 2
ctx = mx.gpu() # we use one GPU to speed up training process
bidirectional = True 
fasttext = True # we load the pretrained fasttext model to partly prevent overfit

# for the text-CNN Model
Classifier model parameters:
embed_size = 300
kernel_sizes = [2, 3, 4, 5]
nums_channels = [65, 65, 65, 65]
ctx = mx.gpu()
fasttext = True

Model Training parameters: # the same for both models
learning_rate = 0.005
num_epochs = 65 
K = 5 # we use 5-fold 
batch_size = 64
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
loss = gloss.SoftmaxCrossEntropyLoss() # we use cross-entropy

```

Model Predict
------------

User can apply the trained model to convert any Chinese Weibo post into sentiment in the [predict.py](/predict.py) file. We also perform some test to check whether the model is valid or not. It turns out that LSTM works well on shorter text, while CNN works well on the longer text.

```
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
```

Author
------------

Yingxiang (David) Chen: https://github.com/YC-Coder-Chen  
Zihan (Echo) Yang: https://github.com/echoyang48


Citation
------------
[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
