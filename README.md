# Weibo-Sentiment-Classifier
============

Welcome, this project is a Chinese Weibo-Sentiment-Classifier trained on 10k Weibo data based on 2-layer bidirectional LSTM model using MXNET and Python 3.6. This project is inspired by the interactive deep learning course [Dive into Deep Learning](https://d2l.ai/) and some of the codes in the project are also taken from the above course. Our model achieved 85% accuracy on 5-fold cross-validation set and 83% accuracy on the test set. The project might offer you some insights on how to apply the LSTM model in handling informal Chinese Weibo sentiment data.

Data
------------

The provided dataset came from [Baidu Senta](https://github.com/baidu/Senta). The provided trained model is based on all the training sample from in the project. Users can apply our modeling framework to other Chinese datasets.


Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your model. The default optimizer is "ADAM", users can also change the optimizer to "SGD" or other optimizers supported by MXNET in the [model.py](/model.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained classifier.

```
Classifier model parameters:
embed_size = 300  # determined by the pretrained fasttext model 
num_hiddens = 60
num_layers = 2
ctx = mx.gpu() # we use one GPU to speed up training process
bidirectional = True 
fasttext = True # we load the pretrained fasttext model to partly prevent overfit

Training parameters:
learning_rate = 0.005
num_epochs = 65 
K = 5 # we use 5-fold 
batch_size = 64
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
loss = gloss.SoftmaxCrossEntropyLoss() # we use cross-entropy
```

Model Predict
------------

User can apply the trained model to convert any Chinese Weibo post into sentiment in the [predict.py](/predict.py) file. We also perform some test to check whether the model is valid or not. 

```
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
```

Author
------------

Yingxiang (David) Chen: https://github.com/YC-Coder-Chen
