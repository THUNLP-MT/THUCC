# 文言文阅读理解 #

本软件是面向高考答题的文言文阅读理解系统，可用于高考语文卷的文言文阅读理解题目。

本软件使用了基于注意力机制的多层卷积神经网络框架。

## 运行环境 ##

本软件在以下环境通过测试，但应能在兼容环境下运行。

- 64位linux
- python 2.7
- cuda 7.5
- theano 0.8.2


## 运行方法 ##
训练：

`python train.py vocab.txt train.txt dev.txt`
其中,vocab.txt为训练所用词表, train.txt为训练所用数据集, dev.txt为开发集。

测试：

`python test.py -i input.txt -o output.txt -m WenYan_model`

其中`input.txt`为存放题目的文件，`WenYan_model`为训练好的模型文件，`output.txt`为输出文件。

`input.txt`的格式为`Label\tDocument\tQuestion`,共四行，每一行的`Question`为问题的一个选项。


