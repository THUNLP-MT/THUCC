# THUCC 文言文阅读理解系统

## 内容

- [软件简介](#Introduction)
- [在线演示](#Demo)
- [运行环境](#Environment)
- [使用说明](#User_Manual)
- [数据下载](#DataDownload)
- [开发人员](#Contributors)
- [开源协议](#License)

## 软件简介

文言文阅读理解是THUCC的一项功能，可用于高考语文卷的文言文阅读理解题目。
该功能由基于注意力机制的多层卷积神经网络实现。由于标注预料稀少，目前该系统的效果比较差。

## 在线演示

THUCC 文言文阅读理解系统在线演示[http://166.111.5.245:3456/index](http://166.111.5.245:3456/index)

## 运行环境

本软件在如下环境经过测试，但应能在兼容的环境下运行：

-  64位Linux 
-  Python 2.7

## 使用说明


-该系统使用以下命令进行训练：`python train.py vocab.txt train.txt dev.txt word2vec.txt`
其中,`vocab.txt`为训练所用词表, `train.txt`为训练所用数据集, `dev.txt`为开发集,`word2vec.txt`为预先训练好的词向量文件。
该系统使用以下命令进行答题：`python test.py -i input.txt -o output.txt -m WenYan_model`
-其中`input.txt`为存放题目的文件，`WenYan_model`为训练好的模型文件，`output.txt`为输出文件。
`input.txt`的格式为`Label\tDocument\tQuestion`,共四行。其中,`Label`为该行标记，可为0或1，`Document`为文言文内容，`Question`为问题的一个选项；这样的四行数据组成了一个文言文阅读理解题目。

## 数据下载

请到[这个页面](http://166.111.5.245:8900/machinecomprehension)下载数据和模型

## 开发人员

贡献者：丁延卓

## 开源协议

1. 我们面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
2. 如有机构或个人拟将改软件包用于商业目的，请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)洽谈技术许可协议。
3. 欢迎对该软件包提出任何宝贵意见和建议。请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)。
