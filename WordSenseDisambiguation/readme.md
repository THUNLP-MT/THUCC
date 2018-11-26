THUCC 文言文词义消歧软件
========================

内容
----

-   [软件简介](#软件简介)
-   [在线演示](#在线演示)
-   [运行环境](#运行环境)
-   [使用说明](#使用说明)
-   [测试结果](#测试结果)
-   [数据下载](#数据下载)
-   [开发人员](#开发人员)
-   [开源协议](#开源协议)

## 软件简介

文言文词义消歧是THUCC的一项功能，用于自动判断文言文中多义词的义项。该功能有以下两种实现：

1.  基于卷积神经网络的词义消歧系统，需要监督学习
2.  基于词对齐的词义消歧系统，无监督学习

由于词义消歧相关的标注语料稀少，第二种实现的效果显著好于第一种实现。

## 在线演示

[http://166.111.5.245:6789/cnn](http://166.111.5.245:6789/cnn)

## 运行环境

本软件在如下环境经过测试，但应能在兼容的环境下运行：

-   64位Linux
-   Python 2.7

## 使用说明

基于卷积神经网络的词义消歧系统：

-   下载文言文词义消歧数据集，使用训练语料为各个字训练模型：`python traincnnmulti.py --train [训练集路径] --dev [开发集路径] --dic [词典文件路径]`
-   在测试语料上测试准确率：`python test_dataset.py --test [测试集路径] --dic [词典文件路径]`
-   如果想要使用文言文单语语料重新训练词向量模型，请参考[Gensim](https://radimrehurek.com/gensim/models/word2vec.html)

基于词对齐的词义消歧系统：

-   使用对齐模型进行词义消歧：`python wsd_align.py --wenyan [文言文] --baihua [白话文] --index [待消歧字的位置]`

    例如，使用如下命令对文言文中的第一个字“乃”进行消歧：`python wsd_align.py --wenyan 乃不知有汉 --baihua 竟然不知道有汉朝 --index 0`
-   如果想要使用平行语料重新训练对齐模型，请参考[TsinghuaAligner](http://nlp.csai.tsinghua.edu.cn/~ly/systems/TsinghuaAligner/TsinghuaAligner.html)

## 数据下载

请到[这个页面](http://166.111.5.245:8900/wordsensedisambiguation)下载数据和模型

## 开发人员

贡献者：张嘉成

## 开源协议

1. 我们面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
2. 如有机构或个人拟将改软件包用于商业目的，请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)洽谈技术许可协议。
3. 欢迎对该软件包提出任何宝贵意见和建议。请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)。

