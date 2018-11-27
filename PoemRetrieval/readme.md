# THUCC 诗文检索系统

## 内容

- [软件简介](#Introduction)
- [在线演示](#Demo)
- [运行环境](#Environment)
- [使用说明](#User_Manual)
- [数据下载](#DataDownload)
- [开发人员](#Contributors)
- [开源协议](#License)

## 软件简介

本软件是THUCC中的一项功能，可用于解答高考语文卷的诗文检索题目。

软件使用lucene实现诗文检索的功能。

## 在线演示

THUCC 诗文检索系统在线演示[http://166.111.5.245:9232](http://166.111.5.245:9232)

## 运行环境

本软件在如下环境经过测试，但应能在兼容的环境下运行：

-  64位Linux 
-  Python 3.6
-  Java jre1.6

## 使用说明


进入`package`目录，输入如下命令进行诗文检索：`python retrieval.py -i input.txt -o ans.txt`
其中，`input.txt`为输入文件，存储要检索的诗文，`ans.txt`为输出文件，存储输出的结果。

## 数据下载

请到[这个页面](http://166.111.5.245:8900/poemretrieval)下载数据和模型

## 开发人员

贡献者：董梅平、柳春洋、丁延卓

## 开源协议

1. 我们面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
2. 如有机构或个人拟将改软件包用于商业目的，请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)洽谈技术许可协议。
3. 欢迎对该软件包提出任何宝贵意见和建议。请发邮件至[thumt17@gmail.com](mailto:thumt17@gmail.com)。
