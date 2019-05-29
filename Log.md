## 2019.4.10
开始复现ASRT
读取wav文件，写成string类型。

提取Feature，画频谱图。

读取数据。


## 2019.4.11
尝试提取特征到文件保存

## 2019.4.12
一天的课....晚上十点才坐在实验室...

提取完特征，保存到npy

发现现在提取特征的方法只是分帧、加窗、FFT，没经过滤波器、对数以及DCT，这个之后需要更改。
![](http://static.oschina.net/uploads/space/2014/0115/164958_fSdw_852488.jpg)

## 2019.4.13
发现label应该转换成对应的数字

发现feature的保存存在些许问题

## 2019.4.14
使用GitHub开源代码python\_speech\_features提取特征

重新提取了特征，（1700， 23）维。

但是好像每个batch中去设置最长的维度，可以节省计算。

## 2019.4.15
fbank使用的特征维度更改为80

模型训练一直存在维度不对应的报错，仍在学习。

## 2019.4.16
发现fit_generator需要用generator生成器来生成数据，因此进行这部分的编辑。

## 2019.4.17
程序终于跑通了！

但是明显存在的问题是，对CTC这部分不太了解，这地方设计的补零操作还需要研究。

## 2019.4.18
跑的结果来看，效果很差，97%的词错误率以及200左右的loss，需要调整的东西还有很多。

## 2019.5.6
我又回来啦！忙完了好几个deadline，有时间再来做做这个事情。

给自己定一个计划，就是之后要做些什么。

1.继续弄这个代码，先把CTC这块弄懂，因为相对来说使用CTC搭建模型还是容易的，并且要能知道模型正确率上不去的原因

2.研究DTW、编辑距离这些算法

3.开始学kaldi，做HMM-GMM以及强制对齐这块儿。

## 2019.5.7
了解了CTC的backend.ctc_batch_cost的输入输出。

修改了fit_generator的函数。

## 2019.5.8
利用Callback在fit_generator时加入模型检测。

loss1000左右，需要更改。

## 2019.5.9
使用讯飞的DFCNN框架，经过100次epoch，模型的loss降低到了1以下，词错误率降低到了20以下，但验证集的loss在30以上，尝试加入dropout解决。

## 2019.5.15
DFCNN使用1600*200的语谱图作为特征输入，取得了29.77%的词错误率。

正在编写VGG16。

## 2019.5.26
![词错误率](https://i.imgur.com/9gnzFxa.png)
使用fbnak能达到WER在20%左右（其实还没有添加语言模型，这只是拼音序号的词错误率）

spectrogram	
	DFCNN	29.77%	
	VGG16	28.18%
	

fbank	
	VGG16	28.77%	
	DFCNN	17.86%	

这个是使用32个样例做预测得到的结果，太少了，因此又用fbank做了256个样例预测，得到的WER为20.69%

因为30小时数据量是在太少，因此我又添加了实验室内部的intel的语料，正在尝试结果。