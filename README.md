*标签：ASR， Python， Keras， CTC*

最近在自己动手搭建一个中文语音识别系统，因为也是入门阶段，所以比较吃力，直到在GitHub上找到了一个已经在做的开源工程，找到了做下去的动力，附上原作者项目的GitHub地址：[A Deep-Learning-Based Chinese Speech Recognition System](https://github.com/nl8590687/ASRT_SpeechRecognition)
这位作者人非常好，给予了我不少启发。那么在这里也附上我自己工程的地址：[ASR](https://github.com/hldwby/ASR)
现在工程还处于起步阶段，虽然跑出了一些结果，但并不是很出色，仍旧在做一些调整，有不错的结果的时候就去更新GitHub，那现在就以本文来梳理一下搭建的思路。

## 一、数据集
在最开始，先介绍一下我使用的数据集。

我所使用的数据集是清华大学THCHS30中文语音数据集。
data_thchs30.tgz [OpenSLR国内镜像](http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz) [OpenSLR国外镜像](http://www.openslr.org/resources/18/data_thchs30.tgz)
该数据集的介绍请参考[THCHS-30：一个免费的中文语料库](https://blog.csdn.net/sut_wj/article/details/70662181)

在该数据集中，已经分好训练集、验证集和测试集（分别在train、dev、和test文件夹中），其中训练集有10000个样例，验证集有893个样例，测试集有2495个样例，每个样例大约是10秒左右的语音段。
在thchs30这个文件夹里包含了索引性质的文件（cv和dev好像是一毛一样的）
![thchs30文件夹中的文件](https://upload-images.jianshu.io/upload_images/8958330-9439e9311bc127fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)
wav.txt是音频文件的相对路径
![dev.wav.txt](https://upload-images.jianshu.io/upload_images/8958330-51cf497aa87f1c01.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
syllable.txt是对应的标签
![syllable.txt](https://upload-images.jianshu.io/upload_images/8958330-f941556e98e96399.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/3200)
标签内容首先是文件名，然后是对应的拼音内容，拼音后的数字代表几声，5代表轻声。

## 二、特征提取
通常来讲，语音识别常用的特征有MFCC、Fbank和语谱图。
在本项目中，暂时使用的是80维的Fbank特征，提取特征利用python_speech_features库，将特征提取后保存成npy文件。
提取特征在先前的文章中写了详细的做法：[使用python_speech_features提取音频文件特征](https://www.jianshu.com/p/e32d2d5ccb0d)

对于标签，项目中有dict.txt文件，这个文件的内容是字典，是拼音和汉字的对应，如下图所示。![dict.txt内容](https://upload-images.jianshu.io/upload_images/8958330-de09569638835969.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
将标签中的拼音转换成数字，例：a1为0，a2为1，以此类推。
以第一条数据为例：
lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de5 di3 se4 si4 yue4 de5 lin2 luan2 geng4 shi4 lv4 de5 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2 
转换到对应的数字列表就是：
597 910 1126 159 1121 451 191 505 1051 1209 208 215 874 939 1168 208 570 599 325 910 597 208 1072 420 1099 634 907 1140 14 829
同样，也将标签保存到npy文件中。

##三、模型搭建


待更新....
