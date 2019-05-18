import os

from keras import Input
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, GRU, add, Activation, Lambda, concatenate, Dropout
from keras.optimizers import Adam, Adadelta
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from general_function.data_generator import Data, data_generator
from general_function.file_wav import *
from general_function.file_dict import *
from general_function.general_func import GetEditDistance
from general_function.read_data import DataSpeech
from general_function.muti_gpu import *

import numpy as np
import random

NUM_GPU = 2
ModelName = 'SpeechModel_base'
#DATA_PATH = 'feature/thchs30.train.Feature.npy'
MODEL_PATH = 'model_speech'
model_name = 'keras_ASR_model.h5'
start = time.time()
WER = float('inf')


#data = Data('st-cmds')
print("Load data successful!")
'''
train_x = np.load('feature/thchs30.train.Feature.npy')
train_y = np.load('feature/thchs30.train.Label.npy')
dev_x = np.load('feature/thchs30.dev.Feature.npy')
dev_y = np.load('feature/thchs30.dev.Label.npy')
test_x = np.load('feature/thchs30.test.Feature.npy')
test_y = np.load('feature/thchs30.test.Label.npy')
'''

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        word_error_ratio = ms.TestModel('feature', dataset='dev', data_count=4)
        if word_error_ratio < WER:
            WER = word_error_ratio
            ms.SaveModel()
        return


class ModelSpeech():
    def __init__(self, corpus):
        MS_OUTPUT_SIZE = 1424
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1700
        self.AUDIO_FEATURE_LENGTH = 80
        self.datapath = 'feature'
        self._model, self.base_model = self.CreateModel()
        self.corpus = corpus

    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        '''

        '''

        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        # layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        # layer_h3 = Dropout(0.05)(layer_h3) # 随机中断部分神经网络连接，防止过拟合
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        # layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        # layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        # layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        # layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        # layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        # layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        # layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()
        # print(layer_h15.shape)

        layer_h16 = Reshape((212, 1280))(layer_h15)  # Reshape层

        # layer_h16 = Dropout(0.3)(layer_h16) # 随机中断部分神经网络连接，防止过拟合
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层

        inner = layer_h17
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层

        rnn_size = 128
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)

        gru2 = concatenate([gru_2, gru_2b])

        layer_h20 = gru2
        # layer_h20 = Dropout(0.4)(gru2)
        layer_h21 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h20)  # 全连接层

        # layer_h17 = Dropout(0.3)(layer_h17)
        layer_h22 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h21)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h22)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        
        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        # layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        loss_out = Lambda(self.ctc_lambed_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        model.summary()

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        # ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        model.build((self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
        model = ParallelModel(model, NUM_GPU)

        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        # print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model, model_data
        
        '''

        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            input_data)  # 卷积层
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        # layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
        layer_h3 = Dropout(0.1)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.2)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.3)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.3)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.4)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        ## test.summary()

        layer_h10 = Reshape((212, 1280))(layer_h12)  # Reshape层
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        # layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        layer_h10 = Dropout(0.4)(layer_h10)
        layer_h11 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h10)  # 全连接层
        layer_h11 = Dropout(0.5)(layer_h11)
        layer_h12 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h11)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h12)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        # layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # model.summary()

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)

        # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada_d)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        print('[*提示] 创建模型成功，模型编译成功')
        return model, model_data



    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


    def TrainModel(self, epoch=10, save_step=1000, batch_size=32):

        self.data = DataSpeech('../feature', 'thchs30', 'train', self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH)
        trainGen = self.data.data_generator(batch_size)
        trainNum = self.data.getDataNum()


        dev_data = DataSpeech('../feature', 'thchs30', 'dev', self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH)
        devGen = dev_data.data_generator(batch_size)
        devNum = dev_data.getDataNum()
        #g = devGen
        #x, y = next(g)

        cb = MyCallback()

        print('[*INFO] Training the Model')
        H = self._model.fit_generator(
            trainGen,
            steps_per_epoch=trainNum // batch_size,
            validation_data=devGen,
            validation_steps=devNum // batch_size,
            epochs=epoch,
            callbacks=[cb]
            )

        print('[*INFO] Evaluating the Model')
        '''
        predIdex = self._model.predict_generator(
            devGen,
            steps=((devNum // batch_size)+1)
        )
        '''
        #print(classification_report(x[1], predIdex, target_names=lb.classes_))
        self.TestModel(self.datapath, dataset='dev', data_count=4)


    '''
    def TrainModel(self, epoch=10, save_step=1000, batch_size=32):
        self.data = DataSpeech('../feature', 'thchs30', 'train',  self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH)
        yielddatas = self.data.data_generator(batch_size)
        for epoch in range(epoch):  # 迭代轮数
            print('[running] train epoch %d .' % epoch)
            n_step = 0  # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+' % (epoch, n_step * save_step))
                    # data_genetator是一个生成器函数

                    # self._model.fit_generator(yielddatas, save_step, nb_worker=2)
                    self._model.fit_generator(yielddatas, save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break

                self.SaveModel(comment='_e_' + str(epoch) + '_step_' + str(n_step * save_step))
                self.TestModel(self.datapath, dataset='train', data_count=4)
                self.TestModel(self.datapath, dataset='dev', data_count=4)
            end = time.time()
            print(str(epoch) + "epochs total running time(min)：", (end - start) // 60)
    '''

    '''
    def LoadModel(self, filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName + '.model'):
    
        # 加载模型参数
        
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')
    '''

    def SaveModel(self, filename='model_speech/' + ModelName + '/' + ModelName, comment=''):
        '''
        保存模型参数
        '''
        if not os.path.exists('model_speech/' + str(ModelName)):
            os.makedirs('model_speech/' + str(ModelName))
        self._model.save_weights(filename + comment + '.model')
        self.base_model.save_weights(filename + comment + '.model.base')
        f = open('step' + ModelName + '.txt', 'w')
        f.write(filename + comment)
        f.close()

    def TestModel(self, datapath='../feature', dataset='dev', data_count=32, out_report=False, show_ratio=True):
        '''
        测试检验模型效果
        '''
        self.data = DataSpeech('../feature', 'thchs30', dataset, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH)
        #data = DataSpeech(datapath, self.corpus, dataset)
        # data.LoadDataList(str_dataset)
        num_data = self.data.getDataNum()  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数

            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if (out_report == True):
                txt_obj = open('Test_Report_' + dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入

            txt = ''
            for i in range(data_count):
                #data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据
                data_input = self.data.data[(ran_num + i) % num_data]
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                data_labels = self.data.label[(ran_num + i) % num_data]
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_bias = 0
                while (data_input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % num_data, 'is too long.',
                          '\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    #data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
                    data_input = self.data.data[(ran_num + i + num_bias) % num_data]
                    data_labels = self.data.label[(ran_num + i + num_bias) % num_data]
                # 数据格式出错处理 结束

                pre = self.Predict(data_input, data_input.shape[0] // 8)

                words_n = len(data_labels)  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre)  # 获取编辑距离
                if (edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了

                if (i % 10 == 0 and show_ratio == True):
                    print('Test Count: ', i, '/', data_count)

                txt = ''
                if (out_report == True):
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)

            # print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + dataset + ' set word error ratio: ',
                  word_error_num / words_num * 100, '%')
            if (out_report == True):
                txt = '*[测试结果] 语音识别 ' + dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt_obj.close()
            return word_error_num / words_num * 100

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')


    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''

        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)

        in_len[0] = input_len

        x_in = np.zeros((batch_size, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.base_model.predict(x=x_in)

        # print('base_pred:\n', base_pred)

        # y_p = base_pred
        # for j in range(200):
        #	mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
        #	print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
        #	print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
        #	count=0
        #	for i in range(y_p[0][j].shape[0]):
        #		if(y_p[0][j][i] < mean):
        #			count += 1
        #	print('count:',count)

        base_pred = base_pred[:, :, :]
        # base_pred =base_pred[:, 2:, :]

        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        # print('r', r)

        r1 = K.get_value(r[0][0])
        # print('r1', r1)

        # r2 = K.get_value(r[1])
        # print(r2)

        r1 = r1[0]

        return r1
        pass

if __name__ == '__main__':
    ms = ModelSpeech('thchs30')
    ms.CreateModel()
    # 计算程序运行时间
    ms.TrainModel(epoch = 1000, batch_size = 32, save_step = 500)
    end = time.time()
    print("Running time(min)：", (end - start) // 60)

