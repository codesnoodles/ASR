import random

import numpy as np

from general_function.file_wav import *
from extract_feature import *

DATALIST_PATH = '../datalist/'
DATA_PATH = '/home/wuboyong/corpus/'

class DataSpeech():
    def __init__(self, path, feature_type, corpus, type, audio_length=1600, feature_size=80, LoadToMen = False, MenWavCount = 10000):
        self.datapath = path
        self.feature_type = feature_type
        self.corpus = corpus
        self.type = type
        self.audio_length = audio_length
        self.feature_size = feature_size
        #self.dic_wavList = {}
        #self.dic_symbolList = {}
        #self.data, self.label = self.load_data_and_label()
        #self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2], 1)
        self.wav_list, self.label_list = self.get_wavlist()
        self.DataNum = self.getDataNum()
        self.speech_dict = self.get_dict()

    def get_dict(self):
        dict = {}
        count = 0
        with open('../dict.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for i in lines:
            txt_l = i.split('\t')
            # list_symbol.append(txt_l[0])
            dict[txt_l[0]] = count
            count += 1
        return dict

    def get_wavlist(self):
        datalistFile = DATALIST_PATH + self.corpus + '/' + self.type + '.wav.txt'
        labellistFile = DATALIST_PATH + self.corpus + '/' + self.type + '.syllable.txt'
        return get_wav_list(datalistFile), get_wav_symbol(labellistFile)

    def load_data_and_label(self):
        dataPath = self.datapath + '/' + self.feature_type + '.' + self.corpus + '.' + self.type + '.Feature.npy'
        labelPath = self.datapath + '/' + self.corpus + '.' + self.type + '.Label.npy'
        data = np.load(dataPath)
        label = np.load(labelPath)
        return data, label

    def getDataNum(self):
        return len(self.wav_list)

    def get_feature_label(self):
        ran_num = random.randint(0, self.DataNum - 1)
        # data_input = self.data[ran_num]
        wav_name = self.wav_list[ran_num]
        wave_data, fs = read_wav_data(DATA_PATH + wav_name)
        if self.feature_type == 'spectrogram':
            data_input = getFeature_spectrogram(wave_data, fs)
        elif self.feature_type == 'fbank':
            data_input = getFeature_Fbank(wave_data, fs)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        data_input = data_input[0:self.audio_length]
        label_input = []
        for j in self.label_list[ran_num]:
            label_input.append(self.speech_dict[j])
        return data_input, label_input


    def data_generator(self, batch_size=32):
        labels = np.zeros((batch_size, 1), dtype=np.float)

        while True:
            X = np.zeros((batch_size, self.audio_length, self.feature_size, 1), dtype=np.float)
            y = np.zeros((batch_size, 64), dtype=np.int16)

            input_length = []
            label_length = []

            for i in range(batch_size):
                data_input, label_input = self.get_feature_label()
                # MaxPooling scale
                # input_length.append(data_input.shape[0]//8 + data_input.shape[0]%8)
                input_length.append(data_input.shape[0] // 16)
                #print('data_input' + str(data_input))
                X[i, 0:len(data_input)] = data_input
                y[i, 0:len(label_input)] = label_input
                label_length.append([len(label_input)])

            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T

            #print(input_length)
            #print(label_length)
            #print(X.shape)
            yield [X, y, input_length, label_length], labels

        pass

    #def getData(self):