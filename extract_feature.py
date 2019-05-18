from general_function.file_wav import *
from general_function.file_dict import *
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import numpy as np
import os
import scipy.io.wavfile as wav

DATAPATH = 'datalist/'
FEATURE_TYPE = 'spectrogram'
AUDIO_LENGTH = 1600
FEATURE_SIZE = 80



def save_feature_and_label(corpus, type):
    wav_features = []
    wav_labels = []
    dict = {}
    list_symbol = []  # 初始化符号列表
    datalistFile = DATAPATH + corpus + '/' + type + '.wav.txt'
    # labellistFile = DATAPATH + corpus + '/' + type + '.syllable.txt'
    # savelabelFile = 'feature' + '/' + FEATURE_TYPE + corpus + '.' + type + '.Label.npy'
    savefeatureFile = 'feature' + '/' +  FEATURE_TYPE + '.' + corpus + '.' + type + '.Feature.npy'
    wav_list = get_wav_list(datalistFile)
    #label_list = get_wav_symbol(labellistFile)
    if not os.path.exists('feature'):
        os.makedirs('feature')

    # get dictionary_list
    with open('dict.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    count = 0
    for i in lines:
        txt_l = i.split('\t')
        #list_symbol.append(txt_l[0])
        dict[txt_l[0]] = count
        count += 1
    if FEATURE_TYPE == 'spectrogram':
        for i in wav_list:
            wave_data, fs = read_wav_data('/home/wuboyong/corpus/' + i)
            wav_feature = getFeature_spectrogram(wave_data, fs)
            wav_features.append(wav_feature)
    elif FEATURE_TYPE == 'mfcc':
        for i in wav_list:
            wave_data, fs = read_wav_data('/home/wuboyong/corpus/' + i)
            wav_feature = getFeature_MFCC(wave_data, fs)
            wav_features.append(wav_feature)
    elif FEATURE_TYPE == 'fbank':
        for i in wav_list:
            wave_data, fs = read_wav_data('/home/wuboyong/corpus/' + i)
            wav_feature = getFeature_Fbank(wave_data, fs)
            wav_features.append(wav_feature)
    wav_features = np.array(wav_features)

    np.save(savefeatureFile, wav_features)
    #print(wav_features[1])

    wav_features = np.array(wav_features)
    #print(wav_features.shape)

    '''
    for i in label_list:
        temp = []
        for j in i:
            temp.append(dict[j])
        wav_labels.append(temp)

    #wav_labels = np.array(wav_labels)
    #print(wav_labels.shape)

    #np.save(savefeatureFile, wav_features)
    np.save(savelabelFile, wav_labels)
    '''
def getFeature_spectrogram(wavsignal, fs):
    x = np.linspace(0, 400 - 1, num=400, dtype=np.int16)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    time_window = 25
    if (16000 != fs):
        raise ValueError(
            '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(
                fs) + ' Hz. ')

    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10 + 1  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, int(window_length // 2)), dtype=np.float)  # 用于存放最终的频率特征数据
    # data_line = np.zeros((1, window_length), dtype=np.float)

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400

        data_line = wav_arr[0, p_start:p_end]

        data_line = data_line * w  # 加窗

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0: int(window_length // 2)]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

    data_input = np.log(data_input + 1)
    # print(data_input.shape)
    return data_input

def getFeature_MFCC(wavsignal, fs):
    wav_feature =  mfcc(wavsignal, fs)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    return feature

def getFeature_Fbank(wavsignal, fs):
    fbank_feat = logfbank(wavsignal, fs, nfilt=80)
    return fbank_feat






if __name__ == '__main__':
    type = ['train', 'dev', 'test']
    #type = ['test']
    for i in type:
        save_feature_and_label('thchs30', i)