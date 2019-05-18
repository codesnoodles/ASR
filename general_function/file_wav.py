import time
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def read_wav_data(filename):
    wav = wave.open(filename, 'rb')
    num_frame = wav.getnframes()
    num_channel = wav.getnchannels()
    framerate = wav.getframerate()
    num_sample = wav.getsampwidth()
    str_data = wav.readframes(num_frame)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channel
    wave_data = wave_data.T

    return wave_data, framerate

def wav_show(wave_data, fs):
    time = np.arange(0, len(wave_data)) * (1.0/fs)
    print(time)
    plt.plot(time, wave_data)
    plt.show()

x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
# Hanmming
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))

def getFrequencyFeature3(wavsignal, fs):
    if fs != 16000:
        raise ValueError('[Error] Currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')

    time_window = 25 #ms
    window_length = fs / 1000 * time_window

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10
    data_input = np.zeros((range0_end, 200), dtype=np.float)
    #data_input = []
    data_line = np.zeros((1, 400), dtype=np.float)

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400

        data_line = wav_arr[0, p_start:p_end]
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0: 200]
        #data_input.append(data_line[0: 200])

    data_input = np.log(data_input + 1)
    return data_input
'''
def get_wav_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    dic_filelist = {}
    wav_list = []
    for i in lines:
        if i != ' ':
            line = i.strip()
            line = line.split(' ')
            dic_filelist[line[0]] = line[1]
            wav_list.append(line[0])
    return dic_filelist, wav_list

def get_wav_symbol(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    dic_symbollist = {}
    symbol_list = []
    for i in lines:
        if i != ' ':
            line = i.strip()
            line = line.split(' ')
            dic_symbollist[line[0]] = line[1:]
            symbol_list.append(line[0])
    return dic_symbollist, symbol_list
'''
def get_wav_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    wav_list = []
    for i in lines:
        if i != ' ':
            line = i.strip()
            line = line.split(' ')
            wav_list.append(line[1])
    return wav_list

def get_wav_symbol(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    symbol_list = []
    for i in lines:
        if i != ' ':
            line = i.strip()
            line = line.split(' ')
            symbol_list.append(line[1:])
    return symbol_list



if __name__ == '__main__':
    wave_data, fs = read_wav_data('data/1.wav')
    t0 = time.time()
    freming = getFrequencyFeature3(wave_data, fs)
    t1 = time.time()
    print('time cost:', t1 - t0)
    freming = freming.T
    plt.subplot(111)
    plt.imshow(freming)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.show() 
    #a, b = get_wav_list('datalist/st-cmds/test.wav.txt')
