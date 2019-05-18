import random

import numpy as np

class Data():
    def __init__(self, corpus):
        self.trainDataFile = '../feature/' + corpus + '.test.Feature.npy'
        self.trainLabelFile = '../feature/' + corpus + '.test.Label.npy'
        self.devDataFile = '../feature/' + corpus + '.dev.Feature.npy'
        self.devLabelFile = '../feature/' + corpus + '.dev.Label.npy'
        dataTrain = np.load(self.trainDataFile)
        dataDev = np.load(self.devDataFile)
        print(dataTrain.shape[0])
        print(dataTrain.shape[1])
        exit(0)
        self.data_train = dataTrain[0].reshape(dataTrain[0].shape[0], dataTrain[0].shape[1], 1)
        self.data_dev = dataDev[0].reshape(dataDev[0].shape[0], dataDev[0].shape[1], 1)
        self.label_train = np.load(self.trainLabelFile)
        self.label_dev = np.load(self.devLabelFile)
        self.len_inputs = self.data_train.shape[0]
        self.len_labels = self.data_dev.shape[0]
        self.list_symbol = self.GetSymbolList()

        #for i in self.list_symbol:
        #    n = self.SymbolToNum(i)
        print(self.list_symbol)
        exit(0)


    def shuffle_features_and_labels(self):
        rand_indexes = np.random.permutation(self.data_train.shape[0])
        self.data_train = self.data_train[rand_indexes]
        self.label_train = self.label_train[rand_indexes]

    def GetSymbolList(self):
        '''
        加载拼音符号列表，用于标记符号
        返回一个列表list类型变量
        '''
        txt_obj = open('dict.txt', 'r', encoding='UTF-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_lines = txt_text.split('\n')  # 文本分割
        list_symbol = []  # 初始化符号列表
        for i in txt_lines:
            if (i != ''):
                txt_l = i.split('\t')
                list_symbol.append(txt_l[0])
        txt_obj.close()
        list_symbol.append('_')
        self.SymbolNum = len(list_symbol)
        return list_symbol

    def GetSymbolNum(self):
        '''
        获取拼音符号数量
        '''
        return len(self.list_symbol)

    def SymbolToNum(self, symbol):
        '''
        符号转为数字
        '''
        if (symbol != ''):
            return self.list_symbol.index(symbol)
        return self.SymbolNum

    def NumToVector(self, num):
        '''
        数字转为对应的向量
        '''
        v_tmp = []
        for i in range(0, len(self.list_symbol)):
            if (i == num):
                v_tmp.append(1)
            else:
                v_tmp.append(0)
        v = np.array(v_tmp)
        return v



def data_generator(train_x, train_y, batch_size=32):
    labels = np.zeros((batch_size, 1), dtype=np.float)
    train_length = len(train_x)

    while True:
        X = []
        y = []
        input_length = []
        label_length = []
        for i in range(batch_size):
            ran_num = random.randint(0, train_length-1)
            data_input = train_x[ran_num]
            data_label = train_y[ran_num]
            # MaxPooling scale
            #input_length.append(data_input.shape[0]//8 + data_input.shape[0]%8)
            input_length.append(data_input.shape[0]//8)
            X.append(data_input)
            y.append(data_label)
            label_length.append(len(data_label))

        label_length = np.matrix(label_length)
        input_length = np.array([input_length]).T
        yield [X, y, input_length, label_length], labels
    pass



if __name__ == '__main__':
    data = Data('st-cmds')



