
def getSymbolList(datapath):
    #datapath = datapath.strip('dataset/')
    datapath += '/'
    dictpath = datapath + '/' + 'dict.txt'
    with open(dictpath, encoding='UTF-8') as f:
        lines = f.readlines()

    list_symbol = []
    for i in lines:
        if i != '':
            line = i.strip()
            line = line.split('\t')
            list_symbol.append(line[0])
    list_symbol.append('_')
    return list_symbol

if __name__ == '__main__':
    getSymbolList('../data')