'''
@Author: your name
@Date: 2019-10-25 09:01:10
@LastEditTime: 2019-11-15 09:53:16
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /AutoTextGenerate/src/dataset.py
'''
import torch
import string
from torch.utils import data
from rnn import Model

all_letters = string.ascii_letters + '.:\'\"\n !?'
path = '../dataset/shakespeare.txt'

char2index = {char:i for i, char in enumerate(all_letters)}
index2char = {i:char for i, char in enumerate(all_letters)}
emmbed_size = len(all_letters)

#从文本文件获取待处理的数据
def get_chunk_data(path=path, chunk_size=100, step=100):
    chunk_data = []

    with open(path, 'r') as f:
        print(path)
        data = f.read()
    length = len(data)
    data_size = int((length-chunk_size)/step) + 1
    for i in range(data_size):
        input = data[i*step : i*step+chunk_size-1]
        target = data[i*step+1 : i*step+chunk_size]
        chunk_data.append((input, target))
    return chunk_data

#句子向量化，返回的shape为(len(sentence), emmbed_size)
def word2vec(sentence, emb_type='x'):
    if emb_type == 'x':
        sentence_tensor = torch.zeros([len(sentence), emmbed_size])
    else:
        sentence_tensor = torch.zeros([len(sentence)])

    for i, char in enumerate(sentence):
        if char in char2index:
            if emb_type == 'x':
                sentence_tensor[i, char2index[char]] = 1
            else:
                sentence_tensor[i] = char2index[char]
    return sentence_tensor

class Dataset(data.Dataset):
    def __init__(self, chunk_data):
        self.data = chunk_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = word2vec(self.data[index][0])
        y = word2vec(self.data[index][1], 'y')
        return X, y
'''
chunk_data = get_chunk_data()
training_set = Dataset(chunk_data)
training_generator = data.DataLoader(training_set, batch_size=64, shuffle=True)
for local_batchs, local_labels in training_generator:
    print(local_batchs.shape)
    #RNN = Model(local_batchs.size(2), local_labels.size(2), 128, 1)
    #RNN(local_batchs)
    break
print(len(training_generator))
'''