'''
@Author: your name
@Date: 2019-11-05 16:15:50
@LastEditTime: 2019-11-15 09:43:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AutoTextGenerate/src/main.py
'''
import torch
import logging
import time
from torch import nn
from dataset import *
from rnn import *
from helper import *
from os import listdir
from os.path import isfile, join


train_log_path = '../logs/train.log'

def train(epoch_num=100, lr=0.001):
    chunk_data = get_chunk_data()
    training_set = Dataset(chunk_data)
    train_data = data.DataLoader(training_set, batch_size=64, shuffle=True)
    model = Model(len(all_letters), len(all_letters), 128, 1)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = check_gpu()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(train_log_path)
    info_format = logging.Formatter('%(message)s')
    handler.setFormatter(info_format)
    logger.addHandler(handler)

    logger.info('-'*40 + 'Start training: {}'.format(get_time()) + '-'*40)
    for epoch in range(1, epoch_num + 1):
        total_loss = 0
        
        for local_batchs, local_labels in train_data:
            model.zero_grad()
            local_batchs.to(device)
            output, hidden = model(local_batchs)
            loss = criterion(output, local_labels.view(-1).long())
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item()
        
        if epoch%2 == 0:
            logger.info('{} - Epoch: {}/{}.............Loss: {:.4f}'
            .format(get_time(), epoch, epoch_num, total_loss/len(train_data)))
            #print('Epoch: {}/{}.............'.format(epoch, epoch_num), end=' ')
            #print("Loss: {:.4f}".format(total_loss/len(train_data)))
            if epoch%10 == 0:
                logger.info('*'*40 + ' Saving model - Loss: {:.4f} '.format(total_loss/len(train_data)) + '*'*40)
                path = '../saved_models/rnn_epoch_{0}'.format(epoch)
                torch.save(model.state_dict(), path)

def generate_text(length=100):
    base_path = '../saved_models'
    out_path = '../output/'
    models_path = [f for f in listdir(base_path) if isfile(join(base_path, f))]

    model = Model(len(all_letters), len(all_letters), 128, 1)
    seed = 'ROMEO:'

    for fname in models_path:
        model_path = join(base_path, fname)
        print(model_path)
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            out = seed
            for i in range(length-len(seed)):
                if len(out) < 20:
                    sentence = out
                else:
                    sentence = out[len(out)-20:]
                sen_tensor = word2vec(sentence).unsqueeze(0)
                out_tensor, _ = model(sen_tensor)
                prob = torch.softmax(out_tensor[-1], dim=0).data
                indices = torch.multinomial(prob, 1)
                #values, indices = out_tensor[-1].max(0)
                char_pre = all_letters[indices]
                out += char_pre

            with open(out_path+fname+'.txt', 'w') as f:
                f.write(out)
        except:
            indices = torch.multinomial(prob, 1)
            print(indices)
            pass
        
def main():
    generate_text(300)

if __name__ == "__main__":
    main()