import torch
import logging
import time
from torch import nn
from dataset import *
from rnn import *
from helper import *

train_log_path = '../logs/train.log'

def train(model, train_data, epoch_num=100, lr=0.001):
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

def main():
    chunk_data = get_chunk_data()
    training_set = Dataset(chunk_data)
    training_generator = data.DataLoader(training_set, batch_size=64, shuffle=True)
    RNN = Model(len(all_letters), len(all_letters), 128, 1)
    train(RNN, training_generator)

if __name__ == "__main__":
    main()