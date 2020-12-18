# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils.data import DataLoader

from NetworkLSTM import *
from DatasetHW0 import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Read Dataset
    hw0Dataset = HW0Dataset('data', DatasetType.TrainingLabel)
    dataloader = DataLoader(hw0Dataset, batch_size=5, shuffle=True, num_workers=4)

    # Initialize training
    lstm_net = LSTM_Net(hw0Dataset.max_seq_len, 128, hw0Dataset.training_label_ts.shape[1])
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss

    # Start training
    epoch_num = 100
    for epoch in range(epoch_num):
        for batch_idx, (text, label) in enumerate(dataloader):
            optimizer.zero_grad()
            output = lstm_net(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(text), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss.item()))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
