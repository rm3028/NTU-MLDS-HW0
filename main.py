# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NetworkLSTM import *
from DatasetHW0 import *

from datetime import datetime
import os
import csv

def SaveModel(network, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    now = datetime.now()
    timeStr = now.strftime("%y%m%d_%H%M%S")

    torch.save(network.state_dict(), output_folder + '/Model_' + timeStr + '.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Initalize
    epoch_num = 100
    batch_size = 100
    learning_rate = 0.01

    now = datetime.now()
    output_folder = 'results/result_' + now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'

    writer = SummaryWriter(logPath)

    # Read Dataset
    hw0Dataset = HW0Dataset('data', DatasetType.TrainingLabel)
    dataloader = DataLoader(hw0Dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    lstm_net = LSTM_Net(hw0Dataset.vocab_size, 128, 5, hw0Dataset.label_num)
    if torch.cuda.is_available():
        lstm_net.cuda()

    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Start training
    for epoch in range(epoch_num):
        for batch_idx, batch in enumerate(dataloader):
            text = batch['text'].cuda() if torch.cuda.is_available() else batch['text']
            label = batch['label'].cuda() if torch.cuda.is_available() else batch['label']

            optimizer.zero_grad()
            output = lstm_net(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(text), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss.item()))

        writer.add_scalar(logPath + '/Loss', loss.item(), epoch)

        with torch.no_grad():
            class_correct = list(0. for i in range(hw0Dataset.label_num))
            class_total = list(0. for i in range(hw0Dataset.label_num))

            for batch_idx, batch in enumerate(dataloader):
                text = batch['text'].cuda() if torch.cuda.is_available() else batch['text']
                label = batch['label'].cuda() if torch.cuda.is_available() else batch['label']

                output = lstm_net(text)
                _, predicted = torch.max(output, 1)
                c = (predicted == label).squeeze()

                for i in range(len(label)):
                    class_correct[label[i]] += c[i].item()
                    class_total[label[i]] += 1

            for i in range(hw0Dataset.label_num):
                accuracy = class_correct[i] / class_total[i]
                print('Accuracy of %5s : %2d %%' % (
                    i, 100 * accuracy))
                writer.add_scalar(logPath + '/C' + str(i) + '_Accuracy', loss.item(), epoch)

            writer.add_scalar(logPath + '/Accuracy', sum(class_correct) / sum(class_total), epoch)

            # Save model
            SaveModel(lstm_net, modelPath)

    # Save model and output result file
    hw0Dataset.datasetType = DatasetType.TestingData

    now = datetime.now()
    timeStr = now.strftime("%y%m%d_%H%M%S")

    with open(output_folder + '/Testing_' + timeStr + '.csv', 'w', newline='') as outputFile:
        for batch_idx, batch in enumerate(dataloader):
            text = batch['text'].cuda() if torch.cuda.is_available() else batch['text']

            output = lstm_net(text)
            _, predicted = torch.max(output, 1)

            writer = csv.writer(outputFile)
            writer.writerow(['id', 'label'])

            for i in range(len(predicted)):
                writer.writerow([batch['id'][i].item(), predicted[i].item()])

    writer.close()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
