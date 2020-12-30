
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NetworkLSTM import *
from DatasetHW0 import *

import os
import csv


def GetModelName():
    now = datetime.now()
    timeStr = now.strftime("%y%m%d_%H%M%S")

    return 'Model_' + timeStr

def SaveModel(network, output_folder, modelName):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    torch.save(network.state_dict(), output_folder + '/' + modelName + '.pkl')

def CrossEntropyLoss(predict, target):
    epsilon = 1e-07
    predict_e = torch.clamp(predict, epsilon, 1 - epsilon)

    target_one_hot = torch.zeros(len(target), torch.max(target) + 1, dtype=float, device=target.device)
    target_one_hot = target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    target_one_hot_e = torch.clamp(target_one_hot, epsilon, 1 - epsilon)

    return -torch.mean(torch.sum(target_one_hot_e * torch.log(predict_e), 1))

def EntropyLoss(predict):
    epsilon = 1e-07

    predict_e = torch.clamp(predict, epsilon, 1 - epsilon)
    return -torch.mean(torch.sum(predict_e * torch.log(predict_e), 1))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Initalization
    semi_supervised = True

    epoch_num = 100
    batch_size = 100
    learning_rate = 0.0001
    nolabelWeight = 0.5

    now = datetime.now()
    output_folder = 'results/result_' + now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'

    writer = SummaryWriter(logPath)

    # Read Dataset
    hw0Dataset = HW0Dataset('data', max_seq_length=36, datasetType=DatasetType.TrainingLabel)
    dataloader = DataLoader(hw0Dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    # hw0Dataset.vocab_size = 28125
    lstm_net = LSTM_Net(28125, hw0Dataset.label_num)

    #modelName = 'Model_201230_161739'
    #lstm_net.load_state_dict(torch.load('results/result_201230_1401/model/' + modelName + '.pkl'))

    if torch.cuda.is_available():
        lstm_net.cuda()

    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=learning_rate)

    # Start training
    for epoch in range(epoch_num):
        try:
            if semi_supervised:
                hw0Dataset.datasetType = DatasetType.TrainingUnLabel
                trainingUnLabelEnum = enumerate(dataloader)

            hw0Dataset.datasetType = DatasetType.TrainingLabel
            trainingEnum = enumerate(dataloader)

            trainingNum = len(dataloader.dataset)

            for data0, data1 in trainingEnum if not semi_supervised else zip(trainingEnum, trainingUnLabelEnum):
                if semi_supervised:
                    batch_idx, trainingLabel_batch = data0
                    _, trainingUnLabel_batch = data1

                    text = torch.cat((trainingLabel_batch['text'], trainingUnLabel_batch['text']))
                    label = trainingLabel_batch['label']
                else:
                    batch_idx = data0
                    text = data1['text']
                    label = data1['label']

                if torch.cuda.is_available():
                    text = text.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                lstm_net.train()
                output = lstm_net(text)

                if semi_supervised:
                    loss_label = CrossEntropyLoss(output[:len(label)], label)
                    loss_nolabel = EntropyLoss(output[len(label):])
                    loss = (1 - nolabelWeight) * loss_label + nolabelWeight * loss_nolabel
                else:
                    loss = CrossEntropyLoss(output, label)

                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0 or batch_idx == (dataloader.__len__() - 1):
                    trainedNum = (batch_idx + 1) * (len(trainingLabel_batch['text']) if semi_supervised else len(data1['text']))

                    if semi_supervised:
                        print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Loss_Label: {:.6f}, Loss_Nolabel: {:.6f}').format(
                            (epoch + 1), trainedNum, trainingNum, 100. * trainedNum / trainingNum, loss.item(), loss_label.item(), loss_nolabel.item()))
                    else:
                        print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}').format(
                            (epoch + 1), trainedNum, trainingNum, 100. * trainedNum / trainingNum, loss.item()))

            if semi_supervised:
                writer.add_scalar(logPath + '/Loss', loss.item(), epoch)
                writer.add_scalar(logPath + '/Loss_Label', loss_label.item(), epoch)
                writer.add_scalar(logPath + '/Loss_Nolabel', loss_nolabel.item(), epoch)
            else:
                writer.add_scalar(logPath + '/Loss', loss.item(), epoch)

            with torch.no_grad():
                class_correct = list(0. for i in range(hw0Dataset.label_num))
                class_total = list(0. for i in range(hw0Dataset.label_num))

                hw0Dataset.datasetType = DatasetType.TrainingLabel

                for batch_idx, batch in enumerate(dataloader):
                    text = batch['text'].cuda() if torch.cuda.is_available() else batch['text']
                    label = batch['label'].cuda() if torch.cuda.is_available() else batch['label']

                    lstm_net.eval()
                    output = lstm_net(text)
                    _, predicted = torch.max(output, 1)
                    c = (predicted == label).squeeze()

                    for i in range(len(label)):
                        class_correct[label[i]] += c[i].item()
                        class_total[label[i]] += 1

                for i in range(hw0Dataset.label_num):
                    accuracy = class_correct[i] / class_total[i]
                    print('Accuracy of %s : %2d %%' % (
                        i, 100 * accuracy))
                    writer.add_scalar(logPath + '/C' + str(i) + '_Accuracy', accuracy, epoch)

                writer.add_scalar(logPath + '/Accuracy', sum(class_correct) / sum(class_total), epoch)

                # Save model
                modelName = GetModelName()
                SaveModel(lstm_net, modelPath, modelName)

        except KeyboardInterrupt:
            break

    # Testing and output result file
    with open(output_folder + '/' + modelName + '.csv', 'w', newline='') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(['id', 'label'])

        hw0Dataset.datasetType = DatasetType.TestingData

        for batch_idx, batch in enumerate(dataloader):
            text = batch['text'].cuda() if torch.cuda.is_available() else batch['text']

            lstm_net.eval()
            output = lstm_net(text)
            _, predicted = torch.max(output, 1)

            for i in range(len(predicted)):
                writer.writerow([batch['id'][i].item(), predicted[i].item()])

    print('Finish!')
