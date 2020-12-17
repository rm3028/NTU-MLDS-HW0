# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
import torch

class LSTM_Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Net, self).__init__()
        self.embeddingLayer = torch.nn.Embedding(input_size, hidden_size)
        self.lstmLayer = torch.nn.LSTM(hidden_size, hidden_size)
        self.fcLayer = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, input, hidden):
        x = self.embeddingLayer(input)
        x, hidden = self.lstmLayer(x, hidden)
        x = self.fcLayer(x)
        output = self.softmax(x)
        return output, x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Read datasets
    csv_separator = ' \+\+\+\$\+\+\+ '

    training_label_df = pd.read_csv('data/training_label.csv', names = ['label', 'text'], sep=csv_separator, engine='python')
    training_nolabel_df = pd.read_csv('data/training_nolabel.csv', names = ['text'], sep=csv_separator, engine='python')
    testing_data = pd.read_csv('data/testing_data.csv', sep=r'(?<=\d|\w),(?!\s)', engine='python')

    # Preprocessing
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    training_label_df['encoded_text'] = training_label_df['text'].apply(tokenizer.encode, add_special_tokens = False)
    one_hot_label_df= pd.get_dummies(training_label_df['label'])
    training_nolabel_df['encoded_text'] = training_nolabel_df['text'].apply(tokenizer.encode, add_special_tokens = False)
    testing_data['encoded_text'] = testing_data['text'].apply(tokenizer.encode, add_special_tokens = False)

    max_seq_len = max([len(seq) for seq in training_label_df['encoded_text']])

    lstm_net = LSTM_Net(max_seq_len, 128, one_hot_label_df.shape[1])

    training_label_ts = torch.tensor(training_label_df['encoded_text'].values)

    epoch_num = 100
    for epoch in range(epoch_num):


    testing_data.head()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
