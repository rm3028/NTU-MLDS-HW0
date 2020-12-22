
import torch
from torch.nn import *

class LSTM_Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM_Net, self).__init__()
        self.embeddingLayer = Embedding(input_size, 128)
        self.lstmLayer = LSTM(128, 512, num_layers=5, batch_first=True)
        self.fcLayer1 = Linear(512, 128)
        self.relu = ReLU()
        self.fcLayer2 = Linear(128, output_size)
        self.softmax = Softmax(dim=-1)

    def forward(self, input):
        x = self.embeddingLayer(input)
        x, _ = self.lstmLayer(x)
        x = self.fcLayer1(x[:,-1,:])
        x = self.relu(x)
        x = self.fcLayer2(x)
        output = self.softmax(x)
        return output