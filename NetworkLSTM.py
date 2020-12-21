
import torch
from torch.nn import *

class LSTM_Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Net, self).__init__()
        self.embeddingLayer = Embedding(input_size, hidden_size)
        self.lstmLayer = LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcLayer = Linear(hidden_size, output_size)
        self.softmax = Softmax(dim=-1)

    def forward(self, input):
        x = self.embeddingLayer(input)
        x, _ = self.lstmLayer(x)
        x = self.fcLayer(x[:,-1,:])
        output = self.softmax(x)
        return output