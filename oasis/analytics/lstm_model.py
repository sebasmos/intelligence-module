import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:,-1,:])
        return output
    
class LighterStudentLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LighterStudentLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        
        # output = self.fc(output[:,-2:,:]) # 1585,2,1
        output = self.fc(output[:,-1,:])
        return output

