import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

torch.manual_seed(1)


class AttenCLSTM(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, timestep, num_lstm_layers, fc_units, batch_size, dropout_rate, output_size, device):
        super(AttenCLSTM, self).__init__()
        self.timestep = timestep

        self.drop = nn.Dropout(p=dropout_rate)


        self.lstm1 = nn.LSTM(input_size=input_size1, hidden_size=hidden_size,
                        num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)

        self.lstm2 = nn.LSTM(input_size=input_size2, hidden_size=hidden_size,
                        num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)
        
        self.lstm3 = nn.LSTM(input_size = 2*hidden_size, hidden_size=hidden_size,
                        num_layers = num_lstm_layers, batch_first=True, dropout=dropout_rate)

        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, fc_units, bias=True)
        self.out = nn.Linear(fc_units, output_size, bias=True)
        self.conv = nn.Conv1d(2*hidden_size+1, 1,kernel_size=3)

    def forward(self, x, hidden1, hidden2, merge_hidden):
        i = 0
        n_sample = x[0].shape[0]
        if hidden1!=None:
            hidden1 = (hidden1[0][:,-n_sample:,:].contiguous(), hidden1[1][:,-n_sample:,:].contiguous())
            hidden2 = (hidden2[0][:,-n_sample:,:].contiguous(), hidden2[1][:,-n_sample:,:].contiguous())
            merge_hidden = (merge_hidden[0][:,-n_sample:,:].contiguous(), merge_hidden[1][:,-n_sample:,:].contiguous())
        for input_ in x:
            if(i==0):
                lstm_out1, hidden1 = self.lstm1(input_, hidden1)
                lstm_out1 = self.drop(lstm_out1)
            else:
                lstm_out2, hidden2 = self.lstm2(input_, hidden2)
                lstm_out2 = self.drop(lstm_out2)
                if(self.timestep==1):
                    lstm_out2 = lstm_out2[:,-self.timestep:,:].unsqueeze(1)
                else:
                    lstm_out2 = lstm_out2[:,-self.timestep:,:]
            i += 1
        merge = torch.cat((lstm_out1, lstm_out2), -1)

        lstm_out, merge_hidden = self.lstm3(merge, merge_hidden)
       
        lstm_out = self.drop(lstm_out)
       
        output = F.relu( self.fc(lstm_out) )
        output = self.out(output)
          
        return output, hidden1, hidden2, merge_hidden


class LSTM(nn.Module):
    def __init__(self, device, input_size=7, output_size=1, num_layers=2, hidden_size=128, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm1(x, hidden)
        out = self.fc(lstm_out)
        # out =  F.relu( self.fc(lstm_out) )
        return out, hidden


class Net_CLSTM(nn.Module):
    def __init__(self, AttenCLSTM, timestep, target_lenth, device):
        super(Net_CLSTM, self).__init__()
        self.AttenCLSTM = AttenCLSTM
        self.lstm = LSTM(device)
        self.target_lenth = target_lenth
        self.device = device
        self.fc1 = nn.Linear(7,3, bias=True)
        self.fc2 = nn.Linear(3,1, bias=True)

    def forward(self, x, hidden1, hidden2, merge_hidden, merge_hidden2):
        input_length = x[0].shape[1]
        baseline = x[0][:,:,0]
        lstm_in = baseline.unsqueeze(-1)

        for i in range(6):
            inputs = []
        
            for j in range(len(x)):
                if(j==0):
                    tmp = x[j][:,:,(0, i+1, 7+i)]
                    inputs.append(tmp)
                else:
                    tmp = x[j]
                    inputs.append(tmp)

            output, hidden1, hidden2, merge_hidden = self.AttenCLSTM(inputs, hidden1, hidden2, merge_hidden)
            lstm_in = torch.cat((lstm_in, output), axis=-1)
     
        output, merge_hidden = self.lstm(lstm_in, merge_hidden)
        output = output[:,-1,:].unsqueeze(-1)
        
        return output, hidden1, hidden2, merge_hidden, merge_hidden2, lstm_in[:,-1,:]