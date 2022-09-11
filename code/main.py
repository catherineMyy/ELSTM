import numpy as np
import torch
import torch.nn as nn
from sythetic_dataset_station import SyntheticDataset
from model_station import AttenCLSTM, Net_CLSTM
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
from focal_loss import focal_loss
from data_processing_station import *
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
from predict import prediction
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
random.seed(1)

# parameters
batch_size = 32
epochs = 200
timestep = 5
accuDay = 2
advDay = 1
preDay = 1
period = 5


# Load synthetic dataset
train_size, test_size, date_train, date_test, runoff_MAX, runoff_MIN, X_train, period_train, Y_train, X_test, period_test, Y_test, flood_period = datasetseq_created(timestep, accuDay, advDay, preDay, period)
dataset_train = SyntheticDataset(X_train, period_train, Y_train)
dataset_test  = SyntheticDataset(X_test, period_test, Y_test)
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=False, num_workers=1, drop_last=True)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1, drop_last=False)
input_size1 = int(X_train.shape[-1]/6)+1
input_size2 = period_train.shape[-1]
# 径流量超过1000认为是洪水
kk = 1000
print(runoff_MAX, runoff_MIN)


def train_model(net, preDay, runoff_MAX, runoff_MIN, loss_type, learning_rate, epochs=1000,
                print_every=1,eval_every=1, verbose=1):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    # criterion = focal_loss()
    hidden1 = None
    hidden2 = None
    merge_hidden = None
    merge_hidden2 = None
    best_result = [np.zeros([preDay]), np.zeros([preDay]), np.zeros([preDay])]
    NSE_train = []
    NSE_test = []
    loss_train = []
    loss_test= []
    for epoch in range(epochs): 
        net.train()
        true = np.zeros([1,preDay,1])
        preds = np.zeros([1,preDay,1])
        # station_contri = np.zeros([1,7])
        totle_loss = 0
        for i, data in enumerate(trainloader, 0):
            X_train, period_train, target = data
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            period_train = torch.tensor(period_train, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]      
            inputs = [X_train, period_train]               

            # forward + backward + optimize
            outputs, hidden1, hidden2, merge_hidden, merge_hidden2, lstm_in = net(inputs, hidden1, hidden2, merge_hidden, merge_hidden2)
            h_1, c_1 = hidden1
            h_1.detach_(), c_1.detach_()    # 去掉梯度信息
            hidden1 = (h_1, c_1)
            h_2, c_2 = hidden2
            h_2.detach_(), c_2.detach_()    # 去掉梯度信息
            hidden1 = (h_2, c_2)
            h_3, c_3 = merge_hidden
            h_3.detach_(), c_3.detach_()    # 去掉梯度信息
            merge_hidden = (h_3, c_3)
            # h_4, c_4 = merge_hidden2
            # h_4.detach_(), c_4.detach_()    # 去掉梯度信息
            # merge_hidden2 = (h_4, c_4)

            loss_mse = torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                # loss = loss_mse*torch.sigmoid(loss_mse)
                # loss = torch.tanh(loss_mse)
                loss = loss_mse
                totle_loss += loss_mse.item()              

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            target = target.detach().cpu().numpy()
            out = outputs.detach().cpu().numpy()
            lstm_in = lstm_in.detach().cpu().numpy()
            true = np.concatenate((true, target), axis=0)
            preds= np.concatenate((preds, out), axis=0)

        true = true[1:]
        true = true*(runoff_MAX-runoff_MIN) + runoff_MIN
        preds = preds[1:,:]
        preds = preds*(runoff_MAX-runoff_MIN) + runoff_MIN
        
        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ', totle_loss/(i+1))
                Test_NSE, Eval_mse = eval_model(net, preDay, runoff_MAX, runoff_MIN, testloader, hidden1, hidden2, merge_hidden, merge_hidden2, verbose=1)
                NSE_test.append(Test_NSE)
                loss_test.append(Eval_mse)
                loss_train.append(totle_loss/(i+1))
                for j in range(preDay):
                    train_NSE = get_NSE(true[:,j,:], preds[:,j,:])
                    NSE_train.append(train_NSE)
                    print('Train_NSE of %d day:'%(j+1), train_NSE)
                    if(Test_NSE[j]>best_result[2][j]):
                        best_result[0][j] = epoch
                        best_result[1][j] = train_NSE
                        best_result[2][j] = Test_NSE[j]
                        torch.save(net, 'net_clstm_mse.pt')
                        
    NSE_train = np.array(NSE_train)
    NSE_test = np.array(NSE_test)
    loss_train = np.array(loss_train)
    loss_test = np.array(loss_test)
    plt.figure()
    plt.plot(NSE_train, label='Train_NSE')
    plt.plot(NSE_test, label='Test_NSE')
    plt.title('NSE')
    plt.legend()
    plt.savefig('NSE%f.png'%(best_result[2][0]))
    plt.show()

    plt.figure()
    plt.plot(loss_train, label='train')
    plt.plot(loss_test, label='test')
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss%f.png'%(best_result[2][0]))
    plt.show()

    print('The best result during train:')
    for k in range(preDay):
        print('No.%d day'%(k+1))
        print('epoch', best_result[0][k], 'Train_NSE', best_result[1][k], 'Test_NSE', best_result[2][k])
  

def eval_model(net, preDay, runoff_MAX, runoff_MIN, loader, hidden1, hidden2, merge_hidden, merge_hidden2, verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    net.eval()
    Test_NSE = np.zeros([preDay])
    true = np.zeros([1,preDay,1])
    preds = np.zeros([1,preDay,1])
    for i, data in enumerate(loader, 0):
        loss_mse = torch.tensor(0)
        
        # get the inputs
        X_test, period_test, target = data
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        period_test = torch.tensor(period_test, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)   
        inputs = [X_test, period_test]   
         
        outputs, hidden1, hidden2, merge_hidden, merge_hidden2, lstm_in = net(inputs, hidden1, hidden2, merge_hidden, merge_hidden2)
        # MSE    
        loss_mse = criterion(target,outputs)   
     
        # print statistics
        losses_mse.append( loss_mse.item() )

        target = target.detach().cpu().numpy()
        out = outputs.detach().cpu().numpy()
        true = np.concatenate((true, target), axis=0)
        preds= np.concatenate((preds, out), axis=0)

    true = true[1:]
    true = true*(runoff_MAX-runoff_MIN) + runoff_MIN
    preds = preds[1:,:]
    preds = preds*(runoff_MAX-runoff_MIN) + runoff_MIN

    for j in range(preDay):
        Test_NSE[j] = get_NSE(true[:,j,:], preds[:,j,:])
        print('Test_NSE of %d day:'%(j+1), Test_NSE[j])

    print( ' Eval mse= ', np.array(losses_mse).mean()) 
    return Test_NSE, np.array(losses_mse).mean()

decoder = AttenCLSTM(input_size1, input_size2, hidden_size=128, timestep=timestep, num_lstm_layers=2, fc_units=16, batch_size=batch_size, dropout_rate=0.2, output_size=1, device=device).to(device)
net_clstm_mse = Net_CLSTM(decoder, timestep, preDay, device).to(device)
train_model(net_clstm_mse, preDay, runoff_MAX, runoff_MIN, loss_type='mse',learning_rate=0.001, epochs=epochs, print_every=1, eval_every=1,verbose=1)


# Visualize results
net_clstm_mse = torch.load('net_clstm_mse2.pt')
net_clstm_mse.eval()
nets = [net_clstm_mse]


for net in nets:
    prediction(net, testloader, preDay, date_test, runoff_MAX, runoff_MIN, kk, 'test', device)

