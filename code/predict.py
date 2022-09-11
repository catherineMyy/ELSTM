import numpy as np
import torch
import datetime
import torch.nn as nn
from sythetic_dataset_station import SyntheticDataset
from model_station import AttenCLSTM, Net_CLSTM, FeatureAttention
from torch.utils.data import DataLoader
import random
import sklearn
from data_processing_station import *
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
from focal_loss import focal_loss

def prediction(net, loader, preDay, date, runoff_MAX, runoff_MIN, kk, verbose, device):
    hidden1 = None
    hidden2 = None
    merge_hidden = None
    merge_hidden2 = None
    true = np.zeros([1,preDay,1])
    preds = np.zeros([1,preDay,1])
    mse_arr=[]
    for i, data in enumerate(loader, 0):
        # get the inputs
        X, period, target = data
        X = torch.tensor(X, dtype=torch.float32).to(device)
        period = torch.tensor(period, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        inputs = [X, period]   
        outputs, hidden1, hidden2, merge_hidden, merge_hidden2, lstm_in = net(inputs, hidden1, hidden2, merge_hidden, merge_hidden2)
    
        # print statistics
        target = target.detach().cpu().numpy()
        out = outputs.detach().cpu().numpy()
        lstm_in = lstm_in.detach().cpu().numpy()
        mse = np.mean((target-out)**2)
        mse_arr.append(mse)

        true = np.concatenate((true, target), axis=0)
        preds= np.concatenate((preds, out), axis=0)


    true = true[1:]
    true = true*(runoff_MAX-runoff_MIN) + runoff_MIN
    preds = preds[1:,:]
    preds = preds*(runoff_MAX-runoff_MIN) + runoff_MIN
    pre_date = date[:len(true),:]
    print('data', pre_date.shape)
    

    if(verbose=='train'):
        print('*'*10, 'The result on train set: ', '*'*10)
        k = 1
        for i in range(preDay):
            true_value = true[:,i,:].reshape(-1)
            pre_value = preds[:,i,:].reshape(-1)
            pre_value[pre_value<0] = 0
            NSE = get_NSE(true_value, pre_value)
            get_peak(true_value, pre_value, kk, verbose)
            result = np.concatenate((true_value.reshape(-1,1),pre_value.reshape(-1,1)),axis=1)
            result = pd.DataFrame(result)
            result.to_csv('train_results%f.csv'%(NSE))
            mse = np.mean((true_value-pre_value)**2)
            num_peak = 0
            QA_peak = 0
            pre_peak = 0
            true_peak = 0
            j=0
            while(j<len(pre_value)-1):
                if((true_value[j]>kk)&(true_value[j+1]>=true_value[j])):
                    u = j
                    d = j
                    while(true_value[u+1]>=true_value[u]):
                        u = u+1
                    true_peak += 1
                    re_peak = abs((pre_value[u]-true_value[u])/true_value[u])
                    if(re_peak<0.2):
                        QA_peak += 1
                    if((pre_value[j]>kk)&(pre_value[j+1]>pre_value[j])):
                        while(pre_value[d+1]>=pre_value[d]):
                            d = d+1
                        pre_peak += 1
                    j = u + 1
                    num_peak += 1
                else:
                    j += 1
                # if((true_value[j]>kk)&(pre_value[j+1]<pre_value[j])&(true_value[j+1]<true_value[j])):
            re = abs((pre_value-true_value)/true_value)
            QA = len(re[re<0.2])/len(re)
            QA_ = QA_peak/num_peak
            print('The NSE of next %d day prediction is:  %f'%(k, NSE))
            print('The rmse of next %d day prediction is:  %f'%(k, np.sqrt(mse)))
            print('The num of true peaks: ', true_peak)
            print('The num of predict peaks: ', pre_peak)
            print('The QA of next %d day prediction is:  %f'%(k, QA))
            print('The QA of peak is: ', QA_)


            fig=plt.figure()
            ax1=fig.add_subplot(1,1,1)
            ax1.plot(pre_date, true_value, '--', label='target')   
            ax1.plot(pre_date, pre_value, 'r--', label='prediction')   
            ax1.set_xlabel('Number of instances')
            ax1.set_ylabel('Runoff(m3/s)')    
            plt.legend(loc='upper left')
            # ax2 = ax1.twinx()
            # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
            # ax2.set_ylabel('MSE')
            # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
            # plt.legend(loc='upper right')
            plt.savefig('train_day_%d_result_%f.png'%(k, NSE))
            plt.show()

            plt.figure()
            plt.plot(true_value, true_value, '--', color='black')
            plt.scatter(true_value, pre_value, color='dodgerblue')
            plt.ylabel('prediction')
            plt.xlabel('observation')
            plt.grid()
            plt.title('Nash-Sutcliffe Efficiency')
            plt.savefig('train_diagram%d_NSE%f.png'%(k, NSE))
            plt.show()

            k = k+1

    if(verbose=='test'):
        print('*'*10, 'The result on test set: ', '*'*10)
        k = 1
        for i in range(preDay):
            true_value = true[:,i,:].reshape(-1)
            pre_value = preds[:,i,:].reshape(-1)
            print(true_value.shape, pre_value.shape)
            pre_value[pre_value<0] = 0
            NSE = get_NSE(true_value, pre_value)
            result = np.concatenate((true_value.reshape(-1,1),pre_value.reshape(-1,1)),axis=1)
            result = pd.DataFrame(result)
            result.to_csv('test_results%f.csv'%(NSE))
            mse = np.mean((true_value-pre_value)**2)
           
            RE = np.sum(pre_value-true_value)/np.sum(true_value)
            mae = np.sum(abs(pre_value-true_value))/len(pre_value)
            
            print('The NSE of next %d day prediction is:  %f'%(k, NSE))
            print('The rmse of next %d day prediction is:  %f'%(k, np.sqrt(mse)))
            print('The RE of next %d day prediction is: %f'%(k, RE))
            print('The MAE of next %d day prediction is: %f'%(k, mae))
            
            
            fig=plt.figure()
            font1 = {'family':'Arial',
                        'weight':'normal',
                        'size':16}
            font2 = {'family':'Arial',
            'weight':'normal',
            'size':22}
            ax1=fig.add_subplot(1,1,1)
            ax1.plot(pre_date, true_value, '--', label='observation')   
            ax1.plot(pre_date, pre_value, 'r--', label='prediction')   
            ax1.set_xlabel('Time', font2)
            ax1.set_ylabel('Runoff(m3/s)', font2)  
            plt.xticks(fontproperties = 'Arial', size = 18)
            plt.yticks(fontproperties = 'Arial', size = 18) 
            # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")  
            plt.legend(loc='upper left', prop=font2)
            # ax2 = ax1.twinx()
            # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
            # ax2.set_ylabel('MSE')
            # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
            # plt.legend(loc='upper right')
            plt.savefig('test_day_%d_result_%f.png'%(k, NSE))
            plt.show()

            plt.figure()
            font1 = {'family':'Arial',
                        'weight':'normal',
                        'size':16}
            font2 = {'family':'Arial',
            'weight':'normal',
            'size':22}
            plt.plot(true_value, true_value, '--', color='black')
            plt.scatter(true_value, pre_value, color='dodgerblue')
            plt.ylabel('prediction(m3/s)', font2)
            plt.xlabel('observation(m3/s)', font2)
            plt.xticks(fontproperties = 'Arial', size = 18)
            plt.yticks(fontproperties = 'Arial', size = 18) 
            plt.grid()
            # plt.title('Nash-Sutcliffe Efficiency')
            plt.savefig('test_diagram%d_NSE%f.png'%(k, NSE))
            plt.show()

            k = k+1