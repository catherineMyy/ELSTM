import torch 
import numpy as np
import pandas as pd
from feature_created import *
import matplotlib.pyplot as plt
import datetime


def datasetseq_created(timestep, accuDay, advDay, preDay, period):
    #读取数据
    runoff, prec, eva, temp = data_load()
    # normalization
    runoff, runoff_MAX, runoff_MIN = normalization2(runoff) #获取归一化后的径流和最大最小值，以便对复原预测径流
    prec = normalization1(prec)
    eva = normalization1(eva)
    temp = normalization1(temp)

    target = runoff.values[accuDay-1:,:]
    date = target[:,0:3]
    target = target[:,-1].reshape(-1,1)
    #预测径流日期
    date = date[advDay + timestep-1:,:]
    start = str(int(date[0,0]))+'-'+str(int(date[0,1]))+'-'+str(int(date[0,2]))
    end = str(int(date[-1,0]))+'-'+str(int(date[-1,1]))+'-'+str(int(date[-1,2]))
    datestart = datetime.datetime.strptime(start,'%Y-%m-%d')
    dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
    date = list()
    while(datestart<=dateend):
        datestart+=datetime.timedelta(days=1)
        date.append(datestart)
    date = np.array(date).reshape(-1,1)
    # 预测径流
    label = target[advDay + timestep-1:,:]
    month = runoff['month'].values[accuDay-1:-advDay-preDay+1].reshape(-1,1)
    month = addHis(month, timestep)
    label = TimeToPre(label, preDay)
    lenth = len(label)
    label_date = runoff.drop(['runoff'], axis=1)
    label_date = label_date.drop(labels=range(accuDay+advDay+timestep-2), axis=0) #[year, month, day]
    label_date = label_date.reset_index(drop=True)
    flood_period = label_date[(label_date['month']==5)|(label_date['month']==6)|(label_date['month']==7)|(label_date['month']==8)|(label_date['month']==9)].index.values
    # label = label[flood_period,:]
    # 划分训练集和测试集的大小
    # lenth_flood = len(label)
    train_size = int(lenth * 0.7)
    test_size = lenth - train_size
    print('the size of training and testing: ')
    print(train_size,test_size)

    # 提取径流特征，用滑动窗口增加历史信息
    runoff_fea = target[:-advDay-preDay+1,:]
    runoff_fea1 = runoff_fea[:,-1].reshape(-1,1)
    runoff_fea = addHis(runoff_fea1, timestep)
    runoff_fea = runoff_fea.reshape(-1,timestep,1)
    # runoff_fea = runoff_fea[flood_period,:]

    # 提取同期气象水文数据作为特征输入
    loc = accuDay + advDay + timestep - 2
    print(loc)
    period_runoff = period_runoff_feature(runoff, period, loc, lenth)

    # 计算条件累计降水
    acc_prec = acc_pre(prec, accuDay)
    acc_prec = acc_prec[:-advDay-preDay+1,:]
    acc_prec = addHis(acc_prec, timestep)
    
    # 提取降水、气温、蒸散发及其历史信息作为输入特征
    prec_fea = reshape_data(prec, accuDay, advDay, preDay, timestep)
    # eva_fea = reshape_data(eva, accuDay, advDay, preDay, timestep)
    # temp_fea = reshape_data(temp, accuDay, advDay, preDay, timestep)

    X = np.concatenate((runoff_fea, prec_fea, acc_prec), axis=-1)
    np.save('featureX',X)
    period = period_runoff
    temp = np.concatenate((X[:,-1,:]), axis=-1)
    Y = label
    np.save('target', Y)


    print('the size of feature and target:')
    print(X.shape, period.shape, Y.shape)
    # print(X[:,:,0], Y[-10:,:])

    X_train, X_test = X[:train_size, :], X[train_size:, :]
    period_train, period_test = period[:train_size, :], period[train_size:, :]
    Y_train, Y_test = Y[:train_size, :], Y[train_size:, :]
    date_train, date_test = date[:train_size, :], date[train_size:, :]
    
    return train_size, test_size, date_train, date_test, runoff_MAX, runoff_MIN, X_train, period_train, Y_train, X_test, period_test, Y_test, flood_period


def data_load():
    # 读取数据
    data = pd.read_csv('/data/label(2000-2009).csv', header=0)
    data.drop('num', axis=1, inplace=True)
    data.columns = ['year','month','day','runoff']
    Pre = pd.read_csv('/data/Pre-day(2000-2009).csv')
    Eva = pd.read_csv('/data/EDay_processed.csv')
    Temp = pd.read_csv('/data/Temp.csv')
    return data, Pre, Eva, Temp

def reshape_data(data, accuDay, advDay, preDay, timestep):
    # 将数据转变为输入格式
    data = data.values[:,3:]
    # data = normalization1(data)
    data = data[accuDay-1:-advDay-preDay+1,:]
    data = addHis(data, timestep)
    return data

def normalization1(dataset):
    newDataFrame = pd.DataFrame(index=dataset.index)
    columns = dataset.columns.tolist()
    for c in columns:
        if((c=='year')|(c == 'month')|(c == 'day')):
            newDataFrame[c] = dataset[c].tolist()
        else:
            d = dataset[c]
            MAX = d.max()
            MIN = d.min()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame

def normalization2(dataset):
    newDataFrame = pd.DataFrame(index=dataset.index)
    columns = dataset.columns.tolist()
    for c in columns:
        if((c=='year')|(c == 'month')|(c == 'day')):
            newDataFrame[c] = dataset[c].tolist()
        else:
            d = dataset[c]
            MAX = d.max()
            MIN = d.min()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame, MAX, MIN

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_NSE(x, y):
    NSE = 1 - (sum((y - x) ** 2)) / (sum((x - np.mean(x)) ** 2))
    return NSE
def mse(x, y):
    mse = np.mean((y-x)**2)
    return mse

def get_peak(x, y, k, verbose):
    if(verbose=='train'):
        NSE = list()
        i = 0
        peak = list()
        num_peak = 0
        peak_QA = 0
        while(i<len(x)):
            if((x[i]>=k)&(i!=0)&(i!=(len(x)-1))):
                num_peak += 1
                if((x[i+1]<=x[i])&(x[i-1]<x[i])):
                    j = i
                    u = i
                    while(x[j-1]<x[j]):
                        j = j-1
                    while((x[u+1]<x[u])|(x[u]>=500)):
                        u = u+1
                    re = abs(np.max(y[j:u])-np.max(x[j:u]))/np.max(x[j:u])
                    if(re<0.2):
                        peak_QA += 1
                    i = u+1
                    tmp = get_NSE(x[j:i], y[j:i])
                    NSE.append(tmp)
                    fig=plt.figure()
                    ax1=fig.add_subplot(1,1,1)
                    ax1.plot(x[j:i], '-', color='royalblue', label='observation')   
                    ax1.plot(y[j:i], '-', color='olivedrab', label='prediction')   
                    ax1.set_xlabel('Number of instances')
                    ax1.set_ylabel('Runoff')    
                    plt.legend(loc='upper right')
                    # ax2 = ax1.twinx()
                    # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
                    # ax2.set_ylabel('MSE')
                    # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
                    # plt.legend(loc='upper right')
                    plt.savefig('train_%dflood_%f.png'%(num_peak, tmp))
                    plt.show()
                elif((x[i-1]<x[i])&(x[i+1]>x[i])):
                    j = i
                    u = i
                    while(x[j-1]<x[j]):
                        j = j-1
                    while(x[u+1]>=x[u]):
                        u = u+1
                        i = u+1
                    while((x[u+1]<x[u])|(x[u]>=500)):
                        u = u+1
                        i = u+1
                    re = abs(np.max(y[j:u])-np.max(x[j:u]))/np.max(x[j:u])
                    if(re<0.2):
                        peak_QA += 1
                    tmp = get_NSE(x[j:i], y[j:i])
                    NSE.append(tmp)
                    fig=plt.figure()
                    ax1=fig.add_subplot(1,1,1)
                    ax1.plot(x[j:i], '-', color='royalblue', label='observation')   
                    ax1.plot(y[j:i], '-', color='olivedrab', label='prediction')   
                    ax1.set_xlabel('Number of instances')
                    ax1.set_ylabel('Runoff')    
                    plt.legend(loc='upper right')
                    # ax2 = ax1.twinx()
                    # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
                    # ax2.set_ylabel('MSE')
                    # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
                    # plt.legend(loc='upper right')
                    plt.savefig('train_%dflood_%f.png'%(num_peak, tmp))
                    plt.show()
            else:
                i = i+1
        print('The totle num of flood is: ', num_peak)
        QA = peak_QA/num_peak
        print('The QA of peak is: ', QA)
        NSE = np.array(NSE)
        NSE_ave = np.mean(NSE)
        print('The NSE of flood is: ', NSE_ave)
    if(verbose=='test'):
        NSE = list()
        i = 0
        peak = list()
        num_peak = 0
        peak_QA = 0
        while(i<len(x)):
            if((x[i]>=k)&(i!=0)&(i!=(len(x)-1))):
                num_peak += 1
                if((x[i+1]<=x[i])&(x[i-1]<x[i])):
                    j = i
                    u = i
                    while((x[j-1]<x[j])&(x[j-1]>=100)):
                        j = j-1
                    while((x[u+1]<x[u])&(x[u]>=100)):
                        u = u+1
                    re = abs(np.max(y[j:u])-np.max(x[j:u]))/np.max(x[j:u])
                    if(re<0.2):
                        peak_QA += 1
                    i = u+1
                    tmp = get_NSE(x[j:i], y[j:i])
                    NSE.append(tmp)
                    fig=plt.figure(figsize=(8,5))
                    font1 = {'family':'Arial',
                        'weight':'normal',
                        'size':12}
                    font2 = {'family':'Arial',
                    'weight':'normal',
                    'size':18}
                    ax1=fig.add_subplot(1,1,1)
                    ax1.plot(x[j:i], '-', color='royalblue', label='observation')   
                    ax1.plot(y[j:i], '-', color='olivedrab', label='prediction') 
                    plt.xlabel('Number of instances', font2)
                    plt.ylabel('Runoff(m3/s)', font2)
                    plt.xticks(fontproperties = 'Arial', size = 12)
                    plt.yticks(fontproperties = 'Arial', size = 12) 
                    # ax1.set_xlabel()
                    # ax1.set_ylabel()    
                    plt.legend(loc='upper right', prop=font2)
                    # ax2 = ax1.twinx()
                    # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
                    # ax2.set_ylabel('MSE')
                    # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
                    # plt.legend(loc='upper right')
                    plt.savefig('test_%dflood_%f.png'%(num_peak, tmp))
                    plt.show()
                elif((x[i-1]<x[i])&(x[i+1]>x[i])):
                    j = i
                    u = i
                    while((x[j-1]<x[j])&(x[j-1]>=100)):
                        j = j-1
                    while(x[u+1]>=x[u]):
                        u = u+1
                        i = u+1
                    while((x[u+1]<x[u])&(x[u]>=100)):
                        u = u+1
                        i = u+1
                    re = abs(np.max(y[j:u])-np.max(x[j:u]))/np.max(x[j:u])
                    if(re<0.2):
                        peak_QA += 1
                    tmp = get_NSE(x[j:i], y[j:i])
                    NSE.append(tmp)
                    fig=plt.figure(figsize=(8,5))
                    font1 = {'family':'Arial',
                        'weight':'normal',
                        'size':12}
                    font2 = {'family':'Arial',
                    'weight':'normal',
                    'size':18}
                    ax1=fig.add_subplot(1,1,1)
                    ax1.plot(x[j:i], '-', color='royalblue', label='observation')   
                    ax1.plot(y[j:i], '-', color='olivedrab', label='prediction')   
                    plt.xlabel('Number of instances', font2)
                    plt.ylabel('Runoff(m3/s)', font2)
                    plt.xticks(fontproperties = 'Arial', size = 12)
                    plt.yticks(fontproperties = 'Arial', size = 12) 
                    # ax1.set_xlabel()
                    # ax1.set_ylabel()    
                    plt.legend(loc='upper right', prop=font2)
                    # ax2 = ax1.twinx()
                    # ax2.plot(mse_arr, label='MSE', color='darkorange' ,ls='-.')
                    # ax2.set_ylabel('MSE')
                    # plt.title('Predict curve mse=%.3f NSE=%.3f' %(mse,NSE))
                    # plt.legend(loc='upper right')
                    plt.savefig('test_%dflood_%f.png'%(num_peak, tmp))
                    plt.show()
            else:
                i = i+1
        print('The totle num of flood is: ', num_peak)
        QA = peak_QA/num_peak
        print('The QA of peak is: ', QA)
        NSE = np.array(NSE)
        NSE_ave = np.mean(NSE)
        print('The NSE of flood is: ', NSE_ave)