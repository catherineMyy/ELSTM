import numpy as np


def acc_pre(prec, accuDay):
    # 计算条件累计降雨量，累计天数为accuDay
    prec = prec.values[:,3:]
    result = list()
    lenth = len(prec) - accuDay + 1
    for i in range(lenth):
        tmp = np.sum(prec[i:i+accuDay,:],axis=0)
        result.append(tmp)
    result = np.array(result)
    # print(result)
    # print(result.shape)
    return result

def addHis(dataset, timestep):
    # 加入历史信息，时间窗口长度为timestep
    result = list()
    lenth = len(dataset) - timestep + 1
    for i in range(lenth):
        tmp = dataset[i:i + timestep,:]
        result.append(tmp)
    result = np.array(result, dtype=np.float32)
    return result

def TimeToPre(label, preDay):
    result = list()
    lenth = len(label) - preDay + 1
    for i in range(lenth):
        tmp = label[i:i+preDay]
        result.append(tmp)
    result = np.array(result)
    return result

def period_runoff_feature(data, lenth, pre_begin, pre_lenth):
    year_begin = data['year'][0]
    extract_feature = list()
    forward = int(lenth/2)
    backward = lenth - forward
    for date in range(pre_begin, pre_begin+pre_lenth, 1):
        # print(date)
        year = data['year'][date]
        month = data['month'][date]
        day = data['day'][date]
        period_feature = np.zeros(lenth)
        if(year==year_begin):
            if(date<lenth):
                period_feature[-date:] = data['runoff'][0:date]
                extract_feature.append(period_feature)
            else:
                period_feature = data['runoff'][date-lenth:date]
                extract_feature.append(period_feature)
        elif((year==year_begin + 1)&(day<forward)):
            period_feature = data['runoff'][0:lenth]
            extract_feature.append(period_feature)
        elif((month==2)&(day==29)):
            loc = data[(data['year'] == year-1)&(data['month'] == month)&(data['day'] == day-1)].index.values
            loc = loc[0]
            period_feature[0:forward] = data['runoff'][loc-forward+1:loc+1] 
            period_feature[forward:] = data['runoff'][loc+1:loc+backward+1]
            extract_feature.append(period_feature)
        else:
            loc = data[(data['year'] == year-1)&(data['month'] == month)&(data['day'] == day)].index.values
            loc = loc[0]
            period_feature[0:forward] = data['runoff'][loc-forward+1:loc+1] 
            period_feature[forward:] = data['runoff'][loc+1:loc+backward+1]
            extract_feature.append(period_feature)
    extract_feature = np.array(extract_feature)
    extract_feature = extract_feature.reshape(-1, lenth, 1)
    return extract_feature

def period_station_feature(data, lenth, pre_begin, pre_lenth):
    year_begin = data['year'][0]
    extract_feature = list()
    forward = int(lenth/2)
    backward = lenth - forward
    values = data.values[:,3:]
    for date in range(pre_begin, pre_begin+pre_lenth, 1):
        year = data['year'][date]
        month = data['month'][date]
        day = data['day'][date]
        period_feature = np.zeros([lenth, 6])
        if(year==year_begin):
            if(date<lenth):
                period_feature[-date:, :] = values[0:date, :]
                extract_feature.append(period_feature)
            else:
                period_feature = values[date-lenth:date, :]
                extract_feature.append(period_feature)
        elif((year==year_begin + 1)&(day<forward)):
            period_feature = values[0:lenth, :]
            extract_feature.append(period_feature)
        elif((month==2)&(day==29)):
            loc = data[(data['year'] == year-1)&(data['month'] == month)&(data['day'] == day-1)].index.values
            loc = loc[0]
            period_feature[0:forward, :] = values[loc-forward+1:loc+1, :] 
            period_feature[forward:, :] = values[loc+1:loc+backward+1, :]
            extract_feature.append(period_feature)
        else:
            loc = data[(data['year'] == year-1)&(data['month'] == month)&(data['day'] == day)].index
            loc = loc[0]
            period_feature[0:forward, :] = values[loc-forward+1:loc+1, :] 
            period_feature[forward:, :] = values[loc+1:loc+backward+1, :]
            extract_feature.append(period_feature)
    extract_feature = np.array(extract_feature)
    return extract_feature