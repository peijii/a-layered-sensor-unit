#！/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ChenPeiJi time:2020/8/1
import numpy as np
import pandas as pd
from feature_function import *
import os
from fnmatch import fnmatch
import re

DATA_PATH = os.path.abspath(os.path.dirname(__file__))
SUB1_PATH = os.path.join(DATA_PATH, 'data', 'Test001')
SUB2_PATH = os.path.join(DATA_PATH, 'data', 'Test002')


def BackPlanData(subject_path, plan):
    
    def select_day1periods(x, sample_rate=1000):
        return pd.concat([x[500:3500], x[5500:8500]], axis=0, ignore_index=None)
    
    def select_day2periods(x, sample_rate=1000):
        return pd.concat([x[0:3000], x[5000:8000]], axis=0, ignore_index=None)
        
    res = []
    for i in os.listdir(subject_path):
        day_path = subject_path + str(os.sep) + i
        res.append([day_path + str(os.sep) + path for path in os.listdir(day_path) 
                    if path.endswith(plan)][0])
    
    Data = []
    Label = []
    for i in res:
        baseline_data_loc = i + str(os.sep) + os.listdir(i)[-1]
        baseline_data = pd.read_excel(baseline_data_loc, header=None).iloc[:, 2:].mean(axis=0)
        for j in os.listdir(i):
            if not fnmatch(j, '*relax.xlsx'):
                label = int(re.split(r'[_,.]', j)[-2])
                sti_data = pd.read_excel(i + str(os.sep) + j, header=None).iloc[:, 2:]
                want_data = sti_data - baseline_data
                final_data = pd.concat([sti_data, want_data], axis=1)
                final_data.columns = list(range(12))
                if i.split(str(os.sep))[-2] == 'Day1':
                    selet_data = final_data.apply(select_day1periods)
                    selet_data.index = list(range(selet_data.shape[0]))
                elif i.split(str(os.sep))[-2] == 'Day2':
                    selet_data = final_data.apply(select_day2periods)
                    selet_data.index = list(range(selet_data.shape[0]))
                selet_data = np.array(selet_data)
                Data.append(selet_data)
                Label.append(label)
                print(i + str(os.sep) + j + ' over !')
    Data = np.array(Data)
    Label = np.array(Label)
    return Data, Label

def extract_feature(data, label):

    featureData = []
    featureLabel = []
    timeWindow = 500
    strideWindow = 500

    for i in range(data.shape[0]):
        length = int((data.shape[1] - timeWindow) / strideWindow) + 1
        all_data = data[i]
        for j in range(length):
            pre = all_data[strideWindow*j:strideWindow*j+timeWindow, [0, 1, 2, 6, 7, 8]]
            rms = featureRMS(all_data[strideWindow*j:strideWindow*j+timeWindow, [3, 4, 5, 9, 10, 11]])
            mav = featureMAV(all_data[strideWindow*j:strideWindow*j+timeWindow, [3, 4, 5, 9, 10, 11]])
            wl = featureWL(all_data[strideWindow*j:strideWindow*j+timeWindow, [3, 4, 5, 9, 10, 11]])
            zc = featureZC(all_data[strideWindow*j:strideWindow*j+timeWindow, [3, 4, 5, 9, 10, 11]])
            ssc = featureSSC(all_data[strideWindow*j:strideWindow*j+timeWindow, [3, 4, 5, 9, 10, 11]])
            pre_feature = np.squeeze(np.array(pd.concat([pd.DataFrame(pre[:, k]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).iloc[1:, :] for k in range(pre.shape[1])])))
            
            featureStack = np.hstack((rms, mav, wl, zc, ssc, pre_feature))
            featureData.append(featureStack)
            featureLabel.append(label[i])

    featureData = np.array(featureData)
    featureLabel = np.array(featureLabel)
    Label = []
    for i in featureLabel:
        if 1 <= i <= 4:
            Label.append(1)
        elif 5 <= i <= 8:
            Label.append(2)
        elif 9 <= i <= 12:
            Label.append(3)
        else:
            Label.append(4)

    Label = np.array(Label)

    return featureData, Label

if __name__ == '__main__':
    data, label = BackPlanData(SUB1_PATH, plan='A')
    data, label = extract_feature(data, label)
    print(data.shape)
    print(label.shape)
