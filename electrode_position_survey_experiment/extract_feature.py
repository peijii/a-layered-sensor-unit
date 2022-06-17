#！/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ChenPeiJi
"""
特征提取程序
"""
from feature_function import featureRMS, featureMAV, featureWL, featureSSC
from get_data import *
from feature_list import *
import pandas as pd
import numpy as np
import os

#提取emg特征
def get_emg_dataframe(RawX, ProX):
    """
    :param RawX: 没有基线
    :param ProX: 有基线
    :return: 返回一个包括有基线和没有基线的特征矩阵 --dataframe
    """
    raw_emg = RawX[:, :, 3:]
    emg = ProX[:, :, 3:]

    RMS_raw_emg = []
    MAV_raw_emg = []
    WL_raw_emg = []
    SSC_raw_emg = []

    RMS_emg = []
    MAV_emg = []
    WL_emg = []
    SSC_emg = []

    for i in range(raw_emg.shape[0]):
        raw_rms = featureRMS(raw_emg[i, :, :])
        raw_mav = featureMAV(raw_emg[i, :, :])
        raw_wl = featureWL(raw_emg[i, :, :])
        raw_ssc = featureSSC(raw_emg[i, :, :])

        rms = featureRMS(emg[i, :, :])
        mav = featureMAV(emg[i, :, :])
        wl = featureWL(emg[i, :, :])
        ssc = featureSSC(emg[i, :, :])

        RMS_raw_emg.append(raw_rms)
        MAV_raw_emg.append(raw_mav)
        WL_raw_emg.append(raw_wl)
        SSC_raw_emg.append(raw_ssc)

        RMS_emg.append(rms)
        MAV_emg.append(mav)
        WL_emg.append(wl)
        SSC_emg.append(ssc)

    RMS_raw_emg = np.array(RMS_raw_emg)
    MAV_raw_emg = np.array(MAV_raw_emg)
    WL_raw_emg = np.array(WL_raw_emg)
    SSC_raw_emg = np.array(SSC_raw_emg)

    RMS_emg = np.array(RMS_emg)
    MAV_emg = np.array(MAV_emg)
    WL_emg = np.array(WL_emg)
    SSC_emg = np.array(SSC_emg)

    RMS_RAW = pd.DataFrame(RMS_raw_emg, columns=rRMS)
    MAV_RAW = pd.DataFrame(MAV_raw_emg, columns=rMAV)
    WL_RAW = pd.DataFrame(WL_raw_emg, columns=rWL)
    SSC_RAW = pd.DataFrame(SSC_raw_emg, columns=rSSC)

    RMS = pd.DataFrame(RMS_emg, columns=sRMS)
    MAV = pd.DataFrame(MAV_emg, columns=sMAV)
    WL = pd.DataFrame(WL_emg, columns=sWL)
    SSC = pd.DataFrame(SSC_emg, columns=sSSC)

    emg_feature = pd.concat([RMS_RAW, MAV_RAW, WL_RAW, SSC_RAW, RMS, MAV, WL, SSC], axis=1)

    return  emg_feature

#提取形变信号
def get_press_dataframe(RawX, ProX):
    """
    :param RawX: 原始压力信号
    :param ProX: 减去基线压力信号
    :return:
    """
    raw_press = RawX[:, :, 0:3]
    press = ProX[:, :, 0:3]

    press2_feature = []
    press3_feature = []
    for i in range(raw_press.shape[0]):
        row1 = pd.DataFrame(raw_press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[0, :]
        row2 = pd.DataFrame(raw_press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[1, :]
        row3 = pd.DataFrame(raw_press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[2, :]
        press1_feature = pd.concat([row1, row2, row3], axis=0)
        press1_feature = np.array(press1_feature)
        press2_feature.append(press1_feature)

        row4 = pd.DataFrame(press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[0, :]
        row5 = pd.DataFrame(press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[1, :]
        row6 = pd.DataFrame(press[i, :, :]).describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.iloc[2, :]
        press_feature = pd.concat([row4, row5, row6], axis=0)
        press_feature = np.array(press_feature)
        press3_feature.append(press_feature)

    press2_feature = np.array(press2_feature)
    press3_feature = np.array(press3_feature)

    pr_feature = pd.concat([pd.DataFrame(press2_feature), pd.DataFrame(press3_feature)], axis=1)
    pr_feature.columns = rfeature_list

    return pr_feature

#将前两步得到的EMG feature 和 deformation feature 拼接到一起
def back_emg_press_dataframe_tocsv(emg_feature, pr_feature, labels, plan):
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'all_feature_' + plan + '.csv')
    all_feature = pd.concat([emg_feature, pr_feature], axis=1)
    all_feature.drop(['raw_count1', 'raw_count2', 'raw_count3', 'count1', 'count2', 'count3'], axis=1, inplace=True)
    all_feature_label = pd.concat([all_feature, pd.DataFrame(labels)], axis=1)
    #保存目录，可以保存到你想要的目录
    all_feature_label.to_csv(save_path, index=None)

# 主函数
def main():
    pattern = ['A', 'B', 'C', 'D']
    for i in range(4):
        RawX, ProX, labels = returnData(plan=pattern[i])
        emg_feature = get_emg_dataframe(RawX, ProX)
        pr_feature = get_press_dataframe(RawX, ProX)
        back_emg_press_dataframe_tocsv(emg_feature, pr_feature, labels, plan=pattern[i])


# 运行主函数
if __name__ == '__main__':
    main()
