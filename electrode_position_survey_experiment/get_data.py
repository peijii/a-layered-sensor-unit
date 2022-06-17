#！/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ChenPeiJi
import os
import numpy as np
import pandas as pd
import re

# 主函数，返回所有未处理数据，处理数据和标签
def returnData(plan='A'):
    # 返回一个list
    def all_list():
        data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
        def Get_listofPlan(n=18):
            # 这个函数返回目标路径下所有文件名称的一个生成器
            def get_A_info(data_path, plan=plan):
                """
                :param data_path: 包含各个方案的文件夹
                :param plan: 哪个方案
                :return: 一个该方案下所有文件名的生成器
                """
                if plan == 'A':
                    data_path = os.path.join(data_path, 'A') 
                elif plan == 'B':
                    data_path = os.path.join(data_path, 'B') 
                elif plan == 'C':
                    data_path = os.path.join(data_path, 'C') 
                elif plan == 'D':
                    data_path = os.path.join(data_path, 'D') 
                iterool_list = os.listdir(data_path)
                for i in iterool_list:
                    yield i
                    yield from os.listdir(os.sep.join([data_path, str(i)]))
            list_ = list(get_A_info(data_path = data_path, plan=plan))
            return [list_[i:i+n] for i in range(0, len(list_), n)]
        res = Get_listofPlan()
        path_list = []
        for i in res:
            for j in range(1, len(i)):
                parent_path = i[0]
                path_list.append(os.sep.join([data_path, plan, parent_path, i[j]]))
        return path_list

    res = all_list()
    res.sort()
    before_data = []
    preprocess_data = []

    label = []
    rest = [i for i in res[16::17]]
    test = [i for i in res if i not in rest]
    
    for i in range(len(test)):
        assert re.split(r'[/, _, .]', rest[int(i / 16)])[6] == re.split(r'[/, _, .]', test[i])[6]
        sti_data = np.array(
            pd.read_excel(test[i], header=None).iloc[:, 2:]
        )
        rest_data = np.array(
            pd.read_excel(rest[int(i / 16)], header=None).iloc[:, 2:]
        )
        mark = int(re.split(r'[/, _, .]', test[i])[-2])

        wandata = sti_data - rest_data.mean(axis=0).reshape(1, -1)

        before_data.append(sti_data)
        preprocess_data.append(wandata)
        label.append(mark)

    before_data = np.array(before_data)
    preprocess_data = np.array(preprocess_data)
    label = np.array(label)

    samplerate = 1000
    start1 = 0
    end1 = 3

    start2 = 5
    end2 = 8

    raw_move1 = before_data[:, start1 * samplerate:end1 * samplerate, :]
    raw_move2 = before_data[:, start2 * samplerate:end2 * samplerate, :]
    pro_move1 = preprocess_data[:, start1 * samplerate:end1 * samplerate, :]
    pro_move2 = preprocess_data[:, start2 * samplerate:end2 * samplerate, :]
    RawX = np.append(raw_move1, raw_move2, axis=0)
    ProX = np.append(pro_move1, pro_move2, axis=0)
    label = np.array([int(x) for x in label])
    labels = np.append(label, label, axis=0)

    return RawX, ProX, labels