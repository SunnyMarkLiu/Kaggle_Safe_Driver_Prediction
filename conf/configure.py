#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 下午3:14
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time


class Configure(object):

    base_data_path = '/d_2t/lq/kaggle/Kaggle_Safe_Driver_Prediction/input/'
    original_train_path = base_data_path + 'train.csv'
    original_test_path = base_data_path + 'test.csv'

    processed_train_path = base_data_path + 'operate_{}_train.h5'
    processed_test_path = base_data_path + 'operate_{}_test.h5'

    submission_path = '../result/submission_{}.csv.gz'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
