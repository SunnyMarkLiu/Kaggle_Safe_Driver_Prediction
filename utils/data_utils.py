#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
import cPickle
from conf.configure import Configure


def load_dataset(base_data_dir, op_scope):
    train_path = Configure.processed_train_path.format(base_data_dir, op_scope)
    if not os.path.exists(train_path):
        train = pd.read_csv(Configure.original_train_path)

        if base_data_dir == 'sub_datas':
            train = train.sample(frac=0.2, random_state=42, replace=True)
            train.reset_index(inplace=True)
    else:
        with open(train_path, "rb") as f:
            train = cPickle.load(f)

    test_path = Configure.processed_test_path.format(base_data_dir, op_scope)
    if not os.path.exists(test_path):
        test = pd.read_csv(Configure.original_test_path)
        # for public lb test
        if base_data_dir == 'sub_datas':
            test = test.sample(frac = 0.00001, random_state=42, replace=True)
            test.reset_index(inplace=True)

    else:
        with open(test_path, "rb") as f:
            test = cPickle.load(f)

    return train, test


def save_dataset(base_data_dir, train, test, op_scope):

    if train is not None:
        train_path = Configure.processed_train_path.format(base_data_dir, op_scope)
        with open(train_path, "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        test_path = Configure.processed_test_path.format(base_data_dir, op_scope)
        with open(test_path, "wb") as f:
            cPickle.dump(test, f, -1)
