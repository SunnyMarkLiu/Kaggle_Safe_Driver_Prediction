#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-8 下午4:29
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from optparse import OptionParser

import numpy as np
import pandas as pd
from utils import data_utils
from conf.configure import Configure
import warnings

warnings.filterwarnings('ignore')


def impute_missing_data(all_df):
    """
    填充缺失值
    """
    mode_features = ['ps_ind_04_cat', 'ps_car_07_cat', 'ps_car_11', 'ps_ind_04_cat',
                     'ps_car_01_cat', 'ps_ind_02_cat', 'ps_car_09_cat', 'ps_ind_05_cat',
                     'ps_car_07_cat']

    for col in mode_features:
        all_df[col].fillna(value=all_df[col].mode(), inplace=True)

    remain_minus_one = ['ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03', 'ps_car_14']
    for col in remain_minus_one:
        all_df[col].fillna(value=all_df[col].mean(), inplace=True)
    return all_df


def main(base_data_dir):
    op_scope = 0
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    train_target = train['target']
    del train['target']

    all_df = pd.concat([train, test])
    # all_df.replace(-1, np.NaN, inplace=True)
    #
    # print('---> perform impute missing data')
    # all_df = impute_missing_data(all_df)

    train = all_df.iloc[:train.shape[0], :]
    test = all_df.iloc[train.shape[0]:, :]

    train.loc[:, 'target'] = train_target.values
    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(base_data_dir, train, test, op_scope + 1)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-d", "--base_data_dir",
        dest="base_data_dir",
        default="full_datas",
        help="""base dataset dir: 
                    full_datas, 
                    sub_datas"""
    )

    options, _ = parser.parse_args()
    print("========== perform preprocess ==========")
    main(options.base_data_dir)
