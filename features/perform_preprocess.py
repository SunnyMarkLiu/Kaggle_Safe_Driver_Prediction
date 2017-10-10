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
from utils import data_utils, dataframe_util, jobs
from conf.configure import Configure


def impute_missing_data(df):
    """
    填充缺失值
    """
    missing_df = dataframe_util.contains_null(df)

    cat_cols = [col for col in missing_df.column_name if 'cat' in col]
    bin_cols = [col for col in missing_df.column_name if 'bin' in col]
    con_cols = [col for col in missing_df.column_name if col not in bin_cols + cat_cols]

    for col in cat_cols:
        df[col].fillna(value=df[col].mode(), inplace=True)

    for col in bin_cols:
        df[col].fillna(value=df[col].mode(), inplace=True)

    for col in con_cols:
        df[col].fillna(value=df[col].mean(), inplace=True)

    return df


def main(base_data_dir):
    op_scope = 0
    if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
        return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    # train.replace(-1, np.NaN, inplace=True)
    # test.replace(-1, np.NaN, inplace=True)
    #
    # print('---> perform impute missing data')
    # train = jobs.parallelize_dataframe(train, impute_missing_data)
    # test = jobs.parallelize_dataframe(test, impute_missing_data)

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
