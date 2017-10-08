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
from utils import data_utils, jobs
from conf.configure import Configure


def main(base_data_dir):
    op_scope = 0
    if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
        return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

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
