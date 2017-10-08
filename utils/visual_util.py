#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-8 下午3:19
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import matplotlib.pyplot as plt


def train_test_histogram_check_feature_distribution(train, test, feature_names, figsize=(10, 40)):
    """
    绘制 train 和 test 的相关特征的分布直方图
    :param train: 
    :param test: 
    :param feature_names: 所要 check 的 feature list
    :param figsize: 
    :return: 
    """
    fig, axes = plt.subplots(len(feature_names), 2, figsize=figsize)
    fig.tight_layout()

    left = 0  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.3  # the amount of width reserved for blank space between subplots
    hspace = 0.7  # the amount of height reserved for white space between subplot

    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)
    count = 0
    for i, ax in enumerate(axes.ravel()):
        if i % 2 == 0:
            title = 'Train: ' + feature_names[count]
            ax.hist(train[feature_names[count]], bins=30, normed=False)
            ax.set_title(title)
        else:
            title = 'Test: ' + feature_names[count]
            ax.hist(test[feature_names[count]], bins=30, normed=False)
            ax.set_title(title)
            count = count + 1
    plt.show()
