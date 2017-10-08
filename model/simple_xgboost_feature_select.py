#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-28 下午12:50
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from utils import metric
from utils import data_utils

from conf.configure import Configure
from utils import feature_util


def main():
    # final operate dataset
    files = os.listdir(Configure.base_data_path)
    op_scope = 0
    for f in files:
        if 'operate' in f:
            op = int(f.split('_')[1])
            if op > op_scope:
                op_scope = op

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(op_scope)

    id_test = test['id']

    y_train_all = train['target']

    train.drop(['id', 'target'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    print("train: {}, test: {}".format(train.shape, test.shape))
    print('---> feature check before modeling')
    feature_util.feature_check_before_modeling(train, test, train.columns)

    print("---> start cv training")
    train_all = train
    X_test = test
    df_columns = train.columns.values
    d_test = xgb.DMatrix(X_test, feature_names=df_columns)

    kfold = 5
    skf = StratifiedShuffleSplit(n_splits=kfold, random_state=42)

    xgb_params = {
        'eta': 0.01,
        'subsample': 0.9,
        'max_depth': 8,
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'nthread': -1,
        'silent': 1
    }

    train_rmses = []
    train_ginis = []
    valid_rmses = []
    valid_ginis = []
    roof_predict_test = 0

    for i, (train_index, valid_index) in enumerate(skf.split(train_all, y_train_all)):
        print('-----> Perform Fold %d/%d' % (i + 1, kfold))
        X_train, X_valid = train_all.ix[train_index], train_all.ix[valid_index]
        y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(xgb_params,
                          d_train,
                          num_boost_round=100,
                          evals=watchlist,
                          early_stopping_rounds=100,
                          feval=metric.gini_xgb,
                          maximize=True,
                          verbose_eval=20)

        # predict train
        predict_train = model.predict(d_train)
        train_rmse = mean_squared_error(y_train, predict_train)
        train_gini = metric.gini_normalized(y_train, predict_train)

        train_rmses.append(train_rmse)
        train_ginis.append(train_gini)

        # predict validate
        predict_valid = model.predict(d_valid)
        valid_rmse = mean_squared_error(y_valid, predict_valid)
        valid_gini = metric.gini_normalized(y_valid, predict_valid)

        valid_rmses.append(valid_rmse)
        valid_ginis.append(valid_gini)

        # Predict on our test data
        p_test = model.predict(d_test)
        roof_predict_test += p_test / kfold

    print('----------------------------------------------')
    print('Mean train rmse: {},  train gini: {}'.format(np.mean(train_rmses), np.mean(train_ginis)))
    print('Mean valid rmse: {},  valid gini: {}'.format(np.mean(valid_rmses), np.mean(valid_ginis)))

    print('---> predict submit')
    df_sub = pd.DataFrame({'id': id_test, 'target': roof_predict_test})
    submission_path = '../result/{}_submission_{}.csv.gz'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, compression='gzip')
    print('---> submit to kaggle')
    kg_password = raw_input("kaggle password: ")
    kg_comment = raw_input("submit comment: ")
    cmd = "kg submit {} -u sunnymarkliu -p '{}' -c porto-seguro-safe-driver-prediction -m '{}'".format(submission_path,
                                                                                                       kg_password,
                                                                                                       kg_comment)
    os.system(cmd)


if __name__ == "__main__":
    print("========== simple xgboost model for select features ==========")
    main()
