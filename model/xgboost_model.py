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

import pandas as pd
import xgboost as xgb
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

    y_train = train['target']

    train.drop(['id', 'target'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    print("train: {}, test: {}".format(train.shape, test.shape))
    print('---> feature check before modeling')
    feature_util.feature_check_before_modeling(train, test, train.columns)

    print("---> start cv training")
    X_train = train
    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    # kfold = 5
    # skf = StratifiedKFold(n_splits=kfold, random_state=42)

    xgb_params = {
        'eta': 0.01,
        'subsample': 0.9,
        'max_depth': 8,
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'nthread': -1,
        'silent': 1
    }

    cv_result = xgb.cv(dict(xgb_params),
                       dtrain,
                       num_boost_round=400,
                       early_stopping_rounds=100,
                       verbose_eval=20,
                       show_stdv=False,
                       feval=metric.gini_xgb,
                       maximize=True
                       )

    best_num_boost_rounds = len(cv_result)
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))
    # train model
    print('---> training on total training data')
    model = xgb.train(dict(xgb_params), dtrain,
                      num_boost_round=best_num_boost_rounds)

    print('---> predict submit')
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'id': id_test, 'target': y_pred})
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
    print("========== apply xgboost model ==========")
    main()
