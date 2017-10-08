#!/usr/bin/env bash

cd features
sh run.sh
cd ../model/

python simple_xgboost_feature_select.py
