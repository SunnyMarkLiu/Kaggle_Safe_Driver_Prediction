#!/usr/bin/env bash

# base dataset direction:
# -> full_datas
# -> sub_datas

base_data_dir='sub_datas'
echo "==> base_data_dir:" ${base_data_dir}

cd features
sh run.sh ${base_data_dir}
cd ../model/

python simple_xgboost_feature_select.py
