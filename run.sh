#!/usr/bin/env bash

# base dataset direction:
# -> full_datas
# -> sub_datas

base_data_dir='full_datas'
echo "==> base_data_dir:" ${base_data_dir}

cd features
sh run.sh ${base_data_dir}
cd ../model/

if [ ${base_data_dir} == 'sub_datas' ]
then
    python simple_xgboost_feature_select.py -d ${base_data_dir}
else
    python xgboost_model.py -d ${base_data_dir}
fi
