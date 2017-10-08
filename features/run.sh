#!/usr/bin/env bash
# base dataset direction:
# -> full_datas
# -> sub_datas

base_data_dir=$1
python perform_preprocess.py -d ${base_data_dir}
