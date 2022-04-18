#!/bin/bash

#parallel
mkdir -p $4
python -m tracking.main track --dataset_path=dataset --improved=True --degree_thres=$1 --score_thres=$2 --divider=$3 --result_path=tracking/tracking_results/$4_$1_$2_$3.pkl --score_func=0
python -m tracking.main evaluate --result_path=tracking/tracking_results/$4_$1_$2_$3.pkl > "$4/res_$1_$2_$3.txt"
rm -f tracking_results/$4_$1_$2_$3.pkl