#!/bin/bash

# for d in "${degrees[@]}"; do
#     for score in "${score_thres[@]}"; do
#         for divide in "${divider[@]}"; do 
#             python -m tracking.main track --dataset_path=dataset --improved=True --degree_thres=$d --score_thres=$score --divider=$divide
#             python -m tracking.main evaluate > "tracking/results/res_${d}_${score}_${divide}.txt"
#         done;
#     done;
# done

#parallel
python -m tracking.main track --dataset_path=dataset --improved=True --degree_thres=$1 --score_thres=$2 --divider=$3 --result_path=tracking/tracking_results/results1_$1_$2_$3.pkl
python -m tracking.main evaluate --result_path=tracking/tracking_results/results1_$1_$2_$3.pkl > "tracking/results1/res_$1_$2_$3.txt"