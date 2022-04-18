#!/bin/bash
filename="results"

mkdir -p ${filename}

degrees=(60)
score_thres=(1.5 2.5)
divider=(4 6)

for d in "${degrees[@]}"; do
    for s in "${score_thres[@]}"; do
        for di in "${divider[@]}"; do 
            python -m tracking.main track --dataset_path=dataset --improved=True --degree_thres=$d --score_thres=$s --divider=$di --result_path=tracking/tracking_results/${filename}_${d}_${s}_${di}.pkl --score_func=0
            python -m tracking.main evaluate --result_path=tracking/tracking_results/${filename}_${d}_${s}_${di}.pkl > "${filename}/res_${d}_${s}_${di}.txt"
            rm -f tracking/tracking_results/${filename}_${d}_${s}_${di}.pkl
        done;
    done;
done