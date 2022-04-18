import os
import re
import sys

import numpy as np

if __name__ == "__main__":
    path = sys.argv[1]
    filename_files = []
    means = []
    medians = []
    for filename in os.listdir(path):
        filename_files.append(filename)
        with open(os.path.join(path, filename), "r") as f:
            lines = f.readlines()
            mean = lines[-2].split(",")
            median = lines[-1].split(",")

            means.append([float(re.sub("[^0-9.-]", "", s)) for s in mean])
            medians.append([float(re.sub("[^0-9.-]", "", s)) for s in median])

    baseline_mean_mota, baseline_mean_motp = 0.3242546364826732, 0.6178319349826938
    baseline_median_mota, baseline_median_motp = 0.2969026999896306, 0.6222910549319655
    print(
        f"baseline mean metric: {baseline_mean_mota / baseline_mean_motp} baseline median metric: {baseline_median_mota / baseline_median_motp}"
    )
    means = np.asarray(means)
    medians = np.asarray(medians)
    mean_metric = means[:, 0] / means[:, 1]
    median_metric = medians[:, 0] / medians[:, 1]

    min_mean_metric = np.argmin(mean_metric)
    min_median_metric = np.argmin(median_metric)

    print(
        f"best mean hyperparam: {filename_files[min_mean_metric]} metric: {mean_metric[min_mean_metric]}, mota: {means[min_mean_metric, 0]} motp: {means[min_mean_metric, 1]}"
    )
    print(
        f"best median hyperparam: {filename_files[min_median_metric]} metric: {median_metric[min_median_metric]} mota: {medians[min_median_metric, 0]} motp: {medians[min_median_metric, 1]}"
    )
