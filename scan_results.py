import os
import re

import numpy as np

path = "tracking/results/"

if __name__ == "__main__":
    filename_files = []
    means = []
    medians = []
    for filename in os.listdir(path):
        filename_files.append(filename)
        with open(os.path.join(path, filename), "r") as f:  # open in readonly mode
            lines = f.readlines()
            mean = lines[-2].split(",")
            median = lines[-1].split(",")

            means.append([float(re.sub("[^0-9.-]", "", s)) for s in mean])
            medians.append([float(re.sub("[^0-9.-]", "", s)) for s in median])

    means = np.asarray(means)
    medians = np.asarray(medians)
    max_motp_index_mean = np.argmax(means[:, 1])
    max_motp_index_median = np.argmax(medians[:, 1])
    print(
        f"best mean motp hyperparam: {filename_files[max_motp_index_mean]} motp: {means[max_motp_index_mean, 1]}"
    )
    print(
        f"best median motp hyperparam: {filename_files[max_motp_index_median]} motp: {medians[max_motp_index_median, 1]}"
    )
