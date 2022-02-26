import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from detection.dataset import PandasetDataset
from detection.model import DetectionModelConfig
from detection.modules.loss_target import create_heatmap

if __name__ == "__main__":
    # data_root = os.path.dirname('/u/csc490h/dataset/')
    # model_config = DetectionModelConfig()
    # dataset = PandasetDataset(data_root, model_config)
    # test_vox = dataset[0][0][0].numpy()
    # m, n = test_vox.shape

    # af = np.flipud(test_vox)
    # args = np.argwhere(af)
    # # plt.figure(figsize=(m / 2, n / 2))
    # plt.figure()
    # plt.scatter(args.T[1, :], args.T[0, :], s=2)

    # plt.savefig("plots/vox_vis", bbox_inches="tight")

    test_mat = torch.zeros(5, 5, 2)

    for i in range(5):
        for j in range(5):
            test_mat[i, j, 0] = i
            test_mat[i, j, 1] = j

    heatmap = create_heatmap(test_mat, torch.Tensor([2, 2]), 1)
    plt.figure()
    plt.matshow(heatmap, origin="lower")
    plt.savefig("plots/test_heatmap")
