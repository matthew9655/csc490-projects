import os

import matplotlib.pyplot as plt
import numpy as np

from detection.dataset import PandasetDataset
from detection.model import DetectionModelConfig
from detection.modules.voxelizer import VoxelizerConfig

if __name__ == "__main__":
    # data_root = os.path.dirname("/u/csc490h/dataset/")
    data_root2 = os.path.dirname("dataset/")

    # 2.1.2
    steps = [0.25, 0.5, 1.0, 2.0]
    for step in steps:
        v_config = VoxelizerConfig(
            x_range=(-76.0, 76.0),
            y_range=(-50.0, 50.0),
            z_range=(0.0, 10.0),
            step=step,
        )

        model_config = DetectionModelConfig(voxelizer=v_config)

        dataset = PandasetDataset(data_root2, model_config)
        test_vox = dataset[0][0][0].numpy()
        m, n = test_vox.shape

        af = np.flipud(test_vox)
        args = np.argwhere(af)
        plt.figure()
        plt.scatter(args.T[1, :], args.T[0, :], s=2)
        plt.show()
