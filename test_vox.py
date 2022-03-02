import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from detection.dataset import PandasetDataset
from detection.metrics.evaluator import Evaluator
from detection.model import DetectionModelConfig
from detection.modules.loss_function import heatmap_weighted_mse_loss
from detection.modules.loss_target import create_heatmap

if __name__ == "__main__":
    # data_root = os.path.dirname("/u/csc490h/dataset/")
    # data_root2 = os.path.dirname("dataset/")
    # model_config = DetectionModelConfig()
    # dataset = PandasetDataset(data_root2, model_config)
    # test_vox = dataset[0][0][0].numpy()
    # m, n = test_vox.shape

    # af = np.flipud(test_vox)
    # args = np.argwhere(af)
    # # plt.figure(figsize=(m / 2, n / 2))
    # plt.figure()
    # plt.scatter(args.T[1, :], args.T[0, :], s=2)

    # plt.show()

    # A = torch.tensor([1, 2, 3])
    # B = torch.tensor([4, 5, 6])
    # C = torch.stack((A, B), dim=-1)
    # print(C.size())
    # print(C)
    # A = torch.rand((5, 3))
    # print(A)
    # A = A[:, 0] > 0.3 and A[:, 1] > 0.3
    # idx_x = A.nonzero()
    # print(idx_x)

    # x = A[:, 0]
    # x = x > 0.3
    # idx_x = x.nonzero()
    # A = torch.rand(2, 2)
    # print(A)
    # y = A.ge(0.3)
    # print(y)
    # idx_y = y.nonzero()
    # print(idx_y)

    # B = torch.zeros(2, 2)

    # B[idx_y] = 1
    # print(B)

    # heatmap = torch.rand(5, 5)
    # heatmap_threshold = 0.3
    # mask = heatmap.gt(heatmap_threshold)

    # test2 = torch.tensor([1, 2])

    # test = torch.zeros((5, 5, 2))
    # test[:, :] = test2
    # x = torch.rand(10)
    # print(x)
    # cx1 = x.ge(0.2)
    # cx2 = x.le(0.7)
    # mask_x = torch.all(torch.stack((cx1, cx2), dim=-1), dim=1)
    # print(mask_x)

    # A = torch.rand(1, 10, 10)
    # print(A)
    # test = torch.nn.MaxPool2d(5, stride=1, padding=2)
    # B = test(A)

    # print(B)

    # C = torch.where(A == B, A, torch.tensor(0.0)).reshape(10, 10)
    # print(C)

    # B = B.reshape(10, 10)
    # max_idxs = C.nonzero()
    # first_max_dim = max_idxs[:, 0]
    # sec_max_dim = max_idxs[:, 1]
    # heatmapvals = torch.cat(
    #     [B[x, y].unsqueeze(0) for x, y in zip(first_max_dim, sec_max_dim)]
    # )
    # print(heatmapvals)

    evaluator = torch.load("saved_models/evaluator.pth")
    output_root = "plots"

    result = evaluator.evaluate()
    result_df = result.as_dataframe()
    with open(f"{output_root}/result.csv", "w") as f:
        f.write(result_df.to_csv())

    result.visualize()
    plt.savefig(f"{output_root}/results.png")
    plt.close("all")

    print(result_df)
