import matplotlib.pyplot as plt
import torch

from detection.modules.loss_target import create_heatmap2, create_heatmap3

if __name__ == "__main__":

    H, W = 10, 10

    W_coords, H_coords = torch.arange(W), torch.arange(H)
    H_grid_coords, W_grid_coords = torch.meshgrid(H_coords, W_coords, indexing="ij")
    grid_coords = torch.stack([W_grid_coords, H_grid_coords], dim=-1)
    center = torch.tensor([5, 5])

    heatmap = create_heatmap2(grid_coords, center, 1, 1, 1, 1)

    print(heatmap)

    plt.matshow(heatmap, origin="lower")
    plt.savefig("plots/anisotropic_heatmap.png")
