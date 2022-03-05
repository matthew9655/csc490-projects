import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from detection.main import overfit

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gammas = torch.arange(0, 4, 0.2).to(device)

    losses = np.zeros((gammas.size()[0], 4))

    os.makedirs("focal_loss_test", exist_ok=True)

    for i, gamma in enumerate(gammas):
        loss_metadata = overfit(
            data_root="dataset", output_root="focal_loss_test", gamma=gamma
        )
        losses[i] = torch.tensor(
            [
                loss_metadata.heatmap_loss.item(),
                loss_metadata.offset_loss.item(),
                loss_metadata.size_loss.item(),
                loss_metadata.heading_loss.item(),
            ]
        )

    plt.figure()
    fig, axs = plt.subplots(4, 1)
    plt.subplots_adjust(hspace=0.8)

    gammas = gammas.detach().cpu().numpy()

    titles = ["heatmap loss", "offset loss", "size loss", "heading loss"]

    for i, ax in enumerate(axs):
        ax.plot(gammas, losses[:, i])
        ax.set_title(titles[i])

    plt.savefig("plots/loss_over_gamma")
