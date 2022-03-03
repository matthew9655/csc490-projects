import torch

from detection.main import overfit

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gammas = torch.arange(0, 2.2, 0.1).to(device)

    for gamma in gammas:
        overfit(data_root="dataset", output_root="gamma_detections", gamma=gamma)
