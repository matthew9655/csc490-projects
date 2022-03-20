import torch
from torch import nn

from prediction.model import PredictionModel, PredictionModelConfig


class TwoStageDecoder(nn.Module):
    ## write the code for the new model here
    pass


class TwoStageModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._decoder = TwoStageDecoder(config)
