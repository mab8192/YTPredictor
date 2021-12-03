import torch
import torch.nn as nn
from torch.nn import functional as F

class TitleFeatureExtractor(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

    def forward(self, title):
        return title
