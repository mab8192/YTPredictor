import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.image_fe import ImageFeatureExtractor
from model.title_fe import TitleFeatureExtractor


class ViewCountPredictor(nn.Module):
    def __init__(self, img_model='resnet18', title_model='bert', dtype=torch.double, device='cpu', mse_multiplier=1.) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.mse_multiplier = mse_multiplier
        self.image_fe = ImageFeatureExtractor(model_name=img_model, dtype=dtype, device=self.device)
        self.title_fe = TitleFeatureExtractor(model_name=title_model, dtype=dtype)
        self.input_size = np.prod(self.image_fe.output_shape) + np.prod(self.title_fe.output_shape) + 1
        # we need to reduce dimensions of image and title features to be passable to regression layer
        # as an example, we flatten and setup the linear layer to be the sum of output sizes
        self.flatten = nn.Flatten(start_dim=1)

        # Starting with a single linear layer to check that everything works
        self.regression_model = nn.Sequential(
            nn.BatchNorm1d(self.input_size, dtype=self.dtype, device=self.device),
            nn.Linear(self.input_size, 1, dtype=self.dtype, device=self.device),
            # nn.Linear(self.input_size, 256, dtype=self.dtype, device=self.device),
            # nn.LeakyReLU(negative_slope=0.1),
            # nn.Dropout(0.25),
            # nn.Linear(256, 1, dtype=self.dtype, device=self.device),
            # nn.Dropout(0.25),
            # nn.ReLU()
        )

    def forward(self, image, title, subs):
        img_feats = self.image_fe(image).to(device=self.device, dtype=self.dtype)
        title_feats = self.title_fe(title).to(device=self.device, dtype=self.dtype)
        img_feats = self.flatten(img_feats).to(device=self.device, dtype=self.dtype)
        title_feats = self.flatten(title_feats).to(device=self.device, dtype=self.dtype)
        subs = subs.unsqueeze(1)
        feat = torch.cat((img_feats, title_feats, subs), dim=1).to(device=self.device, dtype=self.dtype)
        return self.regression_model(feat)

    def loss(self, image, title, views, subs):
        feat = self.forward(image, title, subs)
        # mults = self.mse_multiplier * torch.ones_like(feat, dtype=self.dtype, device=self.device)
        # return torch.var(feat - views.unsqueeze(1))
        return F.mse_loss(feat, views.unsqueeze(1))
