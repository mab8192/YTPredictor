import torch
import torch.nn as nn
from torch.nn import functional as F

from image_fe import ImageFeatureExtractor
from title_fe import TitleFeatureExtractor

class ViewCountPredictor(nn.Module):
    def __init__(self, n_img_features, n_title_features) -> None:
        super().__init__()

        self.image_fe = ImageFeatureExtractor(n_img_features)
        self.title_fe = TitleFeatureExtractor(n_title_features)

        self.regression_model = nn.Linear(n_img_features + n_title_features, 1)

    def forward(self, image, title):
        img_feats = self.image_fe(image)
        title_feats = self.title_fe(title)

        feats = torch.cat(img_feats, title_feats)

        views_pred = self.regression_model(feats)
