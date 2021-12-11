import numpy as np
import torch
from torch import functional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import LazyLinear

from model.image_fe import ImageFeatureExtractor
from model.title_fe import TitleFeatureExtractor


class ViewCountPredictor(nn.Module):
    def __init__(self, img_model='resnet18', title_model='bert', dtype=torch.double) -> None:
        super().__init__()
        self.dtype = dtype
        self.image_fe = ImageFeatureExtractor(model_name=img_model, dtype=dtype)
        self.image_fe.requires_grad_(False)
        self.title_fe = TitleFeatureExtractor(model_name=title_model, dtype=dtype)
        self.title_fe.requires_grad_(False)

        # we need to reduce dimensions of image and title features to be passable to regression layer
        # as an example, we flatten and setup the linear layer to be the sum of output sizes
        self.flatten = nn.Flatten(start_dim=1)

        # Starting with a single linear layer to check that everything works
        self.regression_model = nn.Sequential(
            nn.Linear(np.prod(self.image_fe.output_shape) + np.prod(self.title_fe.output_shape) + 1, 256, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(256, 256, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(256, 256, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(256, 256, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(256, 64, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(64, 64, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(64, 64, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(64, 64, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(64, 64, dtype=self.dtype),
            nn.LeakyReLU(),
            nn.Linear(64, 1, dtype=self.dtype)
        )

    def forward(self, image, title, subs):
        img_feats = self.image_fe(image)
        title_feats = self.title_fe(title)
        img_feats = self.flatten(img_feats)
        title_feats = self.flatten(title_feats)
        subs = subs.reshape(-1, 1)
        feat = torch.cat((img_feats, title_feats, subs), dim=1)
        return self.regression_model(feat)
    
    def loss(self, image, title, subs, views):
        feat = self.forward(image, title, subs)
        return F.mse_loss(feat, views.reshape(-1, 1)), feat


if __name__ == '__main__':
    import pathlib

    from dataset import ThumbnailDataset
    from model.yt_transformers import image_transforms
    my_model = ViewCountPredictor()
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
    feature = my_model(data[0][0].reshape((1, *data[0][0].shape)), data[0][1])
    print(f'{feature.shape=}')
    print(f'{feature=}')
