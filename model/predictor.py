import torch
import torch.nn as nn
from torch.nn import functional as F

from image_fe import ImageFeatureExtractor
from title_fe import TitleFeatureExtractor


class ViewCountPredictor(nn.Module):
    def __init__(self, n_img_features, n_title_features, img_model='resnet18', title_model='bert') -> None:
        super().__init__()

        self.image_fe = ImageFeatureExtractor(model=img_model)
        self.title_fe = TitleFeatureExtractor(model=title_model)

        # Starting with a single linear layer to check that everything works
        self.regression_model = nn.Linear(n_img_features + n_title_features, 1)

    def forward(self, image, title):
        img_feats = self.image_fe(image)
        title_feats = self.title_fe(title)
        feats = torch.cat((img_feats, title_feats), dim=1)
        return self.regression_model(feats)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    from YTPredictor.model.yt_transformers import image_transforms
    from YTPredictor import ThumbnailDataset
    my_model = ViewCountPredictor(1000, 768)
    data = ThumbnailDataset(root='/home/corbin/Desktop/school/fall2021/deep/final_project/YTPredictor/youtube_api/',
                            transforms=image_transforms['train'])
    feature = my_model(data[0][0].reshape((1, *data[0][0].shape)), data[0][1]['title'])
    print(f'{feature.shape=}')
    print(f'{feature=}')
