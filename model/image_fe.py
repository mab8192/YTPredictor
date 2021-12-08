import torch
import torchvision.models as models
from torch import nn

# models chosen from here https://pytorch.org/vision/stable/models.html

model_set = {'efficientnet_b2', 'shufflenet', 'regnet_x_32gf', 'efficientnet_b4', 'regnet_y_16gf', 'regnet_y_1_6gf',
             'regnet_x_3_2gf', 'efficientnet_b3', 'efficientnet_b7', 'regnet_y_3_2gf', 'resnet18', 'regnet_x_16gf',
             'regnet_y_8gf', 'vgg16', 'efficientnet_b6', 'resnext50_32x4d', 'efficientnet_b0', 'regnet_y_32gf',
             'googlenet', 'mobilenet_v3_small', 'regnet_y_800mf', 'regnet_x_8gf', 'alexnet', 'densenet',
             'regnet_y_400mf', 'wide_resnet50_2', 'regnet_x_1_6gf', 'mnasnet', 'mobilenet_v3_large', 'efficientnet_b1',
             'squeezenet', 'efficientnet_b5', 'regnet_x_400mf', 'regnet_x_800mf', 'inception', 'mobilenet_v2', 'resnet101'}


class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', fine_tune=False, dtype=torch.double, device=torch.device('cpu')):
        super().__init__()
        self.dtype = dtype
        self.device = device
        assert model_name in model_set, f'Model "{model_name}" is not a valid pre-trained model'
        pretrained_model = getattr(models, model_name)(pretrained=True)
        self.model = nn.Sequential(*list(pretrained_model.children())[: -1]).to(device=self.device, dtype=self.dtype)  # Chop off last classifier layer
        self.model.eval()
        self.output_shape = tuple(self.model(torch.randn((1, 3, 224, 224), device=self.device, dtype=self.dtype)).shape[1:])
        if fine_tune:
            self.model.train()

    def forward(self, x):
        return self.model(x).type(self.dtype)


if __name__ == '__main__':
    import pathlib

    from dataset import ThumbnailDataset
    from model.yt_transformers import image_transforms
    my_model = ImageFeatureExtractor()

    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
    img = data[0][0].reshape((1, *data[0][0].shape))
    print(f'{img.shape=}')
    feature = my_model(img)
    print(f'{feature.shape=}')
