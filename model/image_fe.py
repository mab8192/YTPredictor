import torchvision.models as models


# models chosen from here https://pytorch.org/vision/stable/models.html
model_set = {'efficientnet_b2', 'shufflenet', 'regnet_x_32gf', 'efficientnet_b4', 'regnet_y_16gf', 'regnet_y_1_6gf',
             'regnet_x_3_2gf', 'efficientnet_b3', 'efficientnet_b7', 'regnet_y_3_2gf', 'resnet18', 'regnet_x_16gf',
             'regnet_y_8gf', 'vgg16', 'efficientnet_b6', 'resnext50_32x4d', 'efficientnet_b0', 'regnet_y_32gf',
             'googlenet', 'mobilenet_v3_small', 'regnet_y_800mf', 'regnet_x_8gf', 'alexnet', 'densenet',
             'regnet_y_400mf', 'wide_resnet50_2', 'regnet_x_1_6gf', 'mnasnet', 'mobilenet_v3_large', 'efficientnet_b1',
             'squeezenet', 'efficientnet_b5', 'regnet_x_400mf', 'regnet_x_800mf', 'inception', 'mobilenet_v2'}


class ImageFeatureExtractor:
    def __init__(self, model='resnet18') -> None:
        if model not in model_set:
            raise ValueError(f'Model "{model}" is not a valid pre-trained model')
        self.model = getattr(models, model)(pretrained=True)
        self.model.eval()

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(object.__getattribute__(self, 'model'), item)
        except AttributeError:
            return object.__getattribute__(self, item)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    from torchsummary import summary
    from YTPredictor import ThumbnailDataset
    from YTPredictor.model.yt_transformers import data_transforms
    my_model = ImageFeatureExtractor()
    summary(my_model, (3, 224, 224))
    data = ThumbnailDataset(root='/home/corbin/Desktop/school/fall2021/deep/final_project/YTPredictor/youtube_api/',
                            transforms=data_transforms['train'])
    img = data[0][0].reshape((1, *data[0][0].shape))
    print(f'{img.shape=}')
    feature = my_model(img)
    print(f'{feature.shape=}')
