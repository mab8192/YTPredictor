from trainer import get_device, get_dataloader_splits, Trainer, Tester, Predictor
from model.yt_transformers import image_transforms
from model.predictor import ViewCountPredictor
from dataset import ThumbnailDataset
from torch import optim
import pathlib
import torch


def main():
    base = str(pathlib.Path(__file__).parent.resolve())
    dtype = torch.float
    device = get_device()
    my_model = ViewCountPredictor(device=device, dtype=dtype)
    load = input(f'Would you like to load from most recent model? No will overwrite most recent with this one. y/n\n')
    if load.lower() == 'y':
        my_model.load_state_dict(torch.load(base + '/model_checkpoints/model_10.pth'))
        learning_rate, lr_decay = 0.01, 0.5
    else:
        learning_rate, lr_decay = 0.1, 0.99
    data = ThumbnailDataset(root=base + '/youtube_api/', transforms=image_transforms['train'])
    test_data, train_data, val_data = get_dataloader_splits(data, batch_size=32, train_percent=0.7, val_percent=0.15, test_percent=0.15)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), learning_rate) # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)
    trainer = Trainer(model=my_model, train_data=train_data, val_data=val_data, optimizer=optimizer,
                      scheduler=lr_scheduler, epochs=10, device=device, dtype=dtype, show_val=True, max_label=data.max_label)
    my_model = trainer.train()
    tester = Tester(model=my_model, test_data=test_data, device=device, dtype=dtype, max_label=data.max_label, round_to=10_000)
    tester.test()
    pred = Predictor(model=my_model, test_data=test_data, dtype=dtype, device=device, max_label=data.max_label)
    pred.predict()


if __name__ == '__main__':
    main()