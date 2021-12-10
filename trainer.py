import os
import pathlib
import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import ThumbnailDataset
from model.predictor import ViewCountPredictor
from model.yt_transformers import image_transforms


class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, batch_size=50, epochs=10, scheduler=None, round_to=5000, checkpoint_dir="./model_checkpoints"):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.loss_history = []
        self.round_to = round_to

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def validate(self):
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for im, cap, subs, view in self.val_data:
                scores = self.model.forward(im, cap, subs)
                print(scores.argmax(dim=1), view.argmax(dim=1))
                num_correct += (scores.argmax(dim=1) == view.argmax(dim=1)).sum()
                num_samples += scores.shape[0]
        return num_correct / num_samples

    def train(self, save_every = 1):
        # sample minibatch data
        self.model.train()
        for i in range(1, self.epochs + 1):
            start_t = time.time()
            for j, data in enumerate(self.train_data):
                images, captions, subs, views = data
                loss = self.model.loss(images, captions, subs, views)
                self.optimizer.zero_grad()
                loss.backward()
                self.loss_history.append(loss.item())
                self.optimizer.step()
                if j % 5 == 0:
                    acc = self.validate()
                    print('(Iteration {} / {}) loss: {:.4f}, val_acc: {:.6f}'.format(j, len(self.train_data), loss.item(), acc))
                else:
                    print('(Iteration {} / {}) loss: {:.4f}'.format(j, len(self.train_data), loss.item()))
            end_t = time.time()
            print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(i, self.epochs, loss.item(), end_t-start_t))

            if i % save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_{i}.pth"))

        if self.scheduler:
            self.scheduler.step()

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training loss history')
        plt.show()


def get_dataloader_splits(dataset, batch_size=50, train_percent=0.25, val_percent=0.25, test_percent=0.5):
    assert train_percent + val_percent + test_percent == 1.
    data_length = len(dataset)
    val_size = int(data_length * val_percent)
    train_size = int(data_length * train_percent)
    test_size = data_length - train_size - val_size
    test, train, val = torch.utils.data.random_split(dataset, [test_size, train_size, val_size])
    train_data = DataLoader(train, batch_size=batch_size)
    val_data = DataLoader(val, batch_size=batch_size)
    test_data = DataLoader(test, batch_size=batch_size)
    return test_data, train_data, val_data


if __name__ == '__main__':

    my_model = ViewCountPredictor()
    data = ThumbnailDataset(root="./youtube_api",
                            transforms=image_transforms['train'])
    test_data, train_data, val_data = get_dataloader_splits(data)
    learning_rate, lr_decay = 1e-2, 0.99
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), learning_rate) # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)
    trainer = Trainer(model=my_model, train_data=train_data, val_data=val_data, optimizer=optimizer, scheduler=lr_scheduler, epochs=100)
    trainer.train()
    trainer.plot_loss()
