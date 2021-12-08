from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import time


class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, batch_size=32, epochs=10, scheduler=None, round_to=1):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.loss_history = []
        self.val_history = []
        self.round_to = round_to

    def validate(self):
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for im, cap, view in self.val_data:
                scores = self.model.forward(im, cap)
                num_correct += ((self.round_to * torch.round(scores / self.round_to)) == (self.round_to * torch.round(view.unsqueeze(1) / self.round_to))).sum()
                num_samples += scores.shape[0]
        return num_correct / num_samples

    def train(self):
        # sample minibatch data
        self.model.train()
        for i in range(self.epochs):
            start_t = time.time()
            for j, data in enumerate(self.train_data):
                images, captions, views = data
                loss = self.model.loss(images, captions, views)
                self.optimizer.zero_grad()
                loss.backward()
                self.loss_history.append(loss.item())
                self.optimizer.step()
                if j % 5 == 0:
                    acc = self.validate()
                    self.val_history.append(acc)
                    print('(Iteration {} / {}) loss: {:.4f}, val_acc: {:.6f}'.format(j + 1, len(self.train_data), loss.item(), acc))
                else:
                    print('(Iteration {} / {}) loss: {:.4f}'.format(j + 1, len(self.train_data), loss.item()))
            end_t = time.time()
            print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(i + 1, self.epochs, loss.item(), end_t-start_t))
        if self.scheduler:
            self.scheduler.step()

    def plot_loss(self):
        plt.plot(self.loss_history[5:])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training loss history')
        plt.show()

    def plot_validation(self):
        plt.plot(self.val_history)
        plt.xlabel('Iteration (*5)')
        plt.ylabel('Accuracy')
        plt.title('Validation Set Accuracy')
        plt.show()


def get_dataloader_splits(dataset, batch_size=16, train_percent=0.08, val_percent=0.02, test_percent=0.9):
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
    import pathlib
    from YTPredictor.model.yt_transformers import image_transforms
    from YTPredictor.model.predictor import ViewCountPredictor
    from YTPredictor import ThumbnailDataset
    from torch import optim
    my_model = ViewCountPredictor(1000, 768)
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
    test_data, train_data, val_data = get_dataloader_splits(data, train_percent=0.2, val_percent=0.1, test_percent=0.7)
    learning_rate, lr_decay = 0.05, 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), learning_rate) # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)
    trainer = Trainer(model=my_model, train_data=train_data, val_data=val_data, optimizer=optimizer, scheduler=lr_scheduler, epochs=2)
    trainer.train()
    trainer.plot_loss()
    trainer.plot_validation()
