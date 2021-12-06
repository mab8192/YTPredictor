from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import time


class Trainer:
    def __init__(self, model, dataset, optimizer, batch_size=32, epochs=10, scheduler=None):
        self.data_length = len(dataset)
        self.loader = DataLoader(dataset, batch_size=batch_size)
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.loss_history = []

    def get_data(self, j):
        # TODO return images, captions using self.loader
        # homework had something like
        # image_data[j * batch_size:(j + 1) * batch_size], \
        # caption_data[j * batch_size:(j + 1) * batch_size]
        return self.loader[j]

    def train(self):
        # sample minibatch data
        self.model.train()
        for i in range(self.epochs):
            start_t = time.time()
            for j, data in enumerate(self.loader):
                images, captions, views = data
                loss = self.model.loss(images, captions, views)
                self.optimizer.zero_grad()
                loss.backward()
                self.loss_history.append(loss.item())
                self.optimizer.step()
                print('(Iteration {} / {}) loss: {:.4f}'.format(j, self.data_length // self.batch_size, loss.item()))
            end_t = time.time()
            print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(i, self.epochs, loss.item(), end_t-start_t))
        if self.scheduler:
            self.scheduler.step()

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training loss history')
        plt.show()


if __name__ == '__main__':
    import pathlib
    from YTPredictor.model.yt_transformers import image_transforms
    from YTPredictor.model.predictor import ViewCountPredictor
    from YTPredictor import ThumbnailDataset
    from torch import optim
    my_model = ViewCountPredictor(1000, 768)
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
    learning_rate, lr_decay = 0.01, 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), learning_rate) # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)
    trainer = Trainer(model=my_model, dataset=data, optimizer=optimizer, scheduler=lr_scheduler, epochs=1)
    trainer.train()
    trainer.plot_loss()