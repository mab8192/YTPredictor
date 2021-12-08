import os
import time
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import ThumbnailDataset
from model.predictor import ViewCountPredictor
from model.yt_transformers import image_transforms


class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, batch_size=32, epochs=10, scheduler=None, round_to=5000,
                 checkpoint_dir="./model_checkpoints", dtype=torch.double, device=torch.device('cpu'), show_val=True,
                 max_label=1.):
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
        self.device = device
        self.dtype = dtype
        self.show_val = show_val
        self.max_label = max_label

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def validate(self):
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for _, im, cap, view in self.val_data:
                im = im.to(device=self.device, dtype=self.dtype)
                # cap = cap.to(device=self.device, dtype=self.dtype)
                view = view.to(device=self.device, dtype=self.dtype)
                scores = self.model.forward(im, cap)
                num_correct += ((self.round_to * torch.round(self.max_label * scores / self.round_to)) ==
                                (self.round_to * torch.round(self.max_label * view.unsqueeze(1) / self.round_to))).sum()
                num_samples += scores.shape[0]
        return num_correct / num_samples

    def train(self, save_every=1):
        # sample minibatch data
        self.model.train()
        for i in range(1, self.epochs + 1):
            start_t = time.time()
            for j, data in enumerate(self.train_data):
                _, images, captions, views = data
                images = images.to(device=self.device, dtype=self.dtype)
                # captions = captions.to(device=self.device, dtype=self.dtype)
                views = views.to(device=self.device, dtype=self.dtype)
                loss = self.model.loss(images, captions, views)
                self.optimizer.zero_grad()
                loss.backward()
                self.loss_history.append(loss.item())
                self.optimizer.step()
                if j % 5 == 0 and self.show_val:
                    acc = self.validate()
                    self.val_history.append(acc)
                    print('(Iteration {} / {}) loss: {:.4f}, val_acc: {:.6f}'.format(j + 1, len(self.train_data), loss.item(), acc))
                else:
                    print('(Iteration {} / {}) loss: {:.4f}'.format(j + 1, len(self.train_data), loss.item()))
            end_t = time.time()
            print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(i, self.epochs, loss.item(), end_t-start_t))

            if i % save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_{i}.pth"))
            if self.scheduler:
                self.scheduler.step()
        self.plot_loss()
        self.plot_validation()
        return self.model

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


class Tester:
    def __init__(self, model, test_data, round_to=1, dtype=torch.double, device=torch.device('cpu'), max_label=1.):
        self.model = model
        self.test_data = test_data
        self.round_to = round_to
        self.dtype = dtype
        self.device = device
        self.max_label = 1.

    def test(self):
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for _, im, cap, view in self.test_data:
                im = im.to(device=self.device, dtype=self.dtype)
                view = view.to(device=self.device, dtype=self.dtype)
                scores = self.model.forward(im, cap)
                num_correct += ((self.round_to * torch.round(self.max_label * scores / self.round_to)) ==
                                (self.round_to * torch.round(self.max_label * view.unsqueeze(1) / self.round_to))).sum()
                num_samples += scores.shape[0]
        print(f'Accuary: {num_correct / num_samples} ({num_correct} / {num_samples})')
        return num_correct / num_samples


class Predictor:
    def __init__(self, model, test_data, dtype=torch.double, device=torch.device('cpu'), max_label=1.):
        self.model = model
        self.test_data = test_data
        self.dtype = dtype
        self.device = device
        self.max_label = max_label

    def predict(self):
        ret_data = {}
        with torch.no_grad():
            for keys, im, cap, view in self.test_data:
                im = im.to(device=self.device, dtype=self.dtype)
                scores = self.model.forward(im, cap)
                for i, key in enumerate(keys):
                    ret_data[key] = {'predict': scores[i].item() * self.max_label, 'actual': view[i].item() * self.max_label}
        return ret_data


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


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    return device


if __name__ == '__main__':
    dtype = torch.float
    device = get_device()
    my_model = ViewCountPredictor(device=device, dtype=dtype)
    data = ThumbnailDataset(root="./youtube_api/", transforms=image_transforms['train'])
    test_data, train_data, val_data = get_dataloader_splits(data, batch_size=32, train_percent=0.2, val_percent=0.1, test_percent=0.7)
    learning_rate, lr_decay = 0.05, 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), learning_rate) # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)
    trainer = Trainer(model=my_model, train_data=train_data, val_data=val_data, optimizer=optimizer,
                      scheduler=lr_scheduler, epochs=1, device=device, dtype=dtype, show_val=False, max_label=data.max_label)
    my_model = trainer.train()
    tester = Tester(model=my_model, test_data=test_data, device=device, dtype=dtype, max_label=data.max_label)
    tester.test()
