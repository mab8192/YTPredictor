from numpy.lib.shape_base import array_split
from predictor import ViewCountPredictor as VCP
import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
from yt_transformers import image_transforms
from dataset import ThumbnailDataset
import PIL as Image
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader
import torch.optim as optim

my_model = VCP(1000, 768)
my_model.training = True
data = ThumbnailDataset(root='./youtube_api/',
                            transforms=image_transforms['train'])
trainingdataloader = DataLoader(data, batch_size=30, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
criterion.double()
optimizer = optim.Adam(my_model.regression_model.parameters(), lr= 1e-3, eps= 1e-6)
epoch_size = 1000
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainingdataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs_img, inputs_title, labels = data
        inputs_title = list(inputs_title)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_model(inputs_img, inputs_title)
        loss = criterion(outputs.squeeze().float(), labels.float())
        loss.backward()
        optimizer.step()
        #print("loss:" + str(loss))
        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
        elif i > epoch_size:
            break
my_model.training = False

data = ThumbnailDataset(root='./youtube_api/',
                            transforms=image_transforms['train'])
trainingdataloader = DataLoader(data, batch_size=100, shuffle=True)
testing_data, testing_data2, training_labels = next(iter(trainingdataloader))
score = my_model(testing_data, list(testing_data2))
i = 0
for entry in score:
    print("predicted:" + str(entry) + " actual:" + str(training_labels[i]))
    i+=1
