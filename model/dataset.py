import json
import os

import numpy as np
import torch
from PIL import Image


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=lambda x: x) -> None:
        super().__init__()

        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "thumbnails"))))
        self.video_data = json.load(open(os.path.join(root, "datafiltered.json")))


    def __getitem__(self, idx):
        video_id = self.imgs[idx][:-4] # Chop off ".jpg"
        img_path = os.path.join(self.root, "thumbnails", self.imgs[idx])
        data = self.video_data[video_id]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img, (data['title'] if data['title'] else data['description']), float(data.get('viewCount', 0.))

    def __len__(self):
        return len(self.imgs)
