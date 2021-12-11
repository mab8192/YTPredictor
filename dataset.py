import json
import os
import torch
from PIL import Image
import pathlib
import math
from torchvision.transforms import transforms


DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=DEFAULT_TRANSFORMS) -> None:
        super().__init__()

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "thumbnails"))))
        self.video_data = json.load(open(os.path.join(root, "data.json")))
        self.max_label = None
        self.clean_data()

    def __getitem__(self, idx):
        video_id = self.imgs[idx][:-4] # Chop off ".jpg"
        img_path = os.path.join(self.root, "thumbnails", self.imgs[idx])
        data = self.video_data[video_id]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return video_id, img, (data['title'] if data['title'] else data['description']), data.get('viewCount', 0.), float(data.get('subscriberCount', 0.)) # data.get('viewCount', 0.) #

    def __len__(self):
        return len(self.imgs)

    def clean_data(self):
        bad_keys = set()
        vid_keys = {x.split('.jpg')[0] for x in self.imgs}
        max_count = max(self.video_data, key=lambda x: int(self.video_data[x].get('viewCount', 0)))
        max_count = float(self.video_data[max_count]['viewCount'])
        self.max_label = 1. #math.log10(max_count) # math.sqrt(max_count)
        for key in self.video_data:
            if key not in vid_keys:
                bad_keys.add(key)
                continue
            if 'title' not in self.video_data[key] or not self.video_data[key]['title'] or not self.video_data[key]['title'].isascii():
                bad_keys.add(key)
                continue
            if 'viewCount' not in self.video_data[key]:
                bad_keys.add(key)
                continue
            self.video_data[key]['viewCount'] = math.log10(float(self.video_data[key]['viewCount']) + 1.) / self.max_label
            self.video_data[key]['subscriberCount'] = math.log10(float(self.video_data[key]['subscriberCount']) + 1.)

        for key in bad_keys:
            self.video_data.pop(key)
            self.imgs.remove(f'{key}.jpg')
