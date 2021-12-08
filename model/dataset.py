import json
import os
import torch
from PIL import Image
import pathlib


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=lambda x: x) -> None:
        super().__init__()

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "thumbnails"))))
        self.video_data = json.load(open(os.path.join(root, "datafiltered.json")))
        self.clean_data()

    def __getitem__(self, idx):
        video_id = self.imgs[idx][:-4] # Chop off ".jpg"
        img_path = os.path.join(self.root, "thumbnails", self.imgs[idx])
        data = self.video_data[video_id]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img, (data['title'] if data['title'] else data['description']), data.get('viewCount', 0.)

    def __len__(self):
        return len(self.imgs)

    def clean_data(self):
        bad_keys = set()
        vid_keys = {x.split('.jpg')[0] for x in self.imgs}
        max_count = max(self.video_data, key=lambda x: int(self.video_data[x].get('viewCount', 0)))
        max_count = float(self.video_data[max_count]['viewCount'])
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
            self.video_data[key]['viewCount'] = float(self.video_data[key]['viewCount']) / max_count
        for key in bad_keys:
            self.video_data.pop(key)
            self.imgs.remove(f'{key}.jpg')


if __name__ == '__main__':
    from YTPredictor.model.yt_transformers import image_transforms
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
