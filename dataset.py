import json
import os
import torch
from PIL import Image
import pathlib

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
        self.imgs = list(sorted(os.listdir(os.path.join(root, "thumbnailsFiltered"))))
        self.video_data = json.load(open(os.path.join(root, "datafiltered.json")))
        self.clean_data()

    def __getitem__(self, idx):
        video_id = self.imgs[idx][:-4] # Chop off ".jpg"
        img_path = os.path.join(self.root, "thumbnails", self.imgs[idx])
        data = self.video_data[video_id]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img, (data['title'] if data['title'] else data['description']), int(data.get('subscriberCount', 0)), float(data.get('viewCount', 0.))

    def dupeArray(self, item):
        tensor = torch.zeros(100)
        for t in tensor:
            t = item
        return tensor
        

    def get_views_as_box(self, views):
        
        #Places the viewcounts into boxes of certain value ranges.
        
        if views < 1000:
            views = 0.
        #if views < 5000:
        #    boxed[1] = 1
        elif views < 10000:
            views = 1.
        #elif views < 25000:
            #boxed[2] = 1
        #elif views < 50000:
           # boxed[2] = 1
        elif views < 100000:
            views = 2.
        #elif views < 250000:
            #boxed[4] = 1
        elif views < 500000:
            views = 3.
        elif views < 1000000:
            views = 4.
        #elif views < 5000000:
            #boxed[7] = 1
        elif views < 10000000:
            views = 5.
        else:
            views = 6.
        return views

    def __len__(self):
        return len(self.imgs)

    def clean_data(self):
        bad_keys = set()
        vid_keys = {x.split('.jpg')[0] for x in self.imgs}
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
        for key in bad_keys:
            self.video_data.pop(key)
            self.imgs.remove(f'{key}.jpg')


if __name__ == '__main__':
    from model.yt_transformers import image_transforms
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/',
                            transforms=image_transforms['train'])
