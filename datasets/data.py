import random

import numpy as np
import torch.utils.data as data
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class Reader(data.Dataset):
    def __init__(self, image_list, labels_list=[], transform=None, target_transform=None, use_cache=True,
                 loader=default_loader):

        self.images = image_list
        self.loader = loader

        if len(labels_list) is not 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list
        else:
            self.labels = False

        self.transform = transform
        self.target_transform = target_transform

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx])
            if self.labels:
                target = Image.open(self.labels[idx])
            else:
                target = None
        else:
            img, target = self.cache[idx]

        if self.use_cache:
            self.cache[idx] = (img, target)

        seed = np.random.randint(2147483647)
        random.seed(seed)

        if self.transform is not None:
            img = self.transform(img)

        random.seed(seed)
        if self.labels:
            if self.target_transform is not None:
                target = self.target_transform(target)

        return np.array(img), np.array(target)
