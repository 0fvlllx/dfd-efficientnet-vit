"""
Implementation is mainly referenced from /data/maoyuzhe/sagnet
"""

import os
import random

import numpy as np
import pandas as pd
import torchvision

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


class PACS(Dataset):

    def __init__(self, domain, split, transform=None):
        # self.root = os.path.expanduser(root)
        # self.split_dir = os.path.expanduser(split_dir)
        self.domain = domain
        self.split = split
        self.transform = transform
        self.loader = default_loader

        self.preprocess()

    def __getitem__(self, index):
        image_path = self.samples[index][0]
        domain = self.samples[index][1]
        label = self.samples[index][2]

        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, domain, label

    def __len__(self):
        return len(self.samples)

    def preprocess(self):
        split_path = 'paths/{}_{}.txt'.format(self.domain, self.split)
        self.samples = np.genfromtxt(split_path, dtype=str, delimiter=",").tolist()
        self.samples = [(img, int(lbl), int(1) if int(lbl) != 0 else int(0)) for img, lbl in self.samples]

        print('domain: {:6s} split: {:10s} n_images: {:<6d}'
              .format(self.domain, self.split, len(self.samples)))


class FFDataset(Dataset):

    def __init__(self, mode, transform=None):
        self.mode = mode
        self.transform = transform
        self.loader = default_loader
        self.image_path_list = []

        self.collect_image()

    def collect_image(self):
        split_file = 'paths/DFDC_{}_folder.txt'.format(self.mode)
        samples = np.genfromtxt(split_file, dtype=str, delimiter=",").tolist()

        for video_path, label in samples:
            for image in os.listdir(video_path):
                image_path = os.path.join(video_path, image)
                self.image_path_list.append([image_path, int(label),
                                             int(1) if int(label) != 0 else int(0)])

        print('mode: {:10s} n_images: {:<6d}'.format(self.mode,
                                                     len(self.image_path_list)))

        random.shuffle(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index][0]
        method = self.image_path_list[index][1]
        label = self.image_path_list[index][2]

        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, method, label

    def __len__(self):
        return len(self.image_path_list)


#
# if __name__ == '__main__':
#     transform_train = torchvision.transforms.Compose([
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.Resize(224),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406],
#                                          [0.229, 0.224, 0.225])])
#
#     dataset_train = FFDataset(mode='train', transform=transform_train)
#
#     pass


