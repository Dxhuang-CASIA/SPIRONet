import os
import torch
import numpy as np
import cv2
import random
from scipy import ndimage
from torch.utils.data import Dataset

def normalization(img):
    max_ = img.max()
    min_ = img.min()

    img_new = (img - min_) / (max_ - min_)

    return img_new


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['img'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'img': image, 'label': label}
        return sample


class seg_dataset(Dataset):
    def __init__(self, base_path, name, mode, transform, image_size):
        super().__init__()

        self.base_path = base_path
        self.name = name
        self.mode = mode
        self.transform = transform

        self.img_size = image_size
        self.dataset_path = os.path.join(base_path, name, mode)

        self.index_path = os.listdir(os.path.join(self.dataset_path, 'img'))

    def __getitem__(self, item):
        item_name = self.index_path[item]

        img_path = os.path.join(self.dataset_path, 'img', item_name)

        label_path = os.path.join(self.dataset_path, 'label', item_name)

        img = cv2.imread(img_path, 0)
        label = cv2.imread(label_path, 0)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation = cv2.INTER_NEAREST)
        img = normalization(img)
        label[label > 0] = 1.0

        sample = {'img': img, 'label': label}

        if self.mode == 'train':
            sample = self.transform(sample)
        else:
            sample['img'] = torch.from_numpy(sample['img'].astype(np.float32)).unsqueeze(0)
            sample['label'] = torch.from_numpy(sample['label'].astype(np.float32)).long()

        return sample

    def __len__(self):
        return len(self.index_path)