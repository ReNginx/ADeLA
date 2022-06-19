import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

from utils.data_utils import DataUtils
from utils.image_utils import ImageUtils


class CustomColorJitter:
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_selection(self):
        def sample(val):
            if type(val) == tuple:
                return random.uniform(*val)
            else:
                return val

        return sample(self.brightness), sample(self.contrast), sample(self.saturation), sample(self.hue)

    def __call__(self, imgs):
        b, c, s, h = self.get_selection()
        imgs = DataUtils.apply_trans(imgs, lambda img: F.adjust_brightness(img, b))
        imgs = DataUtils.apply_trans(imgs, lambda img: F.adjust_contrast(img, c))
        imgs = DataUtils.apply_trans(imgs, lambda img: F.adjust_saturation(img, s))
        imgs = DataUtils.apply_trans(imgs, lambda img: F.adjust_hue(img, h))

        return imgs


class ImageQuantizer:
    def __init__(self, binsize=8, permute=False):
        self.binsize = binsize
        self.toTensor = torchvision.transforms.ToTensor()
        self.toImg = torchvision.transforms.ToPILImage()
        self.permute = permute

    def change(self, img, binsize=None, permute_val=None):
        img = self.toTensor(img)
        img = (img * (binsize - 1)).round()
        if permute_val is not None:
            img = img.long()
            tmp = torch.zeros_like(img)
            for c in range(3):
                for v in range(binsize):
                    tmp[c, img[c] == v] = permute_val[c][v]

            img = tmp
        img = (img + 0.5) / binsize
        img = self.toImg(img)
        return img

    def __call__(self, imgs):
        if self.binsize is list:
            binsize = random.choice(self.binsize)
        else:
            binsize = self.binsize
        if self.permute:
            permute_val = [torch.randperm(binsize) for _ in range(3)]
        else:
            permute_val = None

        imgs = DataUtils.apply_trans(imgs, lambda img: self.change(img, binsize=binsize, permute_val=permute_val))
        return imgs


class RandomInvert:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, imgs):
        if random.uniform(0, 1) < self.prob:
            return imgs
        return DataUtils.apply_trans(imgs, lambda img: F.invert(img))


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


class Colorize(object):
    def __call__(self, img):
        return ImageUtils.colorize(img)


class MultiColorize:
    def __call__(self, img):
        return ImageUtils.colorize_multiple(img)
