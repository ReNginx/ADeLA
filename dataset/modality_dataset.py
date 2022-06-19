import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentation import ToNumpy
from utils.data_utils import DataUtils

'''
dataset structure:
mp3d/scene_id/video_id/{up|mid|down}/{rgb|semantic|depth}{idx:06d}.{png|jpg}
each list contains up to video_id, it's up to the dataset to handler the rest (using which view, which modality and 
which indices)
'''


class ModalityDataset(Dataset):
    def __init__(self, **kwargs):
        list_path = kwargs.get('list_path', '../mp3d_exo_full/train.txt')
        dataset_root = kwargs.get('dataset_root', '../mp3d_exo_full/')
        label_root = kwargs.get('label_root')
        angle = kwargs.get('angle', '0')
        crop_size = kwargs.get('crop_size', (384, 512))
        self.stage = kwargs.get('stage', 'train')
        folders = [x.strip() for x in open(list_path).readlines()]
        self.list = []

        for folder in folders:
            length = DataUtils.get_video_length(os.path.join(dataset_root, folder, angle))
            for i in range(length):
                target_folder = os.path.join(dataset_root, folder, angle)
                label_folder = os.path.join(label_root, folder, angle)
                self.list.append((target_folder, label_folder, i, 'norm'))

                if self.stage == 'train':
                    target_folder = os.path.join(dataset_root, folder, '0')
                    label_folder = os.path.join(label_root, folder, '0')
                    self.list.append((target_folder, label_folder, i, 'extra'))

        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            ToNumpy(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        folder, label_folder, i, typ = self.list[idx]
        rgb = Image.open(os.path.join(folder, f'rgb_{i:06d}.jpg'))
        hard = Image.open(os.path.join(folder, f'semantic_{i:06d}.png'))

        rgb = self.transform(rgb)
        hard = self.transform(hard).long()

        if self.stage == 'train' and typ == 'norm':
            soft = [Image.open(os.path.join(label_folder, f'confid_{i:06d}_{j}.png')) for j in range(40)]
            soft = [self.transform(s) for s in soft]
            soft = torch.cat(soft, dim=0)
            pesudo = Image.open(os.path.join(label_folder, f'pseudo_{i:06d}.png'))
            pesudo = self.transform(pesudo).long()
            return (rgb, hard, soft, pesudo)
        else:
            pseudo = -torch.ones_like(hard)
            c, h, w = rgb.shape
            soft = -torch.ones((40, h, w))
            return (rgb, hard, soft, pseudo)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--input_height', type=int, default=384)
        parser.add_argument('--input_width', type=int, default=512)
        parser.add_argument('--angle', type=str, default='0')
        parser.add_argument('--dataset_class_name', type=type(ModalityDataset), default=ModalityDataset)
        parser.add_argument('--label_root', type=str)

        return parser
