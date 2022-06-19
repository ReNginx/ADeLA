import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.modality_dataset import ModalityDataset
from dataset.multiview_dataset import MultiViewDataset
from dataset.temporal_dataset import TemporalDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.train_list = kwargs.pop('train_list')
        self.test_list = kwargs.pop('test_list')
        self.batch_size = kwargs.pop('batch_size')
        self.dataset_root = kwargs.pop('dataset_root')
        self.test_dataset_root = kwargs.pop('test_dataset_root', self.dataset_root)
        self.num_workers = kwargs.pop('num_workers')
        self.shuffle = kwargs.pop('shuffle')
        self.kwargs = kwargs

    @staticmethod
    def add_argparse_args(parent_parser, dataset_type=None):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--train_list', type=str, default='train_list.txt')
        parser.add_argument('--test_list', type=str, default='test_list.txt')
        parser.add_argument('--dataset_root', type=str)
        parser.add_argument('--test_dataset_root', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=5)
        parser.add_argument('--shuffle', action='store_true')
        parser.add_argument('--num_workers', type=int, default=8)

        if dataset_type == 'modality':
            parser = ModalityDataset.add_argparse_args(parser)
        elif dataset_type == 'temporal':
            parser = TemporalDataset.add_argparse_args(parser)
        elif dataset_type == 'multiview':
            parser = MultiViewDataset.add_argparse_args(parser)
        else:
            raise NotImplementedError(f'The given dataset {dataset_type} has not been implemented.')
        return parser

    def setup(self, stage=None):
        DatasetClass = self.kwargs['dataset_class_name']
        if stage == None or stage == 'fit':
            self.train = DatasetClass(list_path=self.train_list,
                                      dataset_root=self.dataset_root,
                                      stage='train',
                                      **self.kwargs)
            self.val = DatasetClass(list_path=self.test_list,
                                    dataset_root=self.test_dataset_root,
                                    stage='val',
                                    **self.kwargs)
        else:
            self.test = DatasetClass(list_path=self.test_list,
                                     dataset_root=self.test_dataset_root,
                                     stage='test',
                                     **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                          pin_memory=True)
