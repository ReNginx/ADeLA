import argparse
from torch.utils.data import Dataset
from dataset.temporal_dataset import TemporalDataset


class MultiViewDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        target_angles = kwargs.get('target_angles')
        self.stage = kwargs.get('stage')
        self.all_datasets = []
        self.length = 0
        for target_angle in target_angles:
            params = kwargs.copy()
            params['target_angle'] = target_angle
            self.all_datasets.append(TemporalDataset(**params))
            self.length += len(self.all_datasets[-1])

        self.part_length = len(self.all_datasets[-1])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_id = idx // self.part_length
        indataset_idx = idx % self.part_length
        return self.all_datasets[dataset_id][indataset_idx]

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--input_height', type=int, default=384)
        parser.add_argument('--input_width', type=int, default=512)
        parser.add_argument('--source_angle', type=str, default='0')
        parser.add_argument('--frnt_rng', type=int, default=1)
        parser.add_argument('--btom_rng', type=int, default=1)
        parser.add_argument('--btom_offset', type=int, default=0)
        parser.add_argument('--limit', type=int, default=100)
        parser.add_argument('--dataset_class_name', default=MultiViewDataset)
        parser.add_argument('--step', type=int, default=1)
        parser.add_argument('--quant_binsize', nargs='+', type=int, default=[6, 8, 10])
        parser.add_argument('--target_angles', nargs='+', type=str, default=['30'])

        return parser
