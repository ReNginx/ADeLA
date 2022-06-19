import argparse

from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentation import CustomColorJitter, ToNumpy, Colorize, RandomInvert, ImageQuantizer, MultiColorize
from dataset.seqdataset import SeqDataset, PathDataset
from utils.data_utils import DataUtils

'''
dataset structure:
mp3d/scene_id/video_id/{up|mid|down}/{rgb|semantic|depth}{idx:06d}.{png|jpg}
each list contains up to video_id, it's up to the dataset to handler the rest (using which view, which modality and 
which indices)
'''


class TemporalDataset(Dataset):
    def __init__(self, **kwargs):
        source_angle = kwargs.get('source_angle', '0')
        self.target_angle = kwargs.get('target_angle', '90')
        crop_size = kwargs.get('crop_size', (384, 512))
        frnt_rng = kwargs.get('frnt_rng', 2)
        btom_rng = kwargs.get('btom_rng', 30)
        btom_offset = kwargs.get('btom_offset', 0)
        step = kwargs.get('step')
        self.stage = kwargs.get('stage')

        frnt_rgb = SeqDataset(source=source_angle, typ='rgb', rng=frnt_rng, offset=0, **kwargs)
        frnt_sem = SeqDataset(source=source_angle, typ='semantic', rng=frnt_rng, offset=0, **kwargs)
        frnt_ori = SeqDataset(source=source_angle, typ='semantic', rng=frnt_rng, offset=0, **kwargs)

        btom_rgb = SeqDataset(source=self.target_angle, typ='rgb', rng=btom_rng,
                              offset=(frnt_rng - 1) * step + btom_offset,
                              **kwargs)
        btom_sem = SeqDataset(source=self.target_angle, typ='semantic', rng=btom_rng,
                              offset=(frnt_rng - 1) * step + btom_offset,
                              **kwargs)
        btom_ori = SeqDataset(source=self.target_angle, typ='semantic', rng=btom_rng,
                              offset=(frnt_rng - 1) * step + btom_offset,
                              **kwargs)
        path = PathDataset(source=self.target_angle, rng=btom_rng, offset=(frnt_rng - 1) * step + btom_offset, **kwargs)

        self.crop_size = crop_size
        self.colorjitter = CustomColorJitter(hue=(-0.3, 0.3))
        self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            ToNumpy(),
            transforms.ToTensor()
        ])

        self.colorizer = Colorize()
        self.multi_colorizer = MultiColorize()
        self.inverter = RandomInvert()
        self.quantizer = ImageQuantizer(permute=True)
        self.collection = [path, frnt_rgb, frnt_sem, frnt_ori, btom_rgb, btom_sem, btom_ori]
        self.length = len(frnt_rgb)

        assert len(frnt_rgb) == len(btom_rgb), f'{len(frnt_rgb)=}, {len(btom_rgb)=}'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        paths, front_rgb, front_semantic, front_orig_sem \
            , bottom_rgb, bottom_semantic, bottom_orig_sem = [dataset[idx] for dataset in self.collection]

        single_fs, single_bs = DataUtils.apply_trans((front_semantic, bottom_semantic), self.colorizer)
        multi_fs, multi_bs = DataUtils.apply_trans((front_semantic, bottom_semantic), self.multi_colorizer)

        if self.stage == 'train':
            k_front_rgb, q_bottom_rgb = self.colorjitter((front_rgb, bottom_rgb))
            v_front_rgb, v_bottom_rgb = self.colorjitter((front_rgb, bottom_rgb))
            v_front_rgb, v_bottom_rgb = self.inverter((v_front_rgb, v_bottom_rgb))
            v_front_rgb, v_bottom_rgb = self.quantizer((v_front_rgb, v_bottom_rgb))
        else:
            k_front_rgb, q_bottom_rgb, v_front_rgb, v_bottom_rgb = front_rgb, bottom_rgb, front_rgb, bottom_rgb

        collection = k_front_rgb, v_front_rgb, single_fs, front_orig_sem, \
                     q_bottom_rgb, v_bottom_rgb, single_bs, bottom_orig_sem, multi_fs, multi_bs
        collection = DataUtils.apply_trans(collection, self.transform)
        return [self.target_angle, paths] + collection

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--input_height', type=int, default=384)
        parser.add_argument('--input_width', type=int, default=512)
        parser.add_argument('--source_angle', type=str, default='0')
        parser.add_argument('--target_angle', type=str, default='90')
        parser.add_argument('--frnt_rng', type=int, default=2)
        parser.add_argument('--btom_rng', type=int, default=30)
        parser.add_argument('--btom_offset', type=int, default=0)
        parser.add_argument('--limit', type=int, default=100)
        parser.add_argument('--dataset_class_name', default=TemporalDataset)
        parser.add_argument('--step', type=int, default=2)
        parser.add_argument('--quant_binsize', type=list, default=[6, 8, 10])

        return parser
