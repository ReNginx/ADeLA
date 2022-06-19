import os

from PIL import Image
from torch.utils.data import Dataset

from utils.data_utils import DataUtils


class SeqDataset(Dataset):
    def __init__(self, **kwargs):
        list_path = kwargs.get('list_path', '../mp3d_exo_full/train.txt')
        dataset_root = kwargs.get('dataset_root', '../mp3d_exo_full/')
        source = kwargs.get('source', '0')
        limit = kwargs.get('limit', 200)
        rng = kwargs.get('rng', 1)
        offset = kwargs.get('offset', 0)
        typ = kwargs.get('typ', 'rgb')
        ext = 'png' if typ in ('semantic', 'depth') else 'jpg'
        step = kwargs.get('step', 2)
        folders = [x.strip() for x in open(list_path).readlines()]
        self.list = []

        for f in folders:
            path = os.path.join(dataset_root, f, source)
            if not os.path.isdir(path):
                continue
            length = DataUtils.get_video_length(path)
            idxs = list(range(offset, min(limit + offset, length - offset - (rng - 1) * step)))
            for i in idxs:
                buffer = []
                for j in range(0, rng * step, step):
                    buffer.append(os.path.join(dataset_root, f, source, f'{typ}_{i + j:06d}.{ext}'))

                self.list.append(buffer)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        imgs = []
        for path in self.list[idx]:
            img = Image.open(path)
            imgs.append(img)

        return imgs


class PathDataset(Dataset):
    def __init__(self, **kwargs):
        list_path = kwargs.get('list_path', '../mp3d_exo_full/train.txt')
        dataset_root = kwargs.get('dataset_root', '../mp3d_exo_full/')
        source = kwargs.get('source', '0')
        limit = kwargs.get('limit', 200)
        rng = kwargs.get('rng', 1)
        offset = kwargs.get('offset', 0)
        step = kwargs.get('step', 2)
        folders = [x.strip() for x in open(list_path).readlines()]
        self.list = []

        for f in folders:
            path = os.path.join(dataset_root, f, source)
            if not os.path.isdir(path):
                continue
            length = DataUtils.get_video_length(path)
            idxs = list(range(offset, min(limit + offset, length - offset - (rng - 1) * step)))
            for i in idxs:
                self.list.append((os.path.join(f, source), f'{i:06d}'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]
