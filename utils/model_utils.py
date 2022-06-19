import torch

class ModelUtils:
    @staticmethod
    def get_temporal_feat(frames, model):
        n, t, c, h, w = frames.shape
        frames = frames.reshape(-1, c, h, w)
        feat = model(frames)
        _, c, h, w = feat.shape
        feat = feat.reshape(n, t, c, h, w)
        return feat

    @staticmethod
    def extend(feat):
        n, t, c, h, w = feat.shape
        col_idx = torch.arange(w).float() * torch.ones(h, 1) / w
        row_idx = (torch.arange(h).float() * torch.ones(w, 1)).permute(1, 0) / h
        row_idx = row_idx.to(feat.device)
        col_idx = col_idx.to(feat.device)
        return torch.cat([row_idx * feat, col_idx * feat, feat], dim=2)

