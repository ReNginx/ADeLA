''' Spatial-Temporal Transformer Networks
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ModelUtils
from .encoder_decoder import UNetx3


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class FullAttention(nn.Module):
    """
    Compute 'Scaled Dot Product FullAttention
    """

    def forward(self, key, query, value):
        query, key, value = map(lambda x: x.permute(0, 2, 3, 4, 1), [query, key, value])
        n, c, h, w, t = key.shape
        _, _, _, _, t2 = query.shape
        _, c2, _, _, _ = value.shape
        key = key.reshape(n, c, -1).permute(0, 2, 1)
        query = query.reshape(n, c, -1)
        value = value.reshape(n, c2, -1)
        scores = torch.matmul(key, query
                              ) / math.sqrt(query.size(-2))
        p_attn = F.softmax(scores, dim=-2)
        # print(f'Fullattn {value.shape=} {p_attn.shape=}')
        p_val = torch.matmul(value, p_attn)
        p_val = p_val.reshape(n, c2, h, w, t2).permute(0, 4, 1, 2, 3)
        return p_val, p_attn.reshape(n, t, h, w, h, w)


class SpatialAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SpatialAttention
    """

    def forward(self, key, query, value):
        n, t, c, h, w = key.shape
        _, t2, _, _, _ = query.shape
        _, _, c2, _, _ = value.shape
        key = key.reshape(n, t, c, -1).permute(0, 1, 3, 2)
        query = query.reshape(n, t2, c, -1)
        value = value.reshape(n, t, c2, -1)
        scores = torch.matmul(key, query
                              ) / math.sqrt(query.size(-2))
        p_attn = F.softmax(scores, dim=-2)
        # print(f'SpatAttn {value.shape=} {p_attn.shape=}')
        p_val = torch.matmul(value, p_attn)
        p_val = p_val.reshape(n, t, c2, h, w)
        return p_val, p_attn.reshape(n, t, h, w, h, w)


class AttnBlock(nn.Module):
    def __init__(self, d_model, kq_channels=3):
        super().__init__()
        self.query_embedding = UNetx3(kq_channels, kq_channels, use_norm=True)
        self.key_embedding = UNetx3(kq_channels, kq_channels, use_norm=True)

        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0, bias=False)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))


class SpatialTemporalAttnBlock(AttnBlock):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, kq_channels=3):
        super().__init__(d_model, kq_channels=kq_channels)
        self.attention = FullAttention()

    def forward(self, k, q, v):
        n, t, c, h, w = v.shape
        _, t2, _, _, _ = q.shape
        _value = ModelUtils.get_temporal_feat(v, self.value_embedding)
        weighted_val, _attention = self.attention(k, q, _value)
        output = weighted_val.reshape(n * t2, c, h, w)
        v = self.output_linear(output)
        v = v.reshape(n, t2, c, h, w)
        return v, k, q, _attention


class SpatialAttnBlock(AttnBlock):
    def __init__(self, d_model, kq_channels=3):
        super().__init__(d_model, kq_channels=kq_channels)
        self.attention = SpatialAttention()

    def forward(self, k, q, v):
        n, t, c, h, w = v.shape
        _, t2, _, _, _ = q.shape
        _query = ModelUtils.get_temporal_feat(q, self.query_embedding)
        _key = ModelUtils.get_temporal_feat(k, self.key_embedding)
        _value = ModelUtils.get_temporal_feat(v, self.value_embedding)
        v, _attention = self.attention(k, q, _value)

        return v, _key, _query, _attention

    def update_v(self, v_old, v):
        v = v_old + ModelUtils.get_temporal_feat(v, self.output_linear)
        return v


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, output_channels=None, use_bias=False):
        super(FeedForward, self).__init__()
        if output_channels is None:
            output_channels = d_model
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, output_channels, kernel_size=3, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class SpatialTransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, height=48, width=64, tmp_size=4):
        super().__init__()
        self.attention = SpatialAttnBlock(d_model=hidden, kq_channels=hidden * 3)
        self.feed_forward = FeedForward(d_model=hidden, output_channels=hidden)

    def forward(self, k, q, v):
        v2, k, q, a = self.attention(k, q, v)

        v = self.attention.update_v(v, v2)
        v = v + ModelUtils.get_temporal_feat(v, self.feed_forward)

        return k, q, v, a


if __name__ == '__main__':
    pass
