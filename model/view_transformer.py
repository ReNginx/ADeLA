import torch
import torch.nn as nn

from utils.model_utils import ModelUtils
from .attention import SpatialTransformerBlock
from .encoder_decoder import Decoderx8, Encoderx8Simp


class ViewTransformer(nn.Module):
    def __init__(self, channel=256, height=384, width=512, n_layers=3, tmp_size=20):
        super(ViewTransformer, self).__init__()
        self.transformers = nn.ModuleList(
            [SpatialTransformerBlock(channel) for _ in range(n_layers)])
        self.encoderk = Encoderx8Simp(3, channel, use_norm=True, use_bias=False)
        self.encoderq = Encoderx8Simp(3, channel, use_norm=True, use_bias=False)
        self.encoderv = Encoderx8Simp(3, channel, use_norm=False, use_bias=False)
        self.decoder = Decoderx8(channel, 3)

    def forward(self, k_frames, q_frames, v_frames):
        k = ModelUtils.get_temporal_feat(k_frames, self.encoderk)
        q = ModelUtils.get_temporal_feat(q_frames, self.encoderq)
        v = ModelUtils.get_temporal_feat(v_frames, self.encoderv)

        k = ModelUtils.extend(k)
        q = ModelUtils.extend(q)

        attn_maps = []
        early_v = []
        for module in self.transformers:
            k, q, v, a = module(k, q, v)
            attn_maps.append(a)
            early_v.append(v)

        n, t, c, h, w = v.shape
        v = v.reshape(n, -1, h, w)
        output = self.decoder(v)

        early_pred = []
        for v in early_v:
            res_v = ModelUtils.get_temporal_feat(v, self.decoder)
            early_pred.append(res_v)

        return output, attn_maps, early_pred


if __name__ == '__main__':
    model = ViewTransformer()
    input1 = torch.rand(2, 10, 3, 384, 512)
    input2 = torch.rand(2, 10, 3, 384, 512)
    input3 = torch.rand(2, 1, 3, 384, 512)
    output = model(input1, input2, input3)
    print(output.shape)
