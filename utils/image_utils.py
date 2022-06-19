import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import ImageDraw, Image
from habitat_sim.utils.common import d3_40_colors_rgb
from torch import Tensor

from utils.const import colors_rgb_tensor


class ImageUtils:
    @staticmethod
    def make_grid_with_first_image(lst, captions=None, nrow=8):
        lst = [img[0].clip(0, 1) for img in lst]

        if captions is not None:
            res = []
            for img, cap in zip(lst, captions):
                img = torchvision.transforms.ToPILImage()(img).convert('RGB')
                ImageDraw.Draw(img).text((0, 0), cap, fill=(255, 0, 0))
                res.append(torchvision.transforms.ToTensor()(img))
            lst = res
        return torchvision.utils.make_grid(lst, nrow=nrow)

    @staticmethod
    def convert_pred_to_confidence(pred):
        pred = pred.sum(dim=2)
        pred = F.softmax(pred, dim=0)
        return pred

    @staticmethod
    def convert_pred_to_label(pred):
        pred = pred.unsqueeze(2)
        colors_rgb_tensor_tmp = colors_rgb_tensor.to(pred.device)
        pred = (pred - colors_rgb_tensor_tmp).norm(dim=1).argmin(dim=1)
        return pred

    @staticmethod
    def colorize(semantic_obs):
        semantic_obs = np.array(semantic_obs)
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        return semantic_img.convert('RGB')

    @staticmethod
    def overlay_heatmap(image: Tensor, heatmap: Tensor, poi, size=None):
        '''
        :param image: has the shape of B * C * H * W
        :param heatmap: heatmap has the shape of B * h * w * h * w where h, w is the shape of attention feature map.
        we assume the weight along the first dim (columnwise) sums to 1.
        :param poi: point of interest, should not exceed H, W
        :return: return overlay_heatmap of poi on each of input images, has the shape of B * C * H * W
        '''
        image = image.detach()
        heatmap = heatmap.detach()

        _, C, H, W = image.shape
        B, _, _, h, w = heatmap.shape
        heatmap = heatmap.reshape(B, -1, h, w)
        # heatmap = F.interpolate(heatmap, size=(H, W), mode='nearest')
        heatmap = heatmap[:, :, poi[0] * h // H, poi[1] * w // W].reshape(B, 1, h, w)
        heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=True)
        heatmap_np = np.array(heatmap.cpu())

        heatmap_np = np.stack(
            [cv2.applyColorMap((hm[0] * 255.0).astype(np.uint8), cv2.COLORMAP_JET) for hm in heatmap_np])
        image_np = np.array(image.permute(0, 2, 3, 1).cpu())
        overlay_images = np.stack([cv2.addWeighted(hm, 1.0, (img * 255.0).astype(np.uint8), 0.0, 0)
                                   for hm, img in zip(heatmap_np, image_np)])

        res = torch.Tensor(overlay_images).permute(0, 3, 1, 2).float() / 255.0
        if size is not None:
            res = F.interpolate(res, size=size, mode='bilinear', align_corners=True)

        return res

    @staticmethod
    def normalize(image: Tensor, min_val=0, max_val=1):
        image = image.detach()
        image_re = image.reshape(image.shape[0], -1)
        _min, _max = image_re.min(dim=1)[0], image_re.max(dim=1)[0]
        _min = _min.reshape(image.shape[0], 1, 1, 1)
        _max = _max.reshape(image.shape[0], 1, 1, 1)
        image = (image - _min) / (_max - _min) * (max_val - min_val) + min_val
        return image

    @staticmethod
    def convert_pred_to_label_multi(pred, p=2):
        pred = pred.sum(dim=2).argmax(dim=0)
        return pred

    @staticmethod
    def colorize_multiple(semantic_obs):
        semantic_imgs = []

        semantic_obs = torchvision.transforms.ToTensor()(np.array(semantic_obs)) % 40
        transform = torchvision.transforms.ToPILImage()
        for i in range(40):
            semantic_img = torch.zeros_like(semantic_obs).float()
            mask = semantic_obs == i
            semantic_img[mask] = 1.0
            img = transform(semantic_img)
            img = img.convert('RGB')
            semantic_imgs.append(img)
        return semantic_imgs

    @staticmethod
    def labels_to_tensor(labels):
        to_tensor = torchvision.transforms.ToTensor()
        device = labels.device
        imgs = []
        for l in labels:
            img = ImageUtils.colorize(l.detach().cpu())
            img = to_tensor(img).to(device)
            imgs.append(img)
        return torch.stack(imgs)

if __name__ == "__main__":
    input = torch.rand(2, 3, 384, 512)
    heatmap = torch.rand(2, 24, 32, 24, 32)
    res = ImageUtils.overlay_heatmap(input, heatmap, (384 // 2, 512 // 2))
    # torchvision.transforms.ToPILImage()(heatmap[0, :, :, 12, 16]).resize((512, 384), Image.BILINEAR).save('hm.png')
    # torchvision.transforms.ToPILImage()(res[0]).save('test.png')
