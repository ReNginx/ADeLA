import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.metrics import IoU
from torch import nn

from utils.image_utils import ImageUtils
from .view_transformer import ViewTransformer


class ADeLA(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.model = ViewTransformer(height=hparams.input_height, width=hparams.input_width, tmp_size=hparams.frnt_rng,
                                     channel=hparams.n_channel, n_layers=hparams.n_layers)

        self.indv_btom_ious = nn.ModuleDict()
        self.blnk_output = None
        for angles in self.hparams.target_angles:
            self.indv_btom_ious[angles] = IoU(num_classes=40, dist_sync_on_step=False, compute_on_step=False,
                                              reduction='none')
        self.l1 = nn.L1Loss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--init_lr', type=float, default=1e-2)
        parser.add_argument('--patience', type=int, default=2)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
        parser.add_argument('--log_image_every_n_steps', type=int, default=100)
        parser.add_argument('--log_files', type=str, default='log.txt')
        parser.add_argument('--weight_dep', type=float, default=0.05)
        parser.add_argument('--save_samples', action='store_true')
        parser.add_argument('--test_period', type=int, default=10)
        parser.add_argument('--n_channel', type=int, default=128)
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--target_angles', nargs='+', type=str, default=['30'])
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 30, 35], gamma=0.5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

    @property
    def need_test_current_epoch(self):
        return self.current_epoch > 0 and (self.current_epoch + 1) % self.hparams.test_period == 0

    def val_test_step(self, batch, idx, mode='train'):
        angles, path_idx, k_frnt_rgb, v_frnt_rgb, frnt_sem, frnt_sem_ori, \
        q_btom_rgb, v_btom_rgb, btom_sem, btom_sem_ori, multi_frnt_sem, multi_btom_sem = batch

        fuse_frnt_rgb = torch.stack(k_frnt_rgb, dim=1)
        fuse_btom_rgb = torch.stack(q_btom_rgb, dim=1)

        with torch.no_grad():
            calc_test_flag = 'test' in mode or ('val' in mode and self.need_test_current_epoch)

            if calc_test_flag:
                multi_frnt_sem = torch.stack([torch.stack(x, dim=0) for x in multi_frnt_sem], dim=2)
                pred_fuse = []
                if self.blnk_output is None:
                    self.blnk_output, _, _ = self.model(fuse_frnt_rgb, fuse_btom_rgb,
                                                        torch.zeros_like(multi_frnt_sem[0]))

                for i, frnt in enumerate(multi_frnt_sem):
                    if torch.all(frnt < 1e-8):
                        pred_fuse.append(self.blnk_output)
                    else:
                        pred_btom, _, _ = self.model(fuse_frnt_rgb, fuse_btom_rgb, frnt)
                        pred_fuse.append(pred_btom)

                pred_fuse = torch.stack(pred_fuse, dim=0)
                label_l2_multi = ImageUtils.convert_pred_to_label_multi(pred_fuse, p=2)
                confidence = ImageUtils.convert_pred_to_confidence(pred_fuse)
                self.save_img(label_l2_multi, confidence, path_idx, angles[0])
                self.indv_btom_ious[angles[0]](label_l2_multi, btom_sem_ori[0])
        return 0

    def training_step(self, train_batch, batch_idx):
        angles, path_idx, k_frnt_rgb, v_frnt_rgb, frnt_sem, frnt_sem_ori, \
        q_bottom_rgb, v_bottom_rgb, bottom_sem, orig_bottom_sem, multi_front_sem, multi_bottom_sem = train_batch

        fuse_frnt_rgb = torch.stack(k_frnt_rgb, dim=1)
        fuse_btom_rgb = torch.stack(q_bottom_rgb, dim=1)
        fuse_v_frnt_rgb = torch.stack(v_frnt_rgb, dim=1)
        fuse_v_btom_rgb = torch.stack(v_bottom_rgb, dim=1)

        n, t, c, h, w = fuse_frnt_rgb.shape

        pred_btom_rgb, attn_rgb, early_pred_rgb = self.model(fuse_frnt_rgb, fuse_btom_rgb, fuse_v_frnt_rgb)

        early_pred_loss = 0
        rep_v_btom_rgb = fuse_v_btom_rgb.repeat(1, t, 1, 1, 1)
        for i, early_pred in enumerate(early_pred_rgb):
            weight = 2 ** (i - self.hparams.n_layers + 1)
            pred_v_loss = self.l1(early_pred, rep_v_btom_rgb)
            early_pred_loss += weight * pred_v_loss

        img_loss = self.l1(pred_btom_rgb, v_bottom_rgb[0])
        loss = early_pred_loss

        self.log(f'train_img_loss', img_loss)
        self.log(f'train_early_pred_loss', early_pred_loss)
        self.log(f'train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        return self.val_test_step(val_batch, batch_idx, mode='val')

    def test_step(self, test_batch, batch_idx):
        return self.val_test_step(test_batch, batch_idx, mode='test')

    def get_info(self, iou, extra_info):
        mIoU = iou.mean().detach().cpu().float() * 100
        iou = [round(scr * 100, 2) for scr in iou.detach().cpu().tolist()]
        info = f'''
============================     begin  {extra_info}    ===========================
Mean IoU {mIoU:.2f}
per class IoU {iou}
============================     end  {extra_info}    ===========================
                    '''
        return info

    def calc_and_print_result(self, file_name):
        results = [(k, iou.compute()) for k, iou in self.indv_btom_ious.items()]

        if self.global_rank == 0:
            with open(os.path.join(self.logger.log_dir, file_name), 'w') as f:
                for k, res in results:
                    info = self.get_info(res, f'angle {k}')
                    f.write(info)
                    print(info)

        for iou in self.indv_btom_ious.values():
            iou.reset()

    def save_img(self, label, confidence, path_idx, angle):
        path, idx = path_idx
        path, idx = path[0], idx[0]
        transform = torchvision.transforms.ToPILImage()
        label = transform(label.byte())
        confidence = [transform(x) for x in confidence]
        folder = os.path.join(self.logger.log_dir, 'results', path)
        os.makedirs(folder, exist_ok=True)
        label.save(os.path.join(folder, f'pseudo_{idx}.png'))
        color = ImageUtils.colorize(label)
        color.save(os.path.join(folder, f'colorize_{idx}.png'))
        for i, x in enumerate(confidence):
            x.save(os.path.join(folder, f'confid_{idx}_{i}.png'))

    def validation_epoch_end(self, outputs):
        if not self.need_test_current_epoch:
            return

        file_name = f'val@{self.current_epoch:02d}.txt'
        self.calc_and_print_result(file_name)

    def test_epoch_end(self, test_step_output):
        file_name = f'test_res.txt'
        self.calc_and_print_result(file_name)
