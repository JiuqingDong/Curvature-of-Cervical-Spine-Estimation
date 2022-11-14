import torch
import torch.nn as nn
import torch.nn.functional as F


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_hm_tl = FocalLoss()
        self.L_hm_bl = FocalLoss()
        self.L_hm_tr = FocalLoss()
        self.L_hm_br = FocalLoss()
        self.L_off = RegL1Loss()
        self.L_wh = RegL1Loss()

    def forward(self, pr_decs_en, pr_decs_de, dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3, gt_batch):

        hm_loss_en  = self.L_hm(pr_decs_en['hm'],  gt_batch['hm'])
        wh_loss_en  = self.L_wh(pr_decs_en['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss_en = self.L_off(pr_decs_en['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])

        hm_loss_de  = self.L_hm(pr_decs_de['hm'],  gt_batch['hm'])
        wh_loss_de  = self.L_wh(pr_decs_de['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss_de = self.L_off(pr_decs_de['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])

        hm_loss_de_c4 = self.L_hm(dec_dict_de_c4['hm'], gt_batch['hm'])
        wh_loss_de_c4 = self.L_wh(dec_dict_de_c4['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss_de_c4 = self.L_off(dec_dict_de_c4['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])

        hm_loss_de_c3 = self.L_hm(dec_dict_de_c3['hm'], gt_batch['hm'])
        wh_loss_de_c3 = self.L_wh(dec_dict_de_c3['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss_de_c3 = self.L_off(dec_dict_de_c3['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])

        hm_loss_cat  = self.L_hm(dec_dict_cat['hm'], gt_batch['hm'])
        wh_loss_cat  = self.L_wh(dec_dict_cat['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss_cat = self.L_off(dec_dict_cat['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])

        loss_hm = hm_loss_cat + hm_loss_de + hm_loss_en + hm_loss_de_c4 + hm_loss_de_c3
        wh_loss = wh_loss_en + wh_loss_de + wh_loss_cat + wh_loss_de_c4 + wh_loss_de_c3
        off_loss= off_loss_en + off_loss_de +off_loss_cat + off_loss_de_c4 + off_loss_de_c3

        loss_dec = loss_hm + wh_loss + off_loss

        return loss_dec
