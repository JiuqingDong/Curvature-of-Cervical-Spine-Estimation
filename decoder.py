import torch.nn.functional as F
import numpy as np
import torch
NUM_BOX = 6

class DecDecoder(object):  # test时用到
    def __init__(self, K, conf_thresh):
        self.K = NUM_BOX
        self.conf_thresh = conf_thresh

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)  # 一个样本被网络认为前k个最可能属于的类别

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

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
        feat = feat.permute(0, 2, 3, 1).contiguous()  # （即，预测的output['wh']）维度32*2*96*96----->32*96*96*2
        feat = feat.view(feat.size(0), -1, feat.size(3))
        # 根据ind取出feat中对应的元素;  因为不是dense_wh形式，训练数据中wh的标注batch['wh']的维度是self.max_objs*2，
        # 和预测的输出feat（output['wh']）的维度32*2*96*96不相符，
        # 没有办法进行计算求损失，所以需要根据ind（对象在heatmap图上的索引）取出feat中对应的元素，使其维度和batch['wh']一样，最后维度为32*50*2
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, heat, heat_tl, heat_bl, heat_tr, heat_br, wh, reg, reg_tl, reg_bl, reg_tr, reg_br):  # heat_bl, heat_tr, heat_br,
        # output: num_obj x 7
        # 7: cenx, ceny, w, h, angle, score, cls
        batch, c, height, width = heat.size()
        heat = self._nms(heat)   # [1, 1, 256, 128]
        heat_tl = self._nms(heat_tl)
        heat_bl = self._nms(heat_bl)
        heat_tr = self._nms(heat_tr)
        heat_br = self._nms(heat_br)

        scores, inds, ys, xs = self._topk(heat)
        scores_tl, inds_tl, y_tl, x_tl = self._topk(heat_tl)
        scores_bl, inds_bl, y_bl, x_bl = self._topk(heat_bl)
        scores_tr, inds_tr, y_tr, x_tr = self._topk(heat_tr)
        scores_br, inds_br, y_br, x_br = self._topk(heat_br)

        scores = scores.view(batch, self.K, 1)
        scores_tl = scores_tl.view(batch, self.K, 1)
        scores_bl = scores_bl.view(batch, self.K, 1)
        scores_tr = scores_tr.view(batch, self.K, 1)
        scores_br = scores_br.view(batch, self.K, 1)

        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 2*4)

        reg_tl = self._tranpose_and_gather_feat(reg_tl, inds_tl)
        reg_tl = reg_tl.view(batch, self.K, 2)
        xs_tl = x_tl.view(batch, self.K, 1) + reg_tl[:, :, 0:1]
        ys_tl = y_tl.view(batch, self.K, 1) + reg_tl[:, :, 1:2]

        reg_bl = self._tranpose_and_gather_feat(reg_bl, inds_bl)
        reg_bl = reg_bl.view(batch, self.K, 2)
        xs_bl = x_bl.view(batch, self.K, 1) + reg_bl[:, :, 0:1]
        ys_bl = y_bl.view(batch, self.K, 1) + reg_bl[:, :, 1:2]

        reg_tr = self._tranpose_and_gather_feat(reg_tr, inds_tr)
        reg_tr = reg_tr.view(batch, self.K, 2)
        xs_tr = x_tr.view(batch, self.K, 1) + reg_tr[:, :, 0:1]
        ys_tr = y_tr.view(batch, self.K, 1) + reg_tr[:, :, 1:2]

        reg_br = self._tranpose_and_gather_feat(reg_br, inds_br)
        reg_br = reg_br.view(batch, self.K, 2)
        xs_br = x_br.view(batch, self.K, 1) + reg_br[:, :, 0:1]
        ys_br = y_br.view(batch, self.K, 1) + reg_br[:, :, 1:2]

        # tl_x = (xs - wh[:,:,0:1] + xs_tl) / 2  # 第一位
        # tl_y = (ys - wh[:,:,1:2] + ys_tl) / 2  # 第二位
        # tr_x = (xs - wh[:,:,2:3] + xs_tr) / 2
        # tr_y = (ys - wh[:,:,3:4] + ys_tr) / 2
        # bl_x = (xs - wh[:,:,4:5] + xs_bl) / 2
        # bl_y = (ys - wh[:,:,5:6] + ys_bl) / 2
        # br_x = (xs - wh[:,:,6:7] + xs_br) / 2
        # br_y = (ys - wh[:,:,7:8] + ys_br) / 2

        tl_x = xs - wh[:,:,0:1]
        tl_y = ys - wh[:,:,1:2]
        tr_x = xs - wh[:,:,2:3]
        tr_y = ys - wh[:,:,3:4]
        bl_x = xs - wh[:,:,4:5]
        bl_y = ys - wh[:,:,5:6]
        br_x = xs - wh[:,:,6:7]
        br_y = ys - wh[:,:,7:8]

        # pts = torch.cat([xs, ys,
        #                  xs_tl,ys_tl,
        #                  xs_tr,ys_tr,
        #                  xs_bl,ys_bl,
        #                  xs_br,ys_br,
        #                  scores, scores_tl, scores_bl, scores_tr, scores_br], dim=2).squeeze(0)

        pts = torch.cat([xs, ys,
                         tl_x,tl_y,
                         tr_x,tr_y,
                         bl_x,bl_y,
                         br_x,br_y,
                         scores, scores_tl, scores_bl, scores_tr, scores_br], dim=2).squeeze(0)
        return pts.data.cpu().numpy()
