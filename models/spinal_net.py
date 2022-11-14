from .dec_net import DecNet
from . import resnet_original
import torch.nn as nn
import numpy as np
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS


class SpineNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(SpineNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        channels = [3, 64, 64, 128, 256, 512]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet_original.resnet18(pretrained=pretrained)  # resnet10
        self._, self.trans_model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        self.dec_net = DecNet(heads, final_kernel, head_conv, channels[self.l1])

    def forward(self, x):

        x = self.base_network(x)    # 需要看x的size
        x = self.trans_model(x)     # 出来的size要与之前的保持一致
        dec_dict_en, pr_decs, dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3, = self.dec_net(x)

        return dec_dict_en, pr_decs, dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3,
