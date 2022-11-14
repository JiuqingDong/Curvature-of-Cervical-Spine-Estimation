import torch.nn as nn
import torch
from .model_parts import CombinationModule

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet, self).__init__()
        # 源代码
        # self.dec_c1 = CombinationModule(64, 64, batch_norm=True)  # 跳跃连接 skip connection

        self.dec_c2 = CombinationModule(128, 64, batch_norm=True)  # 跳跃连接 skip connection
        self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
        self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        self.concat4to1 = Concat4to1()
        self.conv_c4 = Conv1x1_c4(in_channels=256)
        self.conv_c3 = Conv1x1_c3(in_channels=128)
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 用0填充m.bias（向量）

    def forward(self, x):

        x_en_result = x[-4]
        c4_combine = self.dec_c4(x[-1], x[-2])  # (2,256,64,32)
        c3_combine = self.dec_c3(c4_combine, x[-3])  # (2,128,128,64)
        c2_combine = self.dec_c2(c3_combine, x[-4])  # (2,64,256,128)

        c4_combine = self.conv_c4(c4_combine)
        c3_combine = self.conv_c3(c3_combine)
        # c2_combine = self.conv_c2(c2_combine)
        x_cat = self.concat4to1(c2_combine, c3_combine, c4_combine)
        # print("c2_combine.size()", c2_combine.size())
        dec_dict_en = {}
        dec_dict_de_c4 = {}
        dec_dict_de_c3 = {}
        dec_dict_de = {}
        dec_dict_cat = {}

        for head in self.heads:
            dec_dict_en[head] = self.__getattr__(head)(x_en_result)
            dec_dict_de[head] = self.__getattr__(head)(c2_combine)
            dec_dict_de_c4[head] = self.__getattr__(head)(c4_combine)
            dec_dict_de_c3[head] = self.__getattr__(head)(c3_combine)
            dec_dict_cat[head] = self.__getattr__(head)(x_cat)
            if 'hm' in head:
                dec_dict_en[head] = torch.sigmoid(dec_dict_en[head])
                dec_dict_de[head] = torch.sigmoid(dec_dict_de[head])
                dec_dict_de_c4[head] = torch.sigmoid(dec_dict_de_c4[head])
                dec_dict_de_c3[head] = torch.sigmoid(dec_dict_de_c3[head])
                dec_dict_cat[head] = torch.sigmoid(dec_dict_cat[head])
            # if 'heat_tl' in head:
            #     dec_dict_en[head] = torch.sigmoid(dec_dict_en[head])
            #     dec_dict_de[head] = torch.sigmoid(dec_dict_de[head])
            #     dec_dict_de_c4[head] = torch.sigmoid(dec_dict_de_c4[head])
            #     dec_dict_de_c3[head] = torch.sigmoid(dec_dict_de_c3[head])
            #     dec_dict_cat[head] = torch.sigmoid(dec_dict_cat[head])
            # if 'heat_bl' in head:
            #     dec_dict_en[head] = torch.sigmoid(dec_dict_en[head])
            #     dec_dict_de[head] = torch.sigmoid(dec_dict_de[head])
            #     dec_dict_de_c4[head] = torch.sigmoid(dec_dict_de_c4[head])
            #     dec_dict_de_c3[head] = torch.sigmoid(dec_dict_de_c3[head])
            #     dec_dict_cat[head] = torch.sigmoid(dec_dict_cat[head])
            # if 'heat_tr' in head:
            #     dec_dict_en[head] = torch.sigmoid(dec_dict_en[head])
            #     dec_dict_de[head] = torch.sigmoid(dec_dict_de[head])
            #     dec_dict_de_c4[head] = torch.sigmoid(dec_dict_de_c4[head])
            #     dec_dict_de_c3[head] = torch.sigmoid(dec_dict_de_c3[head])
            #     dec_dict_cat[head] = torch.sigmoid(dec_dict_cat[head])
            # if 'heat_br' in head:
            #     dec_dict_en[head] = torch.sigmoid(dec_dict_en[head])
            #     dec_dict_de[head] = torch.sigmoid(dec_dict_de[head])
            #     dec_dict_de_c4[head] = torch.sigmoid(dec_dict_de_c4[head])
            #     dec_dict_de_c3[head] = torch.sigmoid(dec_dict_de_c3[head])
            #     dec_dict_cat[head] = torch.sigmoid(dec_dict_cat[head])

        return dec_dict_en, dec_dict_de, dec_dict_cat, dec_dict_de_c4, dec_dict_de_c3,


class Conv1x1_c4(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1_c4, self).__init__()

        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        return x1


class Conv1x1_c3(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1_c3, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)
        return x1


class Concat4to1(nn.Module):
    def __init__(self):
        super(Concat4to1, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, c2,c3,c4):
        con_add = (c2 + c3 + c4)/3
        concat = torch.cat((c2,c3,c4,con_add), dim=1)
        x1 = self.conv1(concat)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        return x1
