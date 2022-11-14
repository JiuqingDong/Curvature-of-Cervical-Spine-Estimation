import torch
import torch.nn as nn
from models.TransBTS.Transformer import TransformerModel
from models.TransBTS.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from models.TransBTS.Unet_skipconnection import Unet
# from .. import resnet_original

class TransformerBTS_copy(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(TransformerBTS_copy, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv2d(
                512,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )
        # 增加一个resnet

        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)


    def encode(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x0, x1, x2, x3, x4, x = x[0], x[1], x[2], x[3], x[4], x[5]

            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)

        else:
            x0, x1, x2, x3, x4, x = x[0], x[1], x[2], x[3], x[4], x[5]
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)
            print("wrong!!!!!!!!!!!!")
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        # x = x.view(x.size(0), 32, 16, 512)
        # x = x.permute(0, 3, 1, 2).contiguous()


        return x1, x2, x3, x4, x, intmd_x

    def encode2(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x0, x1, x2, x3, x4, x = x[0], x[1], x[2], x[3], x[4], x[5]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        # x = x.view(x.size(0), 32, 16, 512)
        # x = x.permute(0, 3, 1, 2).contiguous()
        return x1, x2, x3, x4, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")
    def decode2(self, x):
        raise NotImplementedError("Should be implemented in child class!!")
    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        e2_1, e2_2, e2_3, e2_4, e2_5, intmd_encoder_outputs = self.encode2(x)
        feat_encode2 = []
        feat_encode2.append(e2_1)
        feat_encode2.append(e2_2)
        feat_encode2.append(e2_3)
        feat_encode2.append(e2_4)

        d2_1, d2_2, d2_3, d2_4, d2_5 = self.decode2(e2_1, e2_2, e2_3, e2_4, e2_5, intmd_encoder_outputs, auxillary_output_layers)
        feat_encode2.append(d2_5)
        # for i in range(len(feat_encode2)):
        #    print("feat_encode2[{}].size(): {}".format(i, feat_encode2[i].size()))
        feat_decode2 = []
        feat_decode2.append(d2_1)
        feat_decode2.append(d2_2)
        feat_decode2.append(d2_3)
        feat_decode2.append(d2_4)
        feat_decode2.append(d2_5)
        #for i in range(len(feat_decode2)):
        #    print("feat_decode2[{}].size(): {}".format(i, feat_decode2[i].size()))

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]
            return feat_encode2

        return feat_encode2

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim * 2 / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class BTS_copy(TransformerBTS_copy):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,

        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BTS_copy, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.Softmax = nn.Softmax(dim=1)

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim, out_channels=self.embedding_dim//2)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//2)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//2, out_channels=self.embedding_dim//4)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//4)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//8)

        self.DeUp1 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//8)
        self.DeBlock1 = DeBlock(in_channels=self.embedding_dim//8)


    def decode2(self, x1, x2, x3, x4, x, intmd_x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []

        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()
        x8 = encoder_outputs[all_keys[0]]
        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y4 = self.DeUp4(x8, x4)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)
        y3 = self.DeUp3(y4, x3)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)
        y2 = self.DeUp2(y3, x2)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y1 = self.DeUp1(y2, x1)  # (1, 16, 128, 128, 128)
        y1 = self.DeBlock1(y1)
        #print("x8.size", x8.size())
        return y1, y2, y3, y4, x8

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def TransBTS_copy(dataset='brats', _conv_repr=True, _pe_type="learned"):
    if dataset.lower() == 'brats':
        img_dim = 128
    num_channels = 4
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = BTS_copy(
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        _, model = TransBTS_copy(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
