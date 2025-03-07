import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), nn.BatchNorm2d(C, affine=affine)),
    'max_pool_3x3': lambda C, stride, affine: nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=1),
                                                            nn.BatchNorm2d(C, affine=affine)),
    'skip_connect': lambda C, stride, affine: Identity(C, affine) if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C1,C2,C3, stride, affine: SepConv(C1, C2, C3, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C1,C2,C3, stride, affine: SepConv(C1, C2, C3, 5, stride, 2, affine),
    'sep_conv_7x7': lambda C1,C2,C3, stride, affine: SepConv(C1, C2, C3, 7, stride, 3, affine),
    'dil_conv_3x3': lambda C1,C2, stride, affine: DilConv(C1, C2, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C1,C2, stride, affine: DilConv(C1, C2, 5, stride, 4, 2, affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'SE_block': lambda C, stride, affine: SE_Block(C),
    'Att_block': lambda C, stride, affine: Att_Block(C),
    'BAM_block': lambda C, stride, affine: BAM_block(C),
    'CBAM_block': lambda C, stride, affine: CBAM_block(C),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # define weight
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *[kernel_size,kernel_size]
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, default_mask=[]):
        if default_mask!=[]:
            # logging.info("default_mask type: "+str(type(default_mask)))
            # try:
            # logging.info(self.weight.device)
            # logging.info(default_mask.device)
            masked_weight = self.weight * default_mask
            # except Exception as e:
            #     logging.info(self.weight.shape)
            #     logging.info(default_mask)
        else:
            masked_weight = self.weight

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
                                              padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out

# class MaskedBatchNorm2d(_BatchNorm):
#     def __init__(
#         self,
#         num_features,
#         affine = True
#     ):
#         super().__init__(
#             num_features=num_features, affine=affine)
#         # self.step = BinaryStep.apply
#
#     def forward(self, x, default_mask=[]):
#         if default_mask!=[]:
#             masked_weight = self.weight * default_mask
#         else:
#             masked_weight = self.weight
#
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked.add_(1)
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)
#
#         out = F.batch_norm(
#             x,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean
#             if not self.training or self.track_running_stats
#             else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             masked_weight,
#             self.bias,
#             bn_training,
#             exponential_average_factor,
#             self.eps,
#         )
#
#         return out


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            MaskedConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            MaskedConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))
        # self.op2 = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
        #                  groups=C_in, bias=False),
        #     nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out, affine=affine))
        # self.c_out = C_out

    def forward(self, x, mask_weight=[]):
        if mask_weight != []:
            out = self.op[0](x)
            out = self.op[1](out, mask_weight[0])
            out = self.op[2](out, mask_weight[1])
            # logging.info("shape: "+str(self.op[3].running_var.shape))
            # logging.info("shape: " + str(self.c_out))
            out = self.op[3](out)
            return out
        # else:
        #     return self.op(x)
        # else:
        #     return self.op2(x)



class SepConv(nn.Module):

    def __init__(self, C_in, C_mid, C_out, kernel_size, stride, padding, affine):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            MaskedConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            MaskedConv2d(C_in, C_mid, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_mid, affine=affine),
            nn.ReLU(inplace=False),
            MaskedConv2d(C_mid, C_mid, kernel_size=kernel_size, stride=1, padding=padding, groups=C_mid, bias=False),
            MaskedConv2d(C_mid, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        # self.op2 = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        #     nn.Conv2d(C_in, C_mid, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_mid, affine=affine),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(C_mid, C_mid, kernel_size=kernel_size, stride=1, padding=padding, groups=C_mid, bias=False),
        #     nn.Conv2d(C_mid, C_out, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(C_out, affine=affine)
        # )

    def forward(self, x, mask_weight=[]):
        if mask_weight!=[]:
        # logging.info(x.shape)
            out = self.op[0](x)
            # logging.info(out.shape)
            out = self.op[1](out, mask_weight[0])
            # logging.info(out.shape)
            # logging.info(self.op[2].weight.shape)
            # logging.info(mask_weight[1].shape)
            out = self.op[2](out, mask_weight[1])
            # logging.info(out.shape)
            out = self.op[3](out)
            out = self.op[4](out)
            # logging.info(self.op[5].weight.shape)
            # logging.info(mask_weight[2].shape)
            out = self.op[5](out, mask_weight[2])
            # logging.info(out.shape)
            # try:
            # logging.info(self.op[6].weight.shape)
            # logging.info(mask_weight[3].shape)
            out = self.op[6](out, mask_weight[3])
            # logging.info(out.shape)
            # except Exception as e:
            out = self.op[7](out)
            return out
        # else:
        #     return self.op(x)
        #     # logging.info(self.op[1].weight)
        #     # logging.info(self.op[3].weight)
        #     return self.op2(x)


class Identity(nn.Module):

    def __init__(self, C_in, affine):
        super(Identity, self).__init__()
        # self.bn = nn.BatchNorm2d(C_in, affine=affine)

    def forward(self, x):
        # return self.bn(x)
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn_2 = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn_2(out)
        return out


