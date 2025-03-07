import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: AvgPool(C,stride,affine),
    'max_pool_3x3': lambda C, stride, affine: MaxPool(C,stride,affine),
    'skip_connect': lambda C, stride, affine: Identity(C, affine) if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C,C1,C2, stride, affine: SepConv(C, C1,C2, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C,C1,C2, stride, affine: SepConv(C, C1,C2, 5, stride, 2, affine),
    'sep_conv_7x7': lambda C,C1,C2, stride, affine: SepConv(C, C1,C2, 7, stride, 3, affine),
    'dil_conv_3x3': lambda C,C1, stride, affine: DilConv(C, C1, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C,C1, stride, affine: DilConv(C, C1, 5, stride, 4, 2, affine),
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

def calculate_ratio(mask_k, mask_w):
    sum = 0
    valid = 0
    for i in range(len(mask_k)):
        sum += mask_w[i * 2].shape[0] * mask_w[i * 2].shape[1] * mask_w[i * 2].shape[2] * mask_w[i * 2].shape[3]
        sum += mask_w[i * 2 + 1].shape[0] * mask_w[i * 2 + 1].shape[1] * mask_w[i * 2 + 1].shape[2] * \
               mask_w[i * 2 + 1].shape[3]
        # if i == 0:
        #     logging.info("k0: " + str(mask_k[i]))
        #     logging.info("w0: " + str(mask_w[i * 2]))
        #     logging.info("w1: " + str(mask_w[i * 2 + 1]))
        #     logging.info("w_valid_num0: " + str(mask_w[i * 2].sum()))
        valid += mask_w[i * 2].sum()
        # valid_tmp = 0
        for j in range(len(mask_k[i])):
            if mask_k[i][j] == 1:
                # valid_tmp += mask_w[i * 2 + 1][:, j, :, :].sum()
                valid += mask_w[i * 2 + 1][:, j, :, :].sum()
        # logging.info("w_valid_total1: " + str(mask_w[i * 2 + 1].sum()))
        # logging.info("w_valid_num1: " + str(valid_tmp))
    ratio = valid / sum
    logging.info("ratio: " + str(ratio))
    return ratio

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

class AvgPool(nn.Module):
    def __init__(self, C_in, stride, affine):
        super(AvgPool, self).__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            # nn.BatchNorm2d(C_in, affine=affine)
        )
        self.ratio = 1

    def forward(self, x):
        return self.op(x)

    def reset(self):
        self.op.apply(weight_reset)

class MaxPool(nn.Module):
    def __init__(self, C_in, stride, affine):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            # nn.BatchNorm2d(C_in, affine=affine)
        )
        self.ratio = 1

    def forward(self, x):
        return self.op(x)

    def reset(self):
        self.op.apply(weight_reset)

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        self.ratio = 1
        self.c_out = C_out

    def forward(self, x):
        return self.op(x)

    def reset(self):
        self.op.apply(weight_reset)

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
        self.step = BinaryStep.apply
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, threshold=0,default_mask=None):
        weight_shape = self.weight.shape
        weight = torch.abs(self.weight)
        weight = weight.view(weight_shape[0], -1)
        weight = weight - threshold
        mask = self.step(weight)
        mask = mask.view(weight_shape)
        # logging.info("masks: " + str(mask))
        # logging.info("thre: " + str(threshold))
        if default_mask!=None:
            slc = default_mask==0
            mask[slc] = 0
        ratio = torch.sum(mask) / mask.numel()
        # print("threshold {:3f}".format(self.threshold[0]))
        # print("keep ratio {:.2f}".format(ratio))
        # if ratio <= 0.01:
        #     with torch.no_grad():
        #         self.threshold.data.fill_(0.)
        #     # threshold = self.threshold.view(weight_shape[0], -1)
        #     weight = torch.abs(self.weight)
        #     weight = weight.view(weight_shape[0], -1)
        #     weight = weight - threshold
        #     mask = self.step(weight)
        #     mask = mask.view(weight_shape)
        #     if default_mask != None:
        #         slc = default_mask == 1
        #         default_mask[slc] = mask[slc]
        #         mask = default_mask
        masked_weight = self.weight * mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
                                              padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out, mask

# class MaskedBatchNorm2d(_BatchNorm):
#     def __init__(
#         self,
#         num_features,
#         affine = True
#     ):
#         super().__init__(
#             num_features=num_features, affine=affine)
#         self.step = BinaryStep.apply
#
#     def forward(self, x, threshold=0,default_mask=None):
#         weight_shape = self.weight.shape
#         weight = torch.abs(self.weight)
#         weight = weight.view(weight_shape[0], -1)
#         weight = weight - threshold
#         mask = self.step(weight)
#         mask = mask.view(weight_shape)
#         if default_mask!=None:
#             slc = default_mask == 0
#             mask[slc] = 0
#         ratio = torch.sum(mask) / mask.numel()
#         # print("threshold {:3f}".format(self.threshold[0]))
#         # print("keep ratio {:.2f}".format(ratio))
#         # if ratio <= 0.01:
#         #     with torch.no_grad():
#         #         self.threshold.data.fill_(0.)
#         #     # threshold = self.threshold.view(weight_shape[0], -1)
#         #     weight = torch.abs(self.weight)
#         #     weight = weight.view(weight_shape[0], -1)
#         #     weight = weight - threshold
#         #     mask = self.step(weight)
#         #     mask = mask.view(weight_shape)
#         #     if default_mask != None:
#         #         slc = default_mask == 1
#         #         default_mask[slc] = mask[slc]
#         #         mask = default_mask
#         masked_weight = self.weight * mask
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
#         return out, mask

class SepConv(nn.Module):

    def __init__(self, C_in,C_mid, C_out, kernel_size, stride, padding, affine):
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
        self.step = BinaryStep.apply
        # logging.info(C_out)
        self.C_mid = C_mid

    def get_mask_k(self, param, thre, mask):
        # logging.info("param: "+str(param))
        # logging.info("thre: " + str(thre))
        mask_k = self.step(param - thre)
        # logging.info("mask_k: " + str(mask_k))
        for i in range(len(mask)):
            if mask[i] == 0:
                mask_k[i] = 0
        if mask_k.is_cuda:
            return mask_k
        else:
            return mask_k.cuda()

    def forward(self, x,thre_kernel=0,thre_weight=0,mask_kernel=None,mask_weight=None, kernel_param=[], log=False):
        weight_masks = []
        kernel_masks = []
        if log:
            logging.info("x: "+str(x))
        out = self.op[0](x)
        out,mask = self.op[1](out,thre_weight,mask_weight[0])
        weight_masks.append(mask)
        # if kernel_param!=[]:
        #     try:
        #         m_k = self.get_mask_k(F.sigmoid(kernel_param[0]), thre_kernel, mask_kernel[0])
        #     except Exception as e:
        #         logging.info("Error: "+str(e))
        kernel_masks.append([])
        #     out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[0])).permute(0,3,1,2)
        out, mask = self.op[2](out, thre_weight,mask_weight[1])
        weight_masks.append(mask)
        out = self.op[3](out)
        if kernel_param!=[]:
            m_k = self.get_mask_k(F.sigmoid(kernel_param[1]),thre_kernel,mask_kernel[1])
            kernel_masks.append(m_k)
            # out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[1]).cuda()).permute(0,3,1,2)
            out = (out.permute(0, 2, 3, 1) * m_k * F.sigmoid(kernel_param[1])).permute(0, 3, 1, 2)
        out = self.op[4](out)
        out, mask = self.op[5](out, thre_weight, mask_weight[2])
        weight_masks.append(mask)
        # if kernel_param!=[]:
        #     m_k = self.get_mask_k(F.sigmoid(kernel_param[2]),thre_kernel,mask_kernel[2])
        kernel_masks.append([])
        #     out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[2])).permute(0,3,1,2)
        out, mask = self.op[6](out, thre_weight, mask_weight[3])
        weight_masks.append(mask)
        out = self.op[7](out)
        if kernel_param!=[]:
            m_k = self.get_mask_k(F.sigmoid(kernel_param[3]),thre_kernel,mask_kernel[3])
            kernel_masks.append(m_k)
            # out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[3]).cuda()).permute(0,3,1,2)
            out = (out.permute(0, 2, 3, 1) * m_k * F.sigmoid(kernel_param[3])).permute(0, 3, 1, 2)
        return out, kernel_masks, weight_masks

    def reset(self):
        self.op.apply(weight_reset)

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            MaskedConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=C_in, bias=False),
            MaskedConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))
        self.step = BinaryStep.apply

    def get_mask_k(self, param, thre, mask):
        mask_k = self.step(param - thre)
        for i in range(len(mask)):
            if mask[i] == 0:
                mask_k[i] = 0
        if mask_k.is_cuda:
            return mask_k
        else:
            return mask_k.cuda()

    def forward(self, x, thre_kernel=0,thre_weight=0,mask_kernel=None,mask_weight=None,kernel_param=[],log=False):
        weight_masks, kernel_masks = [], []
        # return self.op(x), kernel_masks,weight_masks
        out = self.op[0](x)
        out, mask = self.op[1](out, thre_weight, mask_weight[0])
        weight_masks.append(mask)
        # if kernel_param!=[]:
        #     m_k = self.get_mask_k(F.sigmoid(kernel_param[0]), thre_kernel, mask_kernel[0])
        kernel_masks.append([])
        #     # logging.info(out.shape)
        #     out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[0])).permute(0,3,1,2)
        out, mask = self.op[2](out, thre_weight, mask_weight[1])
        weight_masks.append(mask)
        out = self.op[3](out)
        if kernel_param!=[]:
            m_k = self.get_mask_k(F.sigmoid(kernel_param[1]), thre_kernel, mask_kernel[1])
            kernel_masks.append(m_k)
            # out = (out.permute(0,2,3,1) * m_k * F.sigmoid(kernel_param[1]).cuda()).permute(0,3,1,2)
            out = (out.permute(0, 2, 3, 1) * m_k * F.sigmoid(kernel_param[1])).permute(0, 3, 1, 2)
        return out, kernel_masks, weight_masks

    def reset(self):
        self.op.apply(weight_reset)

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

class Identity(nn.Module):

    def __init__(self, C_in, affine):
        super(Identity, self).__init__()
        # self.bn = nn.BatchNorm2d(C_in, affine=affine)
        self.ratio = 1

    def forward(self, x):
        # return self.bn(x)
        return x

    def reset(self):
        # self.bn.apply(weight_reset)
        pass


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.ratio = 1

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

    def reset(self):
        pass
        # self.op.apply(weight_reset)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn_2 = nn.BatchNorm2d(C_out, affine=affine)
        self.c_out = C_out

    def forward(self, x, kernel_param=[], mask_kernel=[]):
        x = self.relu(x)
        if mask_kernel != []:
            out1 = self.conv_1(x)
            out1 = (out1.permute(0, 2, 3, 1) * F.sigmoid(kernel_param[0][:kernel_param[0].shape[0] // 2]) * mask_kernel[
                                                                                                                0][:
                                                                                                                   kernel_param[
                                                                                                                       0].shape[
                                                                                                                       0] // 2]).permute(
                0, 3, 1, 2)
            out2 = self.conv_2(x[:, :, 1:, 1:])
            out2 = (out2.permute(0, 2, 3, 1) * F.sigmoid(kernel_param[0][kernel_param[0].shape[0] // 2:]) * mask_kernel[
                                                                                                                0][
                                                                                                            kernel_param[
                                                                                                                0].shape[
                                                                                                                0] // 2:]).permute(
                0, 3, 1, 2)
        else:
            out1 = self.conv_1(x)
            out2 = self.conv_2(x[:, :, 1:, 1:])
        out = torch.cat([out1, out2], dim=1)
        try:
            out = self.bn_2(out)
        except Exception as e:
            logging.info(self.c_out)
            logging.info(self.conv_1.weight.shape)
            logging.info(self.conv_2.weight.shape)
            # logging.info(out1.shape)
            # logging.info(out2.shape)
            # logging.info(self.bn_2.weight.shape)
        if mask_kernel != []:
            return out, mask_kernel
        else:
            return out

    def reset(self):
        self.conv_1.apply(weight_reset)
        self.conv_2.apply(weight_reset)
        self.bn_2.apply(weight_reset)


