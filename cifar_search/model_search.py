import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from prune import BinaryStep
from operations import *
from torch.autograd import Variable
from genotypes import *
import math
from utils import drop_path


def flops_computation(ci, c, op_id, skip_in_reduction=False,is_attention=False, op=False, weighted=False, mask_k=[], mask_w=[], kernel_param=[]):
    UNIT = 0.000001
    CH = 32
    ratio = c / ci
    if op_id == 1 or op_id == 2:
        # if op:
            # sum = op.weight_masks[0].sum() + op.weight_masks[2].sum()
        # sum = mask_w[0].nelement()+mask_w[2].nelement()
        if weighted:
            sum = mask_w[0].sum()
            sum += (mask_w[1].permute(1, 2, 3, 0)*mask_k[1]*torch.log(1 + mask_k[1].sum() * F.sigmoid(kernel_param[1]) / F.sigmoid(kernel_param[1]).sum())).sum()
            sum += (mask_w[2].permute(1, 2, 3, 0) * mask_k[1] * torch.log(
                1 + mask_k[1].sum() * F.sigmoid(kernel_param[1]) / F.sigmoid(kernel_param[1]).sum())).sum()
            sum += (mask_w[3].permute(1, 2, 3, 0) * mask_k[3] * torch.log(
                1 + mask_k[3].sum() * F.sigmoid(kernel_param[3]) / F.sigmoid(kernel_param[3]).sum())).sum()
        else:
            sum = mask_w[0].nelement()
            sum += mask_w[1][0].nelement() * mask_k[1].sum()
            sum += mask_w[2][0].nelement() * mask_k[1].sum()
            try:
                sum += mask_w[3][0].nelement() * mask_k[3].sum()
            except Exception as e:
                logging.info(mask_w)
        return UNIT * sum

    elif op_id == 3 or op_id == 4:
        if weighted:
            sum = mask_w[0].sum()
            sum += (mask_w[1].permute(1, 2, 3, 0)*mask_k[1]*torch.log(1 + mask_k[1].sum() * F.sigmoid(kernel_param[1]) / F.sigmoid(kernel_param[1]).sum())).sum()
        else:
            try:
                # sum = mask_w[0][0].nelement() * mask_k[0].sum()
                sum = mask_w[0].nelement()
                sum += mask_w[1][0].nelement() * mask_k[1].sum()
            except Exception as e:
                logging.info(mask_w[0].shape)
        return UNIT * sum

    elif op_id == 5 or op_id == 6:
        return 0
    elif op_id == 0:
        if skip_in_reduction:
            return UNIT * c * c
        else:
            return 0
    else:
        return 0


def node_computation(weights_node, eta_min, mask=None, single_edge=False):
    weight_sum = weights_node.sum()
    ops = 0
    if single_edge:
        if mask!=None:
            for m in mask:
                if m==1:
                    ops = ops + 1
        else:
            for w_op in weights_node:
                if w_op / weight_sum > eta_min:
                    ops = ops + 1
    else:
        if mask!=None:
            for mi in mask:
                for mj in mi:
                    if mj==1:
                        ops = ops + 1
        else:
            for edge in weights_node:
                for w_op in edge:
                    if w_op / weight_sum > eta_min:
                        ops = ops + 1
    return weight_sum, ops


class MixedOp(nn.Module):

    def __init__(self, C, stride=1, is_first_stage=False, is_attention=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        all_ops = PRIMITIVES
        for primitive in all_ops:
            try:
                if 'sep_conv' in primitive:
                    op = OPS[primitive](C,C,C, stride, True)
                elif 'dil_conv' in primitive:
                    op = OPS[primitive](C,C, stride, True)
                else:
                    op = OPS[primitive](C, stride, True)
            except Exception as e:
                logging.info(str(primitive))
            self._ops.append(op)
        self.stride = stride
        self.C = C
        self.step = BinaryStep.apply

    def forward(self, x, weights, kernel_param, thre, mask_default, mask_k_default, mask_w_default, drop_prob, eta_min, node_sum, discretization=False, log=False):
        mix_op = 0
        k = 0
        mask = []
        mask_k = []
        mask_w = []
        for w, k_w, op, t, m_d, m_k_d, m_w_d in zip(weights, kernel_param, self._ops, thre, mask_default, mask_k_default, mask_w_default):
            # if w > eta_min * node_sum:
            # m = self.step(w/node_sum-t[0])
            if m_d == 0:
                m = 0
            elif w!=node_sum:
                m = self.step(w - t[0])
                if m==0:
                    node_sum = node_sum-w
            else:
                m = m_d
            mask.append(m)
            m_k,m_w = [], []
            # if mask[k]!=0:
            if m_d != 0:
                if discretization:
                    # if mask[k]!=0 and isinstance(self._ops[k], Zero):
                    if m_d != 0 and isinstance(self._ops[k], Zero):
                        self._ops[k] = OPS[PRIMITIVES[k]](self.C, self.stride, True).cuda()
                    if not isinstance(op, Identity):
                        mix_op += drop_path(op(x), drop_prob)
                    else:
                        mix_op += op(x)
                else:
                    # if mask[k]!=0 and isinstance(self._ops[k], Zero):
                    if m_d != 0 and isinstance(self._ops[k], Zero):
                        self._ops[k] = OPS[PRIMITIVES[k]](self.C, self.stride, True).cuda()
                    if not isinstance(op, Identity):
                        if isinstance(op, SepConv) or isinstance(op, DilConv):
                            try:
                                res = op(x,thre_kernel=t[1],thre_weight=t[2],mask_kernel=m_k_d,mask_weight=m_w_d,kernel_param=k_w)
                            except Exception as e:
                                # logging.info("weight: "+str(op.op[1].weight))
                                logging.info("Error 1: " + str(e))
                                logging.info("k: " + str(k))
                            mix_op_tmp = w * drop_path(res[0], drop_prob) * m
                            mix_op += mix_op_tmp
                            m_k,m_w = res[1], res[2]
                        elif isinstance(op, FactorizedReduce):
                            res = op(x, mask_kernel=m_k_d, kernel_param=k_w)
                            mix_op_tmp = w * drop_path(res[0], drop_prob) * m
                            mix_op += mix_op_tmp
                            m_k = res[1]
                        else:
                            mix_op_tmp = w * drop_path(op(x), drop_prob) * m
                            mix_op += mix_op_tmp
                    else:
                        mix_op_tmp = w * op(x) * m
                        mix_op += mix_op_tmp
            k = k + 1
            mask_k.append(m_k)
            mask_w.append(m_w)
        return mix_op, mask, mask_k, mask_w


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, is_first_stage):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.C = C
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.step = BinaryStep.apply

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier
        # self._ops_att = nn.ModuleList()
        self._ops = nn.ModuleList()
        # for i in range(2):
        #     op = MixedOp(C,is_attention=True)
        #     self._ops_att.append(op)
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, is_first_stage)
                self._ops.append(op)

    def get_mask_k(self, param, thre, mask):
        mask_k = self.step(param - thre)
        for i in range(len(mask)):
            if mask[i] == 0:
                mask_k[i] = 0
        return torch.Tensor(mask_k).cuda()

    def forward(self, s0, s1, weights1, weights2, kernel_param, thre, mask_default, mask_k_default, mask_w_default, drop_prob, eta_min, discretization=False, log=False):
        mask = []
        mask_k, mask_w = [], []
        m_k1 = self.get_mask_k(F.sigmoid(kernel_param[0][0][0]), thre[0][0][1], mask_k_default[0][0][0])
        s0 = self.preprocess0(s0)
        s0 = (s0.permute(0,2,3,1) * F.sigmoid(kernel_param[0][0][0]) * m_k1).permute(0,3,1,2)
        m_k2 = self.get_mask_k(F.sigmoid(kernel_param[0][1][0]), thre[0][1][1], mask_k_default[0][1][0])
        s1 = self.preprocess1(s1)
        s1 = (s1.permute(0,2,3,1) * F.sigmoid(kernel_param[0][1][0]) * m_k2).permute(0,3,1,2)
        mask_k.append([[m_k1],[m_k2]])
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            W = weights2[offset:(offset + len(states))]
            weight_sum = (W*torch.Tensor(mask_default[offset:(offset + len(states))]).cuda()).sum()
            sum = 0
            for j, h in enumerate(states):
                try:
                    s,m,m_k,m_w = self._ops[offset + j](h, weights2[offset + j], kernel_param[offset + j+1], thre[offset + j+1], mask_default[offset + j], mask_k_default[offset + j+1], mask_w_default[offset + j], drop_prob, eta_min, weight_sum, discretization, log)
                    weight_sum = weight_sum - (weights2[offset + j] * torch.Tensor(mask_default[offset + j]).cuda()).sum() + (
                                weights2[offset + j] * torch.Tensor(m).cuda()).sum()
                except Exception as e:
                    logging.info("Error 2: "+str(e))
                    logging.info("edge: "+str(offset + j))
                    s,m,m_k,m_w = self._ops[offset + j](h, weights2[offset + j], kernel_param[offset + j+1], thre[offset + j+1], mask_default[offset + j], mask_k_default[offset + j+1], mask_w_default[offset + j], drop_prob, eta_min, weight_sum, discretization, log)
                sum += s
                mask.append(m)
                mask_k.append(m_k)
                mask_w.append(m_w)
            offset += len(states)
            # print('s', type(s))
            states.append(sum)
        return torch.cat(states[2:], dim=1), mask, mask_k, mask_w

class AuxiliaryHeadCIFAR1(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 32x32"""
    super(AuxiliaryHeadCIFAR1, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(11, stride=3, padding=0, count_include_pad=False),  # image size = 8 x 8
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      # nn.AvgPool2d(8, padding=0, count_include_pad=False),  # image size = 1 x 1
      nn.Conv2d(128, 256, 8, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(256, num_classes))


  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    # x = F.softmax(x)
    return x


class AuxiliaryHeadCIFAR2(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 16x16"""
    super(AuxiliaryHeadCIFAR2, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(7, stride=3, padding=0, count_include_pad=False),  # image size = 4 x 4
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      # nn.AvgPool2d(4, padding=0, count_include_pad=False),  # image size = 1 x 1
      nn.Conv2d(128, 256, 4, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(256, num_classes))

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    # x = F.softmax(x)
    return x

class Network(nn.Module):

    def __init__(self, C, num_classes, layers, eta_min, reg_flops, mu, steps=4, multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self.step = BinaryStep.apply
        self.stage1_end = layers//3-1 #4
        self.stage2_end = 2*layers//3-1   #9
        self._C = C
        self.reg_flops = reg_flops
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.eta_min = eta_min
        self.mu = mu
        C_start = C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        is_first_stage = True
        for i in range(layers):
            if i == self.stage1_end:
                C_stage1 = C_curr * multiplier
            if i in [self.stage1_end + 1, self.stage2_end + 1]:
                C_curr = C_curr * 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, is_first_stage)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == self.stage1_end:
                C_aux1 = C_prev
                is_first_stage = False
            if i == self.stage2_end:
                C_aux2 = C_prev
        self.auxiliary_head1 = AuxiliaryHeadCIFAR1(C_aux1, num_classes)
        self.auxiliary_head2 = AuxiliaryHeadCIFAR2(C_aux2, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()


    def forward(self, input, stage_index=0,update_mask=False, update_partial=False,gradual_discretization=False,discretization_include_current=False, log=False):
        logits_aux1 = None
        logits_aux2 = None
        flops = 0
        C = self._C

        start = s0 = s1 = self.stem(input)

        reg_thre = 0

        for i, cell in enumerate(self.cells):
            weights1 = 0
            weights2 = F.sigmoid(self._arch_parameters[i])
            kernel_param = self._kernel_parameters[i]
            mask = self._masks[i]
            mask_k = self._masks_k[i]
            mask_w = self._masks_w[i]
            thre = self._thresholds[i]
            # weights1 = F.sigmoid(self._arch_parameters[i][0])
            # weights2 = F.sigmoid(self._arch_parameters[i][1])
            if i in [self.stage1_end+1, self.stage2_end+1]:
                C *= 2
                reduction = True
            else:
                reduction = False

            if gradual_discretization:
                if stage_index==1:
                    if discretization_include_current:
                        if i<=self.stage1_end:
                            discretization = True
                        else:
                            discretization = False
                    else:
                        discretization=False
                elif stage_index==2:
                    if i<=self.stage1_end:
                        discretization = True
                    else:
                        if discretization_include_current:
                            if i <= self.stage2_end:
                                discretization = True
                            else:
                                discretization = False
                        else:
                            discretization = False
                elif stage_index==3:
                    if i<=self.stage2_end:
                        discretization = True
                    else:
                        if discretization_include_current:
                            discretization = True
                        else:
                            discretization = False
            else:
                discretization=False

            try:
                s0, [s1, m, m_k, m_w] = s1, cell(torch.Tensor(s0).cuda(), torch.Tensor(s1).cuda(), weights1, weights2, kernel_param, thre, mask, mask_k, mask_w, self.drop_path_prob, self.eta_min, discretization,log=log)
            except Exception as e:
                logging.info("cell: "+str(i))
                s0, [s1, m, m_k, m_w] = s1, cell(s0, s1, weights1, weights2, kernel_param, thre, mask, mask_k, mask_w,
                                                 self.drop_path_prob, self.eta_min, discretization, log=True)

            edge_id = 0
            reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]  # The edges connected to previous and previous previous cells
            for w in weights2:
                edge = 0
                op_id = 0
                for w_op in w:
                    if edge_id in reduction_list and reduction:
                        reduce_skip = True
                    else:
                        reduce_skip = False
                    if edge_id == 0:
                        nodes, ops = node_computation(weights2[0:2], self.eta_min, m[0:2])
                    elif edge_id == 2:
                        nodes, ops = node_computation(weights2[2:5], self.eta_min, m[2:5])
                    elif edge_id == 5:
                        nodes, ops = node_computation(weights2[5:9], self.eta_min, m[2:5])
                    elif edge_id == 9:
                        nodes, ops = node_computation(weights2[9:14], self.eta_min, m[2:5])
                    # if (w_op / nodes) > self.eta_min:
                    if m[edge_id][op_id] != 0:
                        edge += torch.log(1 + ops * w_op / nodes) * (
                                self.reg_flops + self.mu * flops_computation(self._C, C, op_id, reduce_skip,mask_k=m_k[edge_id+1][op_id],mask_w=m_w[edge_id][op_id],weighted=True,
                                                                             kernel_param=kernel_param[edge_id+1][op_id]))
                    op_id += 1
                flops = flops + edge
                edge_id += 1

            if update_mask:
                if update_partial:
                    if stage_index==2:
                        if i>self.stage1_end:
                            self._masks[i] = torch.Tensor(m).cuda()
                            self._masks_k[i] = m_k
                            self._masks_w[i] = m_w
                    elif stage_index==3:
                        if i > self.stage2_end:
                            self._masks[i] = torch.Tensor(m).cuda()
                            self._masks_k[i] = m_k
                            self._masks_w[i] = m_w
                    elif stage_index==0:
                        if i > self.stage2_end:
                            self._masks[i] = torch.Tensor(m).cuda()
                            self._masks_k[i] = m_k
                            self._masks_w[i] = m_w
                else:
                    # slc = self._masks[i].nonzero()
                    self._masks[i] = torch.Tensor(m).cuda()
                    # slc = torch.Tensor(self._masks_k[i]).cuda().nonzero()
                    self._masks_k[i] = m_k
                    # slc = torch.Tensor(self._masks_w[i]).cuda().nonzero()
                    self._masks_w[i] = m_w

            reg_thre += torch.sum(torch.exp(-self._thresholds[i][0][0][1]))+torch.sum(torch.exp(-self._thresholds[i][0][1][1]))
            for j in range(len(self.cells[i]._ops)):
                for k in range(len(self.cells[i]._ops[j]._ops)):
                    if self._masks[i][j][k]==1:
                        if isinstance(self.cells[i]._ops[j]._ops[k], SepConv) or isinstance(self.cells[i]._ops[j]._ops[k], DilConv):
                            for l in range(len(self._thresholds[i][j+1][k])):
                                reg_thre += torch.sum(torch.exp(-self._thresholds[i][j+1][k][l]))
                        else:
                            reg_thre += torch.sum(torch.exp(-self._thresholds[i][j+1][k][0]))

            if i <= self.stage1_end:
                if i == self.stage1_end:
                    stage1_out = s1
                    # logits_aux1 = F.softmax(logits_aux1_raw)
                    flops1 = flops
                    if stage_index == 1:
                        logits_aux1 = self.auxiliary_head1(s1)
                        return logits_aux1, flops1, reg_thre
                    else:
                        logits_aux1 = 0
            elif i <= self.stage2_end:
                if i == self.stage2_end:
                    flops2 = flops-flops1
                    if stage_index==2:
                        logits_aux2 = self.auxiliary_head2(s1)
                        return logits_aux1,logits_aux2,flops2, reg_thre
                    else:
                        logits_aux2 = 0
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        # logits = logits + logits_aux2_raw
        # logits = F.softmax(logits)
        flops3 = flops - flops1 - flops2
        if stage_index==3:
            return logits_aux1,logits_aux2,logits,flops3, reg_thre
        return logits_aux1, logits_aux2, logits, flops, reg_thre

    def update_arch(self):
        for i in range(self._layers):
            for j in range(len(self.cells[i]._ops)):
                for k in range(len(self.cells[i]._ops[j]._ops)):
                    if self._masks[i][j][k]==0 and not isinstance(self.cells[i]._ops[j]._ops[k], Zero):
                        stride = 2 if self.cells[i].reduction and j in [0, 1, 2, 3, 5, 6, 9, 10] else 1
                        self.cells[i]._ops[j]._ops[k] = Zero(stride)

    def update_masks(self,stage_index):
        if stage_index == 1:
            end = self.stage1_end + 1
        elif stage_index == 2:
            end = self.stage2_end + 1
        else:
            end = self._layers
        for i in range(end):
            for j in range(len(self.cells[0]._ops)):
                for k in range(len(self.cells[0]._ops[0]._ops)):
                    if self._masks[i][j][k]==1:
                        self._masks[i][j][k] = self.step(F.sigmoid(self._arch_parameters[i][j][k])-self._thresholds[i][j+1][k][0])
                        # self._masks[i][j][k] = self.step(
                        #     1 - self._thresholds[i][j][k][0])

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def prune_kernel(self, stage_index=0):
        if stage_index == 1:
            cell_end = self.stage1_end + 1
        elif stage_index == 2:
            cell_end = self.stage2_end + 1
        else:
            cell_end = self._layers
        kernel_sum_all = []
        kernel_num_all = []
        active_kernel_id_all = []
        for cell_id in range(cell_end):
            weights2 = F.sigmoid(self._arch_parameters[cell_id])
            n = 2
            start = 0
            edge_id_global = 0
            init_num0 = self._masks_k[cell_id][0][0][0].sum()
            init_num1 = self._masks_k[cell_id][0][1][0].sum()
            if init_num0 % 2 != 0:
                init_num0 += 1
            if init_num1 % 2 != 0:
                init_num1 += 1
            if init_num0 == 0:
                init_num0 += 2
            if init_num1 == 0:
                init_num1 += 2
            kernel_num_step = [torch.Tensor([init_num0])[0], torch.Tensor([init_num1])[0]]
            for step in range(self._steps):
                end = start + n
                W = weights2[start:end]
                kernel_num_max = 0
                sum_tmp = 0
                idx_num = 0
                for w_edge in W:
                    op_id = 0
                    for w_op in w_edge:
                        # if w_op > self.eta_min * weight_sum:
                        if self._masks[cell_id][edge_id_global][op_id] == 1:
                            if isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], SepConv) or isinstance(
                                    self.cells[cell_id]._ops[edge_id_global]._ops[op_id], DilConv):
                                num = self._masks_k[cell_id][edge_id_global+1][op_id][-1].sum()
                                if num >= kernel_num_max:
                                    kernel_num_max = num
                                sum_tmp += num
                                idx_num += 1
                        op_id += 1
                    edge_id_global += 1
                if idx_num!=0:
                    kernel_num_avg = torch.Tensor([int(sum_tmp/idx_num)])[0].cuda()
                else:
                    kernel_num_avg = 0
                start = end
                n += 1
                if kernel_num_max % 2 != 0:
                    kernel_num_max += 1
                if kernel_num_max == 0:
                    kernel_num_max = kernel_num_step[-1]
                kernel_num_step.append(kernel_num_max)
            n = 2
            start = 0
            edge_id_global = 0
            equal = []
            for step in range(self._steps):
                end = start + n
                W = weights2[start:end]
                edge_id = 0
                for w_edge in W:
                    op_id = 0
                    for w_op in w_edge:
                        # if w_op > self.eta_min * weight_sum:
                        if self._masks[cell_id][edge_id_global][op_id] == 1:
                            if not isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id],
                                              SepConv) and not isinstance(
                                    self.cells[cell_id]._ops[edge_id_global]._ops[op_id], DilConv):
                                # logging.info("cell,edge,op: "+str(cell_id)+str(edge_id_global)+str(op_id))
                                # logging.info("step: " + str(step))
                                flag1 = False
                                for ele in equal:
                                    if edge_id in ele:
                                        flag1 = True
                                        flag2 = False
                                        for ele2 in equal:
                                            if step + 2 in ele2:
                                                if ele != ele2:
                                                    # logging.info(str(ele)+" add: " + str(ele2))
                                                    ele.extend(ele2)
                                                    equal.remove(ele2)
                                                    flag2 = True
                                                    break
                                                else:
                                                    flag2 = True
                                                    break
                                        if flag2 == False:
                                            # logging.info(str(ele) + " add: " + str(step + 2))
                                            ele.append(step + 2)
                                        break
                                    elif step + 2 in ele:
                                        flag1 = True
                                        # logging.info("ele: "+str(ele))
                                        # logging.info(str(ele) + " add: " + str(edge_id))
                                        ele.append(edge_id)
                                        break
                                if flag1 == False:
                                    # logging.info("equal add: " + str(edge_id) + str(step + 2))
                                    equal.append([edge_id, step + 2])
                                    # logging.info("equal: " + str(equal))

                        op_id += 1
                    edge_id += 1
                    edge_id_global += 1
                start = end
                n += 1
            # logging.info("kernel_num_step: " + str(kernel_num_step))
            # logging.info("equal: " + str(equal))

            # for ele in equal:
            #     max = 0
            #     num = torch.Tensor([0])[0].cuda()
            #     sum_tmp = torch.Tensor([0])[0].cuda()
            #     for e in ele:
            #         # if e!=0 and e!=1 and kernel_num_step[e]>max:
            #         # if kernel_num_step[e] > max:
            #         #     max = kernel_num_step[e]
            #         sum_tmp += kernel_num_step[e]
            #         num += 1
            #     avg = torch.Tensor([int(sum_tmp / num)])[0].cuda()
            #     if avg % 2 != 0:
            #         avg = avg + 1
            #     for e in ele:
            #         # kernel_num_step[e] = max
            #         kernel_num_step[e] = avg

            num = torch.Tensor([0])[0].cuda()
            sum_tmp = torch.Tensor([0])[0].cuda()
            max = 0
            for e in range(len(kernel_num_step)):
                sum_tmp += kernel_num_step[e]
                num += 1
                if kernel_num_step[e] > max:
                    max = kernel_num_step[e]
            avg = torch.Tensor([int(sum_tmp / num)])[0].cuda()
            if avg % 2 != 0:
                avg = avg + 1
            if cell_id <= self.stage1_end:
                init_C = self._C
            elif cell_id <= self.stage2_end:
                init_C = self._C * 2
            else:
                init_C = self._C * 4
            if avg >= init_C // 2:
                for e in range(len(kernel_num_step)):
                    kernel_num_step[e] = avg
            elif max >= init_C // 2:
            # else:
                for e in range(len(kernel_num_step)):
                    kernel_num_step[e] = max
            else:
                for e in range(len(kernel_num_step)):
                    kernel_num_step[e] = init_C // 2

            # logging.info("kernel_num_step: " + str(kernel_num_step))
            # logging.info("type: " + str(type(kernel_num_step[0])))

            if stage_index == 1 and cell_id == self.stage1_end:
                kernel_num_step[-1] = self.auxiliary_head1.features[2].weight.shape[1] // 4
                # for ele in equal:
                #     if len(kernel_num_step)-1 in ele:
                #         for e in ele:
                #             kernel_num_step[e] = kernel_num_step[-1]
                for e in range(len(kernel_num_step)):
                    kernel_num_step[e] = kernel_num_step[-1]
            if stage_index == 2 and cell_id == self.stage2_end:
                kernel_num_step[-1] = self.auxiliary_head2.features[2].weight.shape[1] // 4
                # for ele in equal:
                #     if len(kernel_num_step)-1 in ele:
                #         for e in ele:
                #             kernel_num_step[e] = kernel_num_step[-1]
                for e in range(len(kernel_num_step)):
                    kernel_num_step[e] = kernel_num_step[-1]
            if cell_id == self._layers - 1:
                for step in range(2, len(kernel_num_step)):
                    kernel_num_step[step] = self.classifier.weight.shape[1] // 4
                    # for ele in equal:
                    #     if step in ele:
                    #         for e in ele:
                    #             kernel_num_step[e] = kernel_num_step[step]
                    for e in range(len(kernel_num_step)):
                        kernel_num_step[e] = kernel_num_step[-1]
            kernel_sum_all.append(kernel_num_step)

            # if cell_id==0:
            #     logging.info(kernel_num_step)

            n = 2
            start = 0
            edge_id_global = 0
            active_kernel_id_cell = []
            kernel_num_cell = []
            if cell_id == 0:
                C_prev_prev = self.cells[0].C_prev_prev
                C_prev = self.cells[0].C_prev
            elif cell_id == 1:
                C_prev_prev = self.cells[1].C_prev_prev
                C_prev = int(sum(kernel_sum_all[0][2:]))
            else:
                C_prev_prev = int(sum(kernel_sum_all[cell_id - 2][2:]))
                C_prev = int(sum(kernel_sum_all[cell_id - 1][2:]))
            if self.cells[cell_id].reduction_prev:
                list = [0 for i in range(len(self.cells[cell_id].preprocess0.conv_1.weight[0]))]
                for out_idx in range(len(self.cells[cell_id].preprocess0.conv_1.weight)):
                    for in_idx in range(len(self.cells[cell_id].preprocess0.conv_1.weight[out_idx])):
                        list[in_idx] += abs(self.cells[cell_id].preprocess0.conv_1.weight[out_idx][in_idx]).sum()
                        list[in_idx] += abs(self.cells[cell_id].preprocess0.conv_2.weight[out_idx][in_idx]).sum()
                weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
                idx1_0 = idx1_3[:C_prev_prev].tolist()
                idx1_0.sort(reverse=False)

                idx1_1_tmp1 = self.flatten(
                    torch.nonzero(self._masks_k[cell_id][0][0][0][:len(self._masks_k[cell_id][0][0][0])//2]).tolist())
                idx1_2_tmp1 = self.flatten(
                    torch.nonzero(self._masks_k[cell_id][0][0][0][:len(self._masks_k[cell_id][0][0][0])//2] == 0).tolist())
                if kernel_num_step[0] // 2 > len(idx1_1_tmp1):
                    weight, idx1_3_tmp1 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_2_tmp1]),
                        descending=True)
                    idx1_1_tmp1.extend([idx1_2_tmp1[i] for i in idx1_3_tmp1[:int(kernel_num_step[0]//2 - len(idx1_1_tmp1))]])
                else:
                    weight, idx1_3_tmp1 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_1_tmp1]),
                        descending=True)
                    idx1_1_tmp1 = [idx1_1_tmp1[i] for i in idx1_3_tmp1[:int(kernel_num_step[0] // 2)]]
                idx1_1_tmp1.sort(reverse=False)
                idx1_1_tmp2 = self.flatten(
                    torch.nonzero(self._masks_k[cell_id][0][0][0][len(self._masks_k[cell_id][0][0][0]) // 2:]).tolist())
                idx1_2_tmp2 = self.flatten(
                    torch.nonzero(
                        self._masks_k[cell_id][0][0][0][len(self._masks_k[cell_id][0][0][0]) // 2:] == 0).tolist())
                if kernel_num_step[0] // 2 > len(idx1_1_tmp2):
                    weight, idx1_3_tmp2 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_2_tmp2]),
                        descending=True)
                    idx1_1_tmp2.extend(
                        [idx1_2_tmp2[i] for i in idx1_3_tmp2[:int(kernel_num_step[0] // 2 - len(idx1_1_tmp2))]])
                    # logging.info("1:" + str(len(idx1_1_tmp2)))
                    # logging.info("2:" + str(len(idx1_3_tmp2[:int(kernel_num_step[0] // 2 - len(idx1_1_tmp2))])))
                else:
                    weight, idx1_3_tmp2 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_1_tmp2]),
                        descending=True)
                    idx1_1_tmp2 = [idx1_1_tmp2[i] for i in idx1_3_tmp2[:int(kernel_num_step[0] // 2)]]
                    # logging.info("2:"+str(len(idx1_3_tmp2[:int(kernel_num_step[0] // 2)])))
                idx1_1_tmp2.sort(reverse=False)
                idx1_1 = []
                idx1_1.extend(idx1_1_tmp1)
                idx1_1.extend(idx1_1_tmp2)
                # logging.info(len(idx1_1_tmp1))
            else:
                list = [0 for i in range(len(self.cells[cell_id].preprocess0.op[1].weight[0]))]
                for out_idx in range(len(self.cells[cell_id].preprocess0.op[1].weight)):
                    for in_idx in range(len(self.cells[cell_id].preprocess0.op[1].weight[out_idx])):
                        list[in_idx] += abs(self.cells[cell_id].preprocess0.op[1].weight[out_idx][in_idx]).sum()
                weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
                idx1_0 = idx1_3[:C_prev_prev].tolist()
                idx1_0.sort(reverse=False)

                idx1_1 = self.flatten(
                    torch.nonzero(self._masks_k[cell_id][0][0][0]).tolist())
                idx1_2 = self.flatten(
                    torch.nonzero(self._masks_k[cell_id][0][0][0] == 0).tolist())
                if kernel_num_step[0] > len(idx1_1):
                    weight, idx1_3 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_2]),
                        descending=True)
                    idx1_1.extend([idx1_2[i] for i in idx1_3[:int(kernel_num_step[0] - len(idx1_1))]])
                else:
                    weight, idx1_3 = torch.sort(
                        F.sigmoid(self._kernel_parameters[cell_id][0][0][0][idx1_1]),
                        descending=True)
                    idx1_1 = [idx1_1[i] for i in idx1_3[:int(kernel_num_step[0])]]
                idx1_1.sort(reverse=False)
            # logging.info("idx1: "+str(idx1_1))
            list = [0 for i in range(len(self.cells[cell_id].preprocess1.op[1].weight[0]))]
            for out_idx in range(len(self.cells[cell_id].preprocess1.op[1].weight)):
                for in_idx in range(len(self.cells[cell_id].preprocess1.op[1].weight[out_idx])):
                    list[in_idx] += abs(self.cells[cell_id].preprocess1.op[1].weight[out_idx][in_idx]).sum()
            weight, idx2_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
            idx2_0 = idx2_3[:C_prev].tolist()
            idx2_0.sort(reverse=False)
            # if cell_id==1:
            #     logging.info("weight: " + str(len(self.cells[cell_id].preprocess1.op[1].weight.shape)))
            #     logging.info("all: " + str(len(idx2_3)))
            #     logging.info("c_prev: "+str(len(C_prev)))

            idx2_1 = self.flatten(
                torch.nonzero(self._masks_k[cell_id][0][1][0]).tolist())
            idx2_2 = self.flatten(
                torch.nonzero(self._masks_k[cell_id][0][1][0] == 0).tolist())
            if kernel_num_step[1] > len(idx2_1):
                weight, idx2_3 = torch.sort(
                    F.sigmoid(self._kernel_parameters[cell_id][0][1][0][idx2_2]),
                    descending=True)
                idx2_1.extend([idx2_2[i] for i in idx2_3[:int(kernel_num_step[1] - len(idx2_1))]])
            else:
                weight, idx2_3 = torch.sort(
                    F.sigmoid(self._kernel_parameters[cell_id][0][1][0][idx2_1]),
                    descending=True)
                idx2_1 = [idx2_1[i] for i in idx2_3[:int(kernel_num_step[1])]]
            idx2_1.sort(reverse=False)
            active_kernel_id_cell.append([[idx1_0,idx1_1],[idx2_0,idx2_1]])
            for step in range(self._steps):
                end = start + n
                W = weights2[start:end]
                weight_sum = W.sum()
                edge_id = 0
                for w_edge in W:
                    op_id = 0
                    active_kernel_id_edge = []
                    kernel_num_edge = []
                    for w_op in w_edge:
                        active_kernel_id_op = []
                        kernel_num_op = []
                        # if w_op > self.eta_min * weight_sum:
                        if self._masks[cell_id][edge_id_global][op_id] == 1:
                            kernel_num_op.append(kernel_num_step[edge_id])
                            if isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], SepConv) or isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], DilConv):
                                # num = []
                                for l in range(len(self._masks_k[cell_id][edge_id_global + 1][op_id])):
                                    # num = self._masks_k[cell_id][edge_id_global+1][op_id][l].sum()
                                    if l==0:
                                        num = kernel_num_step[edge_id]
                                    elif l==2:
                                        num = self._masks_k[cell_id][edge_id_global + 1][op_id][1].sum()
                                    else:
                                        num = self._masks_k[cell_id][edge_id_global+1][op_id][l].sum()
                                    if num == 0:
                                        num = torch.Tensor([num + 2])[0]
                                    # num1 = self._masks_k[cell_id][edge_id_global + 1][op_id][2 * l+1].sum()
                                    # if num1 == 0:
                                    #     num1 = torch.Tensor([num1 + 2])[0]
                                    # if num0>num1:
                                    #     num_tmp = num0
                                    # else:
                                    #     num_tmp = num1
                                    if num % 2 != 0:
                                        num += 1
                                    # num.append(num_tmp)
                                    kernel_num_op.append(num)
                                kernel_num_op[-1] = kernel_num_step[step+2]
                                if len(kernel_num_op)>3:
                                    # if cell_id == 0 and edge_id_global == 7 and op_id == 1:
                                    #     logging.info("kernel: " + str(kernel_num_op[0]))
                                    #     logging.info("kernel: " + str(kernel_num_op[-1]))
                                    #     logging.info("kernel: " + str((kernel_num_op[0] + kernel_num_op[-1]) // 4))
                                    if kernel_num_op[2] < (kernel_num_op[0] + kernel_num_op[-1]) // 2:
                                        num = (kernel_num_op[0] + kernel_num_op[-1]) // 2
                                        if num % 2 != 0:
                                            num += 1
                                        kernel_num_op[2] = num
                                        kernel_num_op[3] = num
                                # for l in range(len(kernel_num_op)//2):
                                #     if kernel_num_op[2*l]>kernel_num_op[2*l+1]:
                                #         kernel_num_op[2 * l + 1] = kernel_num_op[2*l]
                                #     else:
                                #         kernel_num_op[2 * l] = kernel_num_op[2 * l + 1]
                                # logging.info(kernel_num_op)
                                # if kernel_num_step[step+2] > kernel_num_op[-1]:
                                for l in range(len(self._masks_k[cell_id][edge_id_global + 1][op_id])):
                                    if l==0:
                                        list = [self._masks_w[cell_id][edge_id_global][op_id][0][k].sum() for k in range(len(self._masks_w[cell_id][edge_id_global][op_id][0]))]
                                        # list = [0 for i in range(len(self._masks_w[cell_id][edge_id_global][op_id][0][0]))]
                                        # for out_idx in range(len(self._masks_w[cell_id][edge_id_global][op_id][0])):
                                        #     for in_idx in range(
                                        #             len(self._masks_w[cell_id][edge_id_global][op_id][0][out_idx])):
                                        #         list[in_idx] += abs(
                                        #             self._masks_w[cell_id][edge_id_global][op_id][0][out_idx][in_idx]).sum()
                                        weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(),descending=True)
                                        idx1_1 = idx1_3[:int(kernel_num_op[1])].tolist()
                                        # if cell_id==0 and edge_id_global==0 and op_id==1:
                                        #     logging.info("active: "+str(len(idx1_1)))
                                            # logging.info("kernel_num_op: " + str(kernel_num_op[1]))

                                    elif l==1 or l==3:
                                        idx1_1 = self.flatten(
                                            torch.nonzero(self._masks_k[cell_id][edge_id_global+1][op_id][l]).tolist())
                                        idx1_2 = self.flatten(
                                            torch.nonzero(self._masks_k[cell_id][edge_id_global+1][op_id][l] == 0).tolist())
                                        if kernel_num_op[l + 1] > len(idx1_1):
                                            weight, idx1_3 = torch.sort(
                                                F.sigmoid(self._kernel_parameters[cell_id][edge_id_global+1][op_id][l][idx1_2]),
                                                descending=True)
                                            idx1_1.extend([idx1_2[i] for i in idx1_3[:int(kernel_num_op[l+1] - len(idx1_1))]])
                                        else:
                                            weight, idx1_3 = torch.sort(
                                                F.sigmoid(
                                                    self._kernel_parameters[cell_id][edge_id_global + 1][op_id][l][idx1_1]),
                                                descending=True)
                                            idx1_1 = [idx1_1[i] for i in idx1_3[:int(kernel_num_op[l + 1])]]
                                    elif l==2:
                                        idx1_1 = active_kernel_id_op[1]
                                    idx1_1.sort(reverse=False)
                                    active_kernel_id_op.append(idx1_1)
                            elif isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], FactorizedReduce):
                                list = [0 for i in range(len(self.cells[cell_id]._ops[edge_id_global]._ops[op_id].conv_1.weight[0]))]
                                for out_idx in range(len(self.cells[cell_id]._ops[edge_id_global]._ops[op_id].conv_1.weight)):
                                    for in_idx in range(len(self.cells[cell_id]._ops[edge_id_global]._ops[op_id].conv_1.weight[out_idx])):
                                        list[in_idx] += abs(
                                            self.cells[cell_id]._ops[edge_id_global]._ops[op_id].conv_1.weight[out_idx][in_idx]).sum()
                                        list[in_idx] += abs(
                                            self.cells[cell_id]._ops[edge_id_global]._ops[op_id].conv_2.weight[out_idx][in_idx]).sum()
                                weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
                                idx1_0 = idx1_3[:int(kernel_num_step[edge_id])].tolist()
                                idx1_0.sort(reverse=False)

                                idx1_1_tmp1 = self.flatten(
                                    torch.nonzero(self._masks_k[cell_id][edge_id_global + 1][op_id][0][
                                                  :len(self._masks_k[cell_id][edge_id_global + 1][op_id][0]) // 2]).tolist())
                                idx1_2_tmp1 = self.flatten(
                                    torch.nonzero(self._masks_k[cell_id][edge_id_global + 1][op_id][0][
                                                  :len(self._masks_k[cell_id][edge_id_global + 1][op_id][0]) // 2] == 0).tolist())
                                if kernel_num_step[edge_id] // 2 > len(idx1_1_tmp1):
                                    weight, idx1_3_tmp1 = torch.sort(
                                        F.sigmoid(self._kernel_parameters[cell_id][edge_id_global + 1][op_id][0][idx1_2_tmp1]),
                                        descending=True)
                                    idx1_1_tmp1.extend([idx1_2_tmp1[i] for i in
                                                        idx1_3_tmp1[:int(kernel_num_step[edge_id] // 2 - len(idx1_1_tmp1))]])
                                else:
                                    weight, idx1_3_tmp1 = torch.sort(
                                        F.sigmoid(
                                            self._kernel_parameters[cell_id][edge_id_global + 1][op_id][0][
                                                idx1_1_tmp1]),
                                        descending=True)
                                    idx1_1_tmp1 = [idx1_1_tmp1[i] for i in
                                                        idx1_3_tmp1[
                                                        :int(kernel_num_step[edge_id] // 2)]]
                                idx1_1_tmp1.sort(reverse=False)
                                idx1_1_tmp2 = self.flatten(
                                    torch.nonzero(self._masks_k[cell_id][edge_id_global + 1][op_id][0][
                                                  len(self._masks_k[cell_id][edge_id_global + 1][op_id][0]) // 2:]).tolist())
                                idx1_2_tmp2 = self.flatten(
                                    torch.nonzero(
                                        self._masks_k[cell_id][edge_id_global + 1][op_id][0][
                                        len(self._masks_k[cell_id][edge_id_global + 1][op_id][0]) // 2:] == 0).tolist())
                                if kernel_num_step[edge_id] // 2 > len(idx1_1_tmp2):
                                    weight, idx1_3_tmp2 = torch.sort(
                                        F.sigmoid(self._kernel_parameters[cell_id][edge_id_global + 1][op_id][0][idx1_2_tmp2]),
                                        descending=True)
                                    idx1_1_tmp2.extend(
                                        [idx1_2_tmp2[i] for i in
                                         idx1_3_tmp2[:int(kernel_num_step[edge_id] // 2 - len(idx1_1_tmp2))]])
                                else:
                                    weight, idx1_3_tmp2 = torch.sort(
                                        F.sigmoid(
                                            self._kernel_parameters[cell_id][edge_id_global + 1][op_id][0][
                                                idx1_1_tmp2]),
                                        descending=True)
                                    idx1_1_tmp2 = [idx1_1_tmp2[i] for i in
                                         idx1_3_tmp2[:int(kernel_num_step[edge_id] // 2 )]]
                                idx1_1_tmp2.sort(reverse=False)
                                idx1_1 = []
                                idx1_1.extend(idx1_1_tmp1)
                                idx1_1.extend(idx1_1_tmp2)
                                active_kernel_id_op.append(idx1_0)
                                active_kernel_id_op.append(idx1_1)
                        active_kernel_id_edge.append(active_kernel_id_op)
                        kernel_num_edge.append(kernel_num_op)
                        # else:
                        #     logging.info("Have pruned: "+str(edge_id_global)+", "+str(op_id))
                        op_id += 1
                    edge_id += 1
                    edge_id_global += 1
                    active_kernel_id_cell.append(active_kernel_id_edge)
                    kernel_num_cell.append(kernel_num_edge)
                start = end
                n += 1

            active_kernel_id_all.append(active_kernel_id_cell)
            kernel_num_all.append(kernel_num_cell)

        self._kernel_sum = kernel_sum_all
        self._kernel_num = kernel_num_all
        self._active_kernel_id = active_kernel_id_all

        logging.info("kernel_sum_all: "+str(kernel_sum_all))
        # logging.info(equal)

        with torch.no_grad():

            for i in range(cell_end):
                if i==0:
                    C_prev_prev = self.cells[0].C_prev_prev
                    C_prev = self.cells[0].C_prev
                elif i==1:
                    C_prev_prev = self.cells[1].C_prev_prev
                    C_prev = int(sum(self._kernel_sum[0][2:]))
                else:
                    C_prev_prev = int(sum(self._kernel_sum[i-2][2:]))
                    C_prev = int(sum(self._kernel_sum[i-1][2:]))
                if self.cells[i].reduction_prev:
                    pre_weight1 = self.cells[i].preprocess0.conv_1.weight
                    pre_weight2 = self.cells[i].preprocess0.conv_2.weight
                    pre_weight3 = self.cells[i].preprocess0.bn_2.weight
                    self.cells[i].preprocess0 = FactorizedReduce(C_prev_prev, int(self._kernel_sum[i][0]), affine=True)
                    self.cells[i].preprocess0.conv_1.weight = Parameter(pre_weight1[self._active_kernel_id[i][0][0][1][:len(self._active_kernel_id[i][0][0][1])//2],:][:,self._active_kernel_id[i][0][0][0]])
                    self.cells[i].preprocess0.conv_2.weight = Parameter(
                        pre_weight2[self._active_kernel_id[i][0][0][1][len(self._active_kernel_id[i][0][0][1]) // 2:],:][:,self._active_kernel_id[i][0][0][0]])
                    active_cat = self._active_kernel_id[i][0][0][1][:len(self._active_kernel_id[i][0][0][1]) // 2]
                    active_cat.extend([item +len(self._active_kernel_id[i][0][0][1]) // 2 for item in self._active_kernel_id[i][0][0][1][len(self._active_kernel_id[i][0][0][1]) // 2:]])
                    self.cells[i].preprocess0.bn_2.weight = Parameter(
                        pre_weight3[active_cat])
                    # logging.info(active_cat)
                    # logging.info(self._active_kernel_id[i][0][0][0])
                else:
                    pre_weight1 = self.cells[i].preprocess0.op[1].weight
                    pre_weight2 = self.cells[i].preprocess0.op[2].weight
                    self.cells[i].preprocess0 = ReLUConvBN(C_prev_prev, int(self._kernel_sum[i][0]), 1, 1, 0, affine=True)
                    self.cells[i].preprocess0.op[1].weight = Parameter(
                        pre_weight1[self._active_kernel_id[i][0][0][1],:][:,self._active_kernel_id[i][0][0][0]])
                    self.cells[i].preprocess0.op[2].weight = Parameter(
                        pre_weight2[self._active_kernel_id[i][0][0][1]])
                pre_weight1 = self.cells[i].preprocess1.op[1].weight
                pre_weight2 = self.cells[i].preprocess1.op[2].weight
                self.cells[i].preprocess1 = ReLUConvBN(C_prev, int(self._kernel_sum[i][1]), 1, 1, 0, affine=True)
                self.cells[i].preprocess1.op[1].weight = Parameter(
                    pre_weight1[self._active_kernel_id[i][0][1][1],:][:,self._active_kernel_id[i][0][1][0]])
                self.cells[i].preprocess1.op[2].weight = Parameter(
                    pre_weight2[self._active_kernel_id[i][0][1][1]])
                # self._masks_k[i][0][0][0] = self._masks_k[i][0][0][0][self._active_kernel_id[i][0][0][0]]
                # self._masks_k[i][0][1][0] = self._masks_k[i][0][1][0][self._active_kernel_id[i][0][1][0]]
                try:
                    self._masks_k[i][0][0][0] = torch.ones(len(self._active_kernel_id[i][0][0][1])).cuda()
                except Exception as e:
                    logging.info("i: " + str(i))
                    logging.info("masks: "+str(self._masks_k[i]))
                    logging.info("ids: " + str(self._active_kernel_id[i]))
                self._masks_k[i][0][1][0] = torch.ones(len(self._active_kernel_id[i][0][1][1])).cuda()
                self._kernel_parameters[i][0][0][0] = Parameter(self._kernel_parameters[i][0][0][0][self._active_kernel_id[i][0][0][1]])
                self._kernel_parameters[i][0][1][0] = Parameter(self._kernel_parameters[i][0][1][0][self._active_kernel_id[i][0][1][1]])
                for j in range(len(self.cells[i]._ops)):
                    for k in range(len(self.cells[i]._ops[j]._ops)):
                        if self._masks[i][j,k] == 1:
                            stride = 2 if self.cells[i].reduction and j in [0, 1, 2, 3, 5, 6, 9, 10] else 1
                            if isinstance(self.cells[i]._ops[j]._ops[k], SepConv):
                                pre_weight = [self.cells[i]._ops[j]._ops[k].op[1].weight,
                                              self.cells[i]._ops[j]._ops[k].op[2].weight,
                                              self.cells[i]._ops[j]._ops[k].op[3].weight,
                                              self.cells[i]._ops[j]._ops[k].op[5].weight,
                                              self.cells[i]._ops[j]._ops[k].op[6].weight,
                                              self.cells[i]._ops[j]._ops[k].op[7].weight]
                                # logging.info(int(self._kernel_num[i][j][k][4]))
                                # logging.info(self._active_kernel_id[i][j+1][k][3])
                                self.cells[i]._ops[j]._ops[k] = OPS[
                                    PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]),int(self._kernel_num[i][j][k][2]),int(self._kernel_num[i][j][k][4]), stride, True)
                                self.cells[i]._ops[j]._ops[k].op[1].weight = Parameter(pre_weight[0][self._active_kernel_id[i][j+1][k][0]])
                                self.cells[i]._ops[j]._ops[k].op[2].weight = Parameter(pre_weight[1][self._active_kernel_id[i][j+1][k][1],:][:,self._active_kernel_id[i][j+1][k][0]])
                                self.cells[i]._ops[j]._ops[k].op[3].weight = Parameter(
                                    pre_weight[2][self._active_kernel_id[i][j + 1][k][1]])
                                self.cells[i]._ops[j]._ops[k].op[5].weight = Parameter(pre_weight[3][self._active_kernel_id[i][j+1][k][2]])
                                self.cells[i]._ops[j]._ops[k].op[6].weight = Parameter(pre_weight[4][self._active_kernel_id[i][j+1][k][3],:][:,self._active_kernel_id[i][j+1][k][2]])
                                self.cells[i]._ops[j]._ops[k].op[7].weight = Parameter(
                                    pre_weight[5][self._active_kernel_id[i][j + 1][k][3]])
                                for l in range(len(self._masks_k[i][j+1][k])):
                                    if l==0 or l==2:
                                        self._masks_w[i][j][k][l] = self._masks_w[i][j][k][l][
                                            self._active_kernel_id[i][j + 1][k][l]]
                                    else:
                                        self._masks_w[i][j][k][l] = self._masks_w[i][j][k][l][
                                            self._active_kernel_id[i][j + 1][k][l],:][:,self._active_kernel_id[i][j + 1][k][l-1]]
                                    # self._masks_k[i][j+1][k][l] = self._masks_k[i][j+1][k][l][self._active_kernel_id[i][j+1][k][l]]
                                    if l==1 or l==3:
                                        self._masks_k[i][j+1][k][l] = torch.ones(
                                            len(self._active_kernel_id[i][j+1][k][l])).cuda()
                                        self._kernel_parameters[i][j+1][k][l] = Parameter(self._kernel_parameters[i][j+1][k][l][
                                            self._active_kernel_id[i][j+1][k][l]])
                            elif isinstance(self.cells[i]._ops[j]._ops[k], DilConv):
                                pre_weight = [self.cells[i]._ops[j]._ops[k].op[1].weight,
                                              self.cells[i]._ops[j]._ops[k].op[2].weight,
                                              self.cells[i]._ops[j]._ops[k].op[3].weight]
                                self.cells[i]._ops[j]._ops[k] = OPS[
                                    PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]),int(self._kernel_num[i][j][k][2]), stride, True)
                                self.cells[i]._ops[j]._ops[k].op[1].weight = Parameter(
                                    pre_weight[0][self._active_kernel_id[i][j+1][k][0]])
                                self.cells[i]._ops[j]._ops[k].op[2].weight = Parameter(
                                    pre_weight[1][self._active_kernel_id[i][j+1][k][1],:][:,self._active_kernel_id[i][j+1][k][0]])
                                self.cells[i]._ops[j]._ops[k].op[3].weight = Parameter(
                                    pre_weight[2][self._active_kernel_id[i][j + 1][k][1]])
                                for l in range(len(self._masks_k[i][j+1][k])):
                                    if l==0:
                                        self._masks_w[i][j][k][l] = self._masks_w[i][j][k][l][
                                            self._active_kernel_id[i][j + 1][k][l]]
                                    else:
                                        self._masks_w[i][j][k][l] = self._masks_w[i][j][k][l][
                                            self._active_kernel_id[i][j + 1][k][l],:][:,self._active_kernel_id[i][j + 1][k][l-1]]
                                    # self._masks_k[i][j+1][k][l] = self._masks_k[i][j+1][k][l][self._active_kernel_id[i][j+1][k][l]]
                                    if l==1:
                                        self._masks_k[i][j + 1][k][l] = torch.ones(
                                            len(self._active_kernel_id[i][j + 1][k][l])).cuda()
                                        self._kernel_parameters[i][j+1][k][l] = Parameter(self._kernel_parameters[i][j+1][k][l][
                                            self._active_kernel_id[i][j+1][k][l]])
                            elif isinstance(self.cells[i]._ops[j]._ops[k], FactorizedReduce):
                                pre_weight = [self.cells[i]._ops[j]._ops[k].conv_1.weight,
                                              self.cells[i]._ops[j]._ops[k].conv_2.weight,
                                              self.cells[i]._ops[j]._ops[k].bn_2.weight]
                                self.cells[i]._ops[j]._ops[k] = FactorizedReduce(int(self._kernel_num[i][j][k][0]), int(self._kernel_num[i][j][k][0]),
                                                                             affine=True)
                                self.cells[i]._ops[j]._ops[k].conv_1.weight = Parameter(pre_weight[0][
                                                                                       self._active_kernel_id[i][j+1][k][1][
                                                                                       :len(self._active_kernel_id[i][j+1][k][
                                                                                                1]) // 2],:][:,self._active_kernel_id[i][j+1][k][0]])
                                self.cells[i]._ops[j]._ops[k].conv_2.weight = Parameter(
                                    pre_weight[1][
                                        self._active_kernel_id[i][j+1][k][1][len(self._active_kernel_id[i][j+1][k][1]) // 2:],:][:,self._active_kernel_id[i][j+1][k][0]])
                                active_cat = self._active_kernel_id[i][j+1][k][1][
                                             :len(self._active_kernel_id[i][j+1][k][1]) // 2]
                                active_cat.extend([item + len(
                                    self._active_kernel_id[i][j+1][k][1]) // 2 for item in self._active_kernel_id[i][j+1][k][1][
                                                  len(self._active_kernel_id[i][j+1][k][1]) // 2:]])
                                self.cells[i]._ops[j]._ops[k].bn_2.weight = Parameter(
                                    pre_weight[2][active_cat])
                                self._masks_k[i][j + 1][k][0] = torch.ones(
                                    len(self._active_kernel_id[i][j + 1][k][1])).cuda()
                                self._kernel_parameters[i][j + 1][k][0] = Parameter(
                                    self._kernel_parameters[i][j + 1][k][0][
                                        self._active_kernel_id[i][j + 1][k][1]])
                            else:
                                try:
                                    self.cells[i]._ops[j]._ops[k] = OPS[
                                        PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]), stride, True)
                                except Exception as e:
                                    logging.info(str(i)+str(j)+str(k))
                                    logging.info(isinstance(self.cells[i]._ops[j]._ops[k], Zero))
                                    logging.info(self._kernel_num[i][j][k])
                                    logging.info(self._kernel_num[i][j][k][0])
            # if i==self.stage1_end:
            #     pre_weight = self.auxiliary_head1.features[2].weight
            #     list = [0 for i in range(len(pre_weight[0]))]
            #     for out_idx in range(len(pre_weight)):
            #         for in_idx in range(len(pre_weight[out_idx])):
            #             list[in_idx] += abs(pre_weight[out_idx][in_idx]).sum()
            #     weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
            #     idx1_1 = idx1_3[:int(sum(self._kernel_sum[i][2:]))].tolist()
            #     idx1_1.sort(reverse=False)
            #     self.auxiliary_head1 = AuxiliaryHeadCIFAR1(int(sum(self._kernel_sum[i][2:])), self._num_classes)
            #     self.auxiliary_head1.features[2].weight = Parameter(pre_weight[:,idx1_1])
            # elif i==self.stage2_end:
            #     pre_weight = self.auxiliary_head2.features[2].weight
            #     list = [0 for i in range(len(pre_weight[0]))]
            #     for out_idx in range(len(pre_weight)):
            #         for in_idx in range(len(pre_weight[out_idx])):
            #             list[in_idx] += abs(pre_weight[out_idx][in_idx]).sum()
            #     weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
            #     idx1_1 = idx1_3[:int(sum(self._kernel_sum[i][2:]))].tolist()
            #     idx1_1.sort(reverse=False)
            #     self.auxiliary_head2 = AuxiliaryHeadCIFAR2(int(sum(self._kernel_sum[i][2:])), self._num_classes)
            #     self.auxiliary_head2.features[2].weight = Parameter(pre_weight[:, idx1_1])
            # elif i==self._layers-1:
            #     pre_weight = self.classifier.weight
            #     list = [0 for i in range(len(pre_weight[0]))]
            #     for out_idx in range(len(pre_weight)):
            #         for in_idx in range(len(pre_weight[out_idx])):
            #             list[in_idx] += abs(pre_weight[out_idx][in_idx]).sum()
            #     weight, idx1_3 = torch.sort(torch.Tensor(list).cuda(), descending=True)
            #     idx1_1 = idx1_3[:int(sum(self._kernel_sum[i][2:]))].tolist()
            #     idx1_1.sort(reverse=False)
            #     self.classifier = nn.Linear(int(sum(self._kernel_sum[i][2:])), self._num_classes)
            #     self.classifier.weight = Parameter(pre_weight[:, idx1_1])


        # logging.info("kernel sum: " + str(self._kernel_sum[0][0]))

    def update_kernel_num(self,stage_index):
        if stage_index == 1:
            cell_end = self.stage1_end + 1
        elif stage_index == 2:
            cell_end = self.stage2_end + 1
        else:
            cell_end = self._layers
        kernel_sum_all = []
        kernel_num_all = []
        for cell_id in range(cell_end):
            weights2 = F.sigmoid(self._arch_parameters[cell_id])
            n = 2
            start = 0
            edge_id_global = 0
            init_num0 = len(self._masks_k[cell_id][0][0][0])
            init_num1 = len(self._masks_k[cell_id][0][1][0])
            kernel_num_step = [torch.Tensor([init_num0])[0], torch.Tensor([init_num1])[0]]
            for step in range(self._steps):
                end = start + n
                W = weights2[start:end]
                kernel_num_max = 0
                for w_edge in W:
                    op_id = 0
                    for w_op in w_edge:
                        # if w_op > self.eta_min * weight_sum:
                        if self._masks[cell_id][edge_id_global][op_id] == 1:
                            if isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], SepConv) or isinstance(
                                    self.cells[cell_id]._ops[edge_id_global]._ops[op_id], DilConv):
                                num = len(self._masks_k[cell_id][edge_id_global+1][op_id][-1])
                                if num >= kernel_num_max:
                                    kernel_num_max = num
                        op_id += 1
                    edge_id_global += 1
                start = end
                n += 1

                if kernel_num_max==0:
                    kernel_num_max = kernel_num_step[-1]

                kernel_num_step.append(kernel_num_max)

            n = 2
            start = 0
            edge_id_global = 0
            kernel_num_cell = []
            for step in range(self._steps):
                end = start + n
                W = weights2[start:end]
                weight_sum = W.sum()
                edge_id = 0
                for w_edge in W:
                    op_id = 0
                    kernel_num_edge = []
                    for w_op in w_edge:
                        kernel_num_op = []
                        # if w_op > self.eta_min * weight_sum:
                        if self._masks[cell_id][edge_id_global][op_id] == 1:
                            kernel_num_op.append(kernel_num_step[edge_id])
                            if isinstance(self.cells[cell_id]._ops[edge_id_global]._ops[op_id], SepConv) or isinstance(
                                    self.cells[cell_id]._ops[edge_id_global]._ops[op_id], DilConv):
                                # num = []
                                for l in range(len(self._masks_k[cell_id][edge_id_global + 1][op_id])):
                                    num = len(self._masks_k[cell_id][edge_id_global + 1][op_id][l])
                                    kernel_num_op.append(num)
                        kernel_num_edge.append(kernel_num_op)
                        # else:
                        #     logging.info("Have pruned: "+str(edge_id_global)+", "+str(op_id))
                        op_id += 1
                    edge_id += 1
                    edge_id_global += 1
                    kernel_num_cell.append(kernel_num_edge)
                start = end
                n += 1

            kernel_sum_all.append(kernel_num_step)
            kernel_num_all.append(kernel_num_cell)

        self._kernel_sum = kernel_sum_all
        self._kernel_num = kernel_num_all

    def prune_kernel_update(self, stage_index=0):
        if stage_index == 1:
            cell_end = self.stage1_end + 1
        elif stage_index == 2:
            cell_end = self.stage2_end + 1
        else:
            cell_end = self._layers
        with torch.no_grad():

            for i in range(cell_end):
                if i==0:
                    C_prev_prev = self.cells[0].C_prev_prev
                    C_prev = self.cells[0].C_prev
                elif i==1:
                    C_prev_prev = self.cells[1].C_prev_prev
                    C_prev = int(sum(self._kernel_sum[0][2:]))
                else:
                    C_prev_prev = int(sum(self._kernel_sum[i-2][2:]))
                    C_prev = int(sum(self._kernel_sum[i-1][2:]))
                logging.info("_kernel_sum:"+str(self._kernel_sum[i]))
                if self.cells[i].reduction_prev:
                    self.cells[i].preprocess0 = FactorizedReduce(C_prev_prev, int(self._kernel_sum[i][0]), affine=True)
                else:
                    self.cells[i].preprocess0 = ReLUConvBN(C_prev_prev, int(self._kernel_sum[i][0]), 1, 1, 0, affine=True)
                self.cells[i].preprocess1 = ReLUConvBN(C_prev, int(self._kernel_sum[i][1]), 1, 1, 0, affine=True)
                for j in range(len(self.cells[i]._ops)):
                    for k in range(len(self.cells[i]._ops[j]._ops)):
                        if self._masks[i][j,k] == 1:
                            stride = 2 if self.cells[i].reduction and j in [0, 1, 2, 3, 5, 6, 9, 10] else 1
                            if isinstance(self.cells[i]._ops[j]._ops[k], SepConv):
                                self.cells[i]._ops[j]._ops[k] = OPS[
                                    PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]),int(self._kernel_num[i][j][k][2]),int(self._kernel_num[i][j][k][4]), stride, True)

                            elif isinstance(self.cells[i]._ops[j]._ops[k], DilConv):
                                try:
                                    self.cells[i]._ops[j]._ops[k] = OPS[
                                        PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]),int(self._kernel_num[i][j][k][2]), stride, True)
                                except Exception as e:
                                    logging.info(self._kernel_num[i][j][k])
                            elif isinstance(self.cells[i]._ops[j]._ops[k], FactorizedReduce):
                                self.cells[i]._ops[j]._ops[k] = FactorizedReduce(int(self._kernel_num[i][j][k][0]), int(self._kernel_num[i][j][k][0]),
                                                                             affine=True)

                            else:
                                try:
                                    self.cells[i]._ops[j]._ops[k] = OPS[
                                        PRIMITIVES[k]](int(self._kernel_num[i][j][k][0]), stride, True)
                                except Exception as e:
                                    logging.info(str(i)+str(j)+str(k))
                                    logging.info(isinstance(self.cells[i]._ops[j]._ops[k], Zero))
                                    logging.info(self._kernel_num[i][j][k])
                                    logging.info(self._kernel_num[i][j][k][0])

    def prune_weight(self, thre, stage_index=0):
        num = 0
        if stage_index == 1:
            end = self.stage1_end + 1
        elif stage_index == 2:
            end = self.stage2_end + 1
        else:
            end = self._layers
        for i in range(end):
            for j in range(len(self.cells[i]._ops)):
                for k in range(len(self.cells[i]._ops[j]._ops)):
                    if isinstance(self.cells[i]._ops[j]._ops[k], SepConv) or isinstance(self.cells[i]._ops[j]._ops[k],
                                                                                        DilConv):
                        num += self.cells[i]._ops[j]._ops[k].prune_weight(thre)
        return num

    def _initialize_alphas(self):
        edge_num = sum(1 for i in range(self._steps) for _ in range(2 + i))
        numix_ops = len(PRIMITIVES)
        self._active_kernel_id = []
        self._kernel_num = []
        self._kernel_sum = []
        self._arch_parameters = []
        self._kernel_parameters = []
        self._masks = []
        self._masks_k = []
        self._masks_w = []
        # self._kernel_masks = []
        # self._weight_masks = []
        self._thresholds = []
        for i in range(self._layers):
            alphas_temp = Variable(torch.zeros(edge_num, numix_ops).cuda(), requires_grad=True)
            self._arch_parameters.append(alphas_temp)
            masks_tmp = torch.ones(edge_num, numix_ops).cuda()
            self._masks.append(masks_tmp)
            kernel_param_edge = []
            masks_k_edge = []
            masks_w_edge = []
            masks_k_edge.append([[torch.ones(self.cells[i].preprocess0.c_out).cuda()], [torch.ones(self.cells[i].preprocess1.c_out).cuda()]])
            kernel_param_edge.append([[Variable(torch.zeros(self.cells[i].preprocess0.c_out).cuda(), requires_grad=True)],
                                       [Variable(torch.zeros(self.cells[i].preprocess1.c_out).cuda(), requires_grad=True)]])
            thre_tmp_edge = []
            thre_tmp_edge.append([[Variable(torch.zeros(1).cuda(), requires_grad=True),
                                   Variable(torch.zeros(1).cuda(), requires_grad=True),
                                   Variable(torch.zeros(1).cuda(), requires_grad=True)],
                                  [Variable(torch.zeros(1).cuda(), requires_grad=True),
                                   Variable(torch.zeros(1).cuda(), requires_grad=True),
                                   Variable(torch.zeros(1).cuda(), requires_grad=True)]])
            for j in range(len(self.cells[i]._ops)):
                kernel_param_op = []
                masks_k_op = []
                masks_w_op = []
                thre_tmp_op = []
                for k in range(len(self.cells[i]._ops[j]._ops)):
                    kernel_param_tmp = []
                    masks_k_tmp = []
                    masks_w_tmp = []
                    thre_tmp_list = []
                    if isinstance(self.cells[i]._ops[j]._ops[k], SepConv):
                        masks_k_tmp.append([])
                        masks_k_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[2].weight.shape[0]).cuda())
                        masks_k_tmp.append([])
                        masks_k_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[6].weight.shape[0]).cuda())
                        kernel_param_tmp.append([])
                        kernel_param_tmp.append(Variable(torch.zeros(self.cells[i]._ops[j]._ops[k].op[2].weight.shape[0]).cuda(), requires_grad=True))
                        kernel_param_tmp.append([])
                        kernel_param_tmp.append(Variable(torch.zeros(self.cells[i]._ops[j]._ops[k].op[6].weight.shape[0]).cuda(), requires_grad=True))
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[1].weight.shape).cuda())
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[2].weight.shape).cuda())
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[5].weight.shape).cuda())
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[6].weight.shape).cuda())

                    if isinstance(self.cells[i]._ops[j]._ops[k], DilConv):
                        masks_k_tmp.append([])
                        masks_k_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[2].weight.shape[0]).cuda())
                        kernel_param_tmp.append([])
                        kernel_param_tmp.append(Variable(torch.zeros(self.cells[i]._ops[j]._ops[k].op[2].weight.shape[0]).cuda(), requires_grad=True))
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[1].weight.shape).cuda())
                        masks_w_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].op[2].weight.shape).cuda())
                    if isinstance(self.cells[i]._ops[j]._ops[k], FactorizedReduce):
                        masks_k_tmp.append(torch.ones(self.cells[i]._ops[j]._ops[k].c_out).cuda())
                        kernel_param_tmp.append(
                            Variable(torch.zeros(self.cells[i]._ops[j]._ops[k].c_out).cuda(),
                                     requires_grad=True))

                        # self.cells[i]._ops[j]._ops[k].op[3].weight.data.fill_(0.5)
                    kernel_param_op.append(kernel_param_tmp)
                    masks_k_op.append(masks_k_tmp)
                    masks_w_op.append(masks_w_tmp)
                    for l in range(3):
                        thre_tmp = Variable(torch.zeros(1).cuda(), requires_grad=True)
                        thre_tmp_list.append(thre_tmp)
                    thre_tmp_op.append(thre_tmp_list)
                kernel_param_edge.append(kernel_param_op)
                masks_k_edge.append(masks_k_op)
                masks_w_edge.append(masks_w_op)
                thre_tmp_edge.append(thre_tmp_op)
            self._kernel_parameters.append(kernel_param_edge)
            self._masks_k.append(masks_k_edge)
            self._masks_w.append(masks_w_edge)

            self._thresholds.append(thre_tmp_edge)


    def _reinitialize_threshold(self):
        for i in range(len(self._thresholds)):
            for j in range(len(self._thresholds[i])):
                for k in range(len(self._thresholds[i][j])):
                    for l in range(len(self._thresholds[i][j][k])):
                        self._thresholds[i][j][k][l].data.fill_(0)


    def _reinitialize_alphas(self):
        for i in range(len(self._arch_parameters)):
            for j in range(len(self._arch_parameters[i])):
                for k in range(len(self._arch_parameters[i][j])):
                    if self._masks[i][j][k]==1:
                        self._arch_parameters[i].data[j][k].fill_(0)

    def current_flops(self, stage_index=0):
        cost_network = 0
        C = self._C
        for i in range(self._layers):
            weights2 = F.sigmoid(self._arch_parameters[i])
            kernel_param = self._kernel_parameters[i]
            # weights1 = F.sigmoid(self._arch_parameters[i][0])
            # weights2 = F.sigmoid(self._arch_parameters[i][1])
            if i in [self.stage1_end, self.stage2_end]:
                C *= 2
                reduction = True
            else:
                reduction = False
            edge_id = 0
            reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]
            for w in weights2:
                op_id = 0
                cost_edge = 0
                for w_op in w:
                    if edge_id in reduction_list and reduction:
                        skip_in_reduction = True
                    else:
                        skip_in_reduction = False
                    if self._masks[i][edge_id][op_id] != 0:
                        cost_edge += flops_computation(self._C, C, op_id, skip_in_reduction,
                                                       mask_k=self._masks_k[i][edge_id + 1][op_id],
                                                       mask_w=self._masks_w[i][edge_id][op_id],
                                                       kernel_param=kernel_param[edge_id + 1][op_id])
                    op_id += 1
                cost_network += cost_edge
                edge_id += 1
            if i == self.stage1_end:
                cost_network1 = cost_network
                if stage_index == 1:
                    return cost_network1
            if i == self.stage2_end:
                cost_network2 = cost_network
                if stage_index == 2:
                    return cost_network2
        if stage_index == 3:
            return cost_network
        return cost_network, cost_network1, cost_network2

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights1, weights2, all_ops, masks):
            gene = []

            n = 2
            start = 0
            edge_id_global = 0
            # logging.info(masks)
            for i in range(self._steps):
                end = start + n
                W = weights2[start:end].copy()
                weight_sum = W.sum()
                edge_id = 0
                for w_edge in W:
                    op_id = 0
                    for w_op in w_edge:
                        # if w_op > self.eta_min * weight_sum:
                        if masks[edge_id_global][op_id] == 1:
                            gene.append((all_ops[op_id], edge_id, i + 2))
                        # else:
                        #     logging.info("Have pruned: "+str(edge_id_global)+", "+str(op_id))
                        op_id += 1
                    edge_id += 1
                    edge_id_global += 1
                start = end
                n += 1
            return gene

        gene_list = []
        for i in range(self._layers):
            all_ops = PRIMITIVES
            # gene_list.append(_parse(F.sigmoid(self._arch_parameters[i][0]).data.cpu().numpy(),
            #                         F.sigmoid(self._arch_parameters[i][1]).data.cpu().numpy(), all_ops, att_ops))
            # logging.info("layer: "+str(i))
            gene_list.append(
                _parse(0, F.sigmoid(self._arch_parameters[i]).data.cpu().numpy(), all_ops, self._masks[i]))
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype._make([gene_list, concat])
        return genotype
