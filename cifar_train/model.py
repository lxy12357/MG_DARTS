import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from operations import *
from torch.autograd import Variable
from utils import drop_path
import torch.nn.functional as F


def flops_computation(ci, c, op_id, skip_in_reduction=False, is_attention=False):
  UNIT = 0.000001
  CH = 32
  ratio = c / ci
  if is_attention:
    if op_id == 1:
      # return UNIT * (4 * (ci * ci/2 * CH * CH * CT) + CH * CH * CT * CH * CH * CT * ci/2 + CH * CH * CT * ci/2)
      return UNIT * (CH * CH + ci * ci / 16 * 2)
    elif op_id==2:
      return ci
    elif op_id == 0:
      return 0
  else:
    if op_id == 1:
      KS = 3
      return UNIT * 2 * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    elif op_id == 2:
      KS = 5
      return UNIT * 2 * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    # elif op_id == 3:
    #     KS = 7
    #     return UNIT * 2 * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    elif op_id == 3:
      KS = 3
      return UNIT * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    elif op_id == 4:
      KS = 5
      return UNIT * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    elif op_id == 5 or op_id == 6:
      KS = 3
      return UNIT * (KS * KS * CH * CH * ci / ratio)
    elif op_id == 0:
      if skip_in_reduction:
        return UNIT * ci * ci * CH * CH
      else:
        return 0
    else:
      return 0

def node_computation(weights_node, eta_min, single_edge=False):
  weight_sum = weights_node.sum()
  ops = 0
  if single_edge:
    for w_op in weights_node:
      if w_op / weight_sum > eta_min:
        ops = ops + 1
  else:
    for edge in weights_node:
      for w_op in edge:
        if w_op / weight_sum > eta_min:
          ops = ops + 1
  return weight_sum, ops

class Cell(nn.Module):

  def __init__(self, genotype, concat, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    # print(C_prev_prev, C_prev, C)
    if reduction:
      op_names, indices_input, indices_output, kernel_nums = zip(*genotype)
    else:
      op_names, indices_input, indices_output, kernel_nums = zip(*genotype)
    kernel_num_step = [C,C,C,C,C,C]
    for name, index_input, index_output, kernel_num in zip(op_names, indices_input, indices_output, kernel_nums):
      # for i in range(len(kernel_num)):
      #   kernel_num[i] = kernel_num[i]+20
      kernel_num_step[index_input] = kernel_num[0]
      kernel_num_step[index_output] = kernel_num[-1]
    self.kernel_num_out = sum(kernel_num_step[2:])
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev,kernel_num_step[0], affine=True)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, kernel_num_step[0], 1, 1, 0, affine=True)
    self.preprocess1 = ReLUConvBN(C_prev, kernel_num_step[1], 1, 1, 0, affine=True)

    self._compile(op_names, indices_input, indices_output, concat, reduction, kernel_nums)

  def _compile(self, op_names, indices_input, indices_output, concat, reduction, kernel_nums):
    assert len(op_names) == len(indices_input)
    self._steps = 4
    self._ops = nn.ModuleList()
    # self._ops0 = nn.ModuleList()
    # self._ops1 = nn.ModuleList()
    # pre_num = 0
    for name, index_input, index_output, kernel_num in zip(op_names, indices_input, indices_output, kernel_nums):
      stride = 2 if reduction and index_input < 2 and index_output >= 2 else 1
      if 'sep_conv' in name:
        op = OPS[name](kernel_num[0],kernel_num[1],kernel_num[2], stride, True)
      elif 'dil_conv' in name:
        op = OPS[name](kernel_num[0], kernel_num[1], stride, True)
      else:
        op = OPS[name](kernel_num[0], stride, True)
      # if index_input == index_output:
      #   if index_input == 0:
      #     self._ops0 += [op]
      #   else:
      #     self._ops1 += [op]
      #   pre_num += 1
      # else:
      self._ops += [op]
    self._indices_input = indices_input
    self._indices_output = indices_output
    self._concat = list(set(indices_output))
    self.multiplier = len(self._concat)

  def forward(self, s0, s1, drop_prob, mask=[]):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    # logging.info(str(s0.shape))
    # logging.info(str(s1.shape))
    # s = 0
    # # logging.info("*********cell starts")
    # # logging.info("s0 shape: " + str(s0.shape))
    # for j in range(len(self._ops0)):
    #   op = self._ops0[j]
    #   h = op(s0)
    #   if self.training and drop_prob > 0.:
    #     if not isinstance(op, Identity):
    #       h = drop_path(h, drop_prob)
    #   s = s + h
    # s0 = s
    # # logging.info("processed s0 shape: " + str(s0.shape))
    # s = 0
    # # logging.info("s1 shape: " + str(s1.shape))
    # for j in range(len(self._ops1)):
    #   op = self._ops1[j]
    #   h = op(s1)
    #   if self.training and drop_prob > 0.:
    #     if not isinstance(op, Identity):
    #       h = drop_path(h, drop_prob)
    #   s = s + h
    # s1 = s
    states = [s0, s1]
    for i in range(4):
      s=0
      for j in range(len(self._indices_output)):
        if self._indices_output[j]==(i+2):
          h=states[self._indices_input[j]]
          op=self._ops[j]
          if mask!=[] and (isinstance(op, SepConv) or isinstance(op, DilConv)):
            # if i==0 and (j==1 or j==0):
            #   h1 = op(h, mask[j])
            #   h2 = op(h)
            #   logging.info(str(i) + str(j))
            #   logging.info(str(h1.shape))
            #   logging.info(str(h2.shape))
            # try:
            h = op(h, mask[j])
            # except Exception as e:
            #   logging.info(j)
          else:
            h=op(h)
          if self.training and drop_prob > 0.:
            if not isinstance(op, Identity):
              h = drop_path(h, drop_prob)
          # try:
          s=s+h
          # except Exception as e:
          #   logging.info(str(i)+str(j))
          #   print(e)
          #   print(type(op))
          #   print(j, self._indices_input[j], self._indices_output[j])
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

# class AuxiliaryHeadCIFAR1(nn.Module):
#
#   def __init__(self, C, num_classes):
#     """assuming input size 32x32"""
#     super(AuxiliaryHeadCIFAR1, self).__init__()
#     self.features = nn.Sequential(
#       nn.ReLU(inplace=True),
#       nn.AvgPool2d(11, stride=3, padding=0, count_include_pad=False),  # image size = 8 x 8
#       nn.Conv2d(C, 128, 1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True),
#       nn.Conv2d(128, 768, 8, bias=False),
#       nn.BatchNorm2d(768),
#       nn.ReLU(inplace=True)
#     )
#     self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(768, num_classes))
#
#   def forward(self, x):
#     x = self.features(x)
#     x = self.classifier(x.view(x.size(0), -1))
#     # x = F.softmax(x)
#     return x
#
#
# class AuxiliaryHeadCIFAR2(nn.Module):
#
#   def __init__(self, C, num_classes):
#     """assuming input size 16x16"""
#     super(AuxiliaryHeadCIFAR2, self).__init__()
#     self.features = nn.Sequential(
#       nn.ReLU(inplace=True),
#       nn.AvgPool2d(7, stride=3, padding=0, count_include_pad=False),  # image size = 4 x 4
#       nn.Conv2d(C, 128, 1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True),
#       nn.Conv2d(128, 768, 4, bias=False),
#       nn.BatchNorm2d(768),
#       nn.ReLU(inplace=True)
#     )
#     self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(768, num_classes))
#
#   def forward(self, x):
#     x = self.features(x)
#     x = self.classifier(x.view(x.size(0), -1))
#     # x = F.softmax(x)
#     return x

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
      nn.AvgPool2d(8, padding=0, count_include_pad=False),  # image size = 1 x 1
      nn.Conv2d(128, 768, 1, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(768, num_classes))

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
      nn.AvgPool2d(4, padding=0, count_include_pad=False),  # image size = 1 x 1
      nn.Conv2d(128, 768, 1, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(768, num_classes))

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    # x = F.softmax(x)
    return x

# class AuxiliaryHeadCIFAR1(nn.Module):
#
#   def __init__(self, C, num_classes):
#     """assuming input size 32x32"""
#     super(AuxiliaryHeadCIFAR1, self).__init__()
#     self.features = nn.Sequential(
#       nn.ReLU(inplace=True),
#       nn.AvgPool2d(11, stride=3, padding=0, count_include_pad=False),  # image size = 8 x 8
#       nn.Conv2d(C, 128, 1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True),
#       # nn.AvgPool2d(8, padding=0, count_include_pad=False),  # image size = 1 x 1
#       nn.Conv2d(128, 256, 8, bias=False),
#       nn.BatchNorm2d(256),
#       nn.ReLU(inplace=True)
#     )
#     self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(256, num_classes))
#
#   def forward(self, x):
#     x = self.features(x)
#     x = self.classifier(x.view(x.size(0), -1))
#     # x = F.softmax(x)
#     return x
#
#
# class AuxiliaryHeadCIFAR2(nn.Module):
#
#   def __init__(self, C, num_classes):
#     """assuming input size 16x16"""
#     super(AuxiliaryHeadCIFAR2, self).__init__()
#     self.features = nn.Sequential(
#       nn.ReLU(inplace=True),
#       nn.AvgPool2d(7, stride=3, padding=0, count_include_pad=False),  # image size = 4 x 4
#       nn.Conv2d(C, 128, 1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True),
#       # nn.AvgPool2d(4, padding=0, count_include_pad=False),  # image size = 1 x 1
#       nn.Conv2d(128, 256, 4, bias=False),
#       nn.BatchNorm2d(256),
#       nn.ReLU(inplace=True)
#     )
#     self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(256, num_classes))
#
#   def forward(self, x):
#     x = self.features(x)
#     x = self.classifier(x.view(x.size(0), -1))
#     # x = F.softmax(x)
#     return x

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    return x

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, genotype):
    super(NetworkCIFAR, self).__init__()
    self._C = C
    genotype_arch = genotype.gene
    layers = len(genotype_arch)
    self._layers = len(genotype_arch)
    self.stage1_end = self._layers // 3-1  # 4
    self.stage2_end = 2 * self._layers // 3-1  # 9
    stem_multiplier = 3
    C_start = C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      genotype1 = genotype_arch[i]
      if i == self.stage1_end:
        C_stage1 = C_curr * cell.multiplier
      if i in [self.stage1_end+1, self.stage2_end+1]:
        C_curr = C_curr * 2
        reduction = True
        # if i == self.stage1_end + 1:
        #   reduction_prev = False
        #   C_prev_prev = C_start
        # if i == self.stage2_end + 1:
        #   # reduction_prev = False
        #   # C_prev_prev = C_start
        #   reduction_prev = True
        #   C_prev_prev = C_stage1
      else:
        reduction = False
      concat=eval("genotype.%s" % "concat")
      cell = Cell(genotype1, concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.kernel_num_out
    #   if i == self.stage1_end:
    #     C_to_auxiliary1 = C_prev
    #   if i == self.stage2_end:
    #     C_to_auxiliary2 = C_prev
    # self.auxiliary_head1 = AuxiliaryHeadCIFAR1(C_to_auxiliary1, num_classes)
    # self.auxiliary_head2 = AuxiliaryHeadCIFAR2(C_to_auxiliary2, num_classes)
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev
    self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.initialize_masks()

  def forward(self, input, early_exit=0):
    logits_aux1 = None
    logits_aux2 = None
    # print(input.shape)
    start = s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # if i == self.stage1_end + 1:
      #   s0 = start
      # if i == self.stage2_end + 1:
      #   # s0 = start
      #   s0 = stage1_out
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob, mask=self._masks_w[i])
      # if i == self.stage1_end:
      #   stage1_out = s1
      #   logits_aux1_raw = self.auxiliary_head1(s1)
      #   logits_aux1 = F.softmax(logits_aux1_raw)
      #   if early_exit==1:
      #     return logits_aux1
      # if i == self.stage2_end:
      #   logits_aux2 = self.auxiliary_head2(s1)
      #   # logits_aux2_raw = logits_aux1_raw + logits_aux2
      #   # logits_aux2 = F.softmax(logits_aux2_raw)
      #   logits_aux2 = F.softmax(logits_aux2)
      #   if early_exit==2:
      #     return logits_aux2
      if i == 2 * self._layers // 3:
        logits_aux = self.auxiliary_head(s1)
        # print(logits_aux.shape)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    # logits = logits + logits_aux2_raw
    # logits = F.softmax(logits)
    # return logits, logits_aux1, logits_aux2
    # return logits,logits_aux2
    return logits, logits_aux
    # return logits, logits, logits

  def initialize_masks(self):
    self._masks_w = []
    for i in range(self._layers):
      self._masks_w.append([])

  def init_weights(self, weight):
    for i in range(self._layers):
      num = 0
      for j in range(len(self.cells[i]._ops)):
        with torch.no_grad():
          if isinstance(self.cells[i]._ops[j], SepConv):
            self.cells[i]._ops[j].op[1].weight = Parameter(torch.Tensor(weight[i][num][0]))
            self.cells[i]._ops[j].op[2].weight = Parameter(torch.Tensor(weight[i][num][1]))
            self.cells[i]._ops[j].op[3].weight = Parameter(torch.Tensor(weight[i][num][2]))
            try:
              self.cells[i]._ops[j].op[5].weight = Parameter(torch.Tensor(weight[i][num][3]))
            except Exception as e:
              logging.info("weight: "+str(len(weight[i][num])))
            self.cells[i]._ops[j].op[6].weight = Parameter(torch.Tensor(weight[i][num][4]))
            self.cells[i]._ops[j].op[7].weight = Parameter(torch.Tensor(weight[i][num][5]))
            num += 1
          elif isinstance(self.cells[i]._ops[j], DilConv):
            self.cells[i]._ops[j].op[1].weight = Parameter(torch.Tensor(weight[i][num][0]))
            self.cells[i]._ops[j].op[2].weight = Parameter(torch.Tensor(weight[i][num][1]))
            self.cells[i]._ops[j].op[3].weight = Parameter(torch.Tensor(weight[i][num][2]))
            num += 1

  def init_final_weights(self, weight):
    for i in range(self._layers):
      for j in range(len(self.cells[i]._ops)):
        with torch.no_grad():
          if isinstance(self.cells[i]._ops[j], SepConv):
            self.cells[i]._ops[j].op[1].weight = Parameter(torch.Tensor(weight[i][j][0].cpu()))
            self.cells[i]._ops[j].op[2].weight = Parameter(torch.Tensor(weight[i][j][1].cpu()))
            self.cells[i]._ops[j].op[3].weight = Parameter(torch.Tensor(weight[i][j][2].cpu()))
            try:
              self.cells[i]._ops[j].op[5].weight = Parameter(torch.Tensor(weight[i][j][3].cpu()))
            except Exception as e:
              logging.info("weight: "+str(len(weight[i][j])))
            self.cells[i]._ops[j].op[6].weight = Parameter(torch.Tensor(weight[i][j][4].cpu()))
            self.cells[i]._ops[j].op[7].weight = Parameter(torch.Tensor(weight[i][j][5].cpu()))
          elif isinstance(self.cells[i]._ops[j], DilConv):
            self.cells[i]._ops[j].op[1].weight = Parameter(torch.Tensor(weight[i][j][0].cpu()))
            self.cells[i]._ops[j].op[2].weight = Parameter(torch.Tensor(weight[i][j][1].cpu()))
            self.cells[i]._ops[j].op[3].weight = Parameter(torch.Tensor(weight[i][j][2].cpu()))


  # def current_flops(self):
  #   cost_network = 0
  #   C = self._C
  #   for i in range(self._layers):
  #     weights = F.sigmoid(self._arch_parameters[i])
  #     if i in [self._layers // 3, self._layers * 2 // 3]:
  #       C *= 2
  #       reduction = True
  #     else:
  #       reduction = False
  #     edge_id = 0
  #     reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]
  #     for w in weights:
  #       op_id = 0
  #       cost_edge = 0
  #       for w_op in w:
  #         if edge_id in reduction_list and reduction:
  #           skip_in_reduction = True
  #         else:
  #           skip_in_reduction = False
  #         weight_sum = 0
  #         ops = 0
  #         if edge_id == 0:
  #           weight_sum, ops = node_computation(weights[0:2], self.eta_min)
  #         elif edge_id == 2:
  #           weight_sum, ops = node_computation(weights[2:5], self.eta_min)
  #         elif edge_id == 5:
  #           weight_sum, ops = node_computation(weights[5:9], self.eta_min)
  #         elif edge_id == 9:
  #           weight_sum, ops = node_computation(weights[9:14], self.eta_min)
  #         if w_op / weight_sum > self.eta_min:
  #           cost_edge += flops_computation(self._C, C, op_id, skip_in_reduction)
  #         op_id += 1
  #       cost_network += cost_edge
  #       edge_id += 1
  #     if i == self._layers // 3:
  #       cost_network1 = cost_network
  #     if i == 2 * self._layers // 3:
  #       cost_network2 = cost_network
  #   return cost_network, cost_network1, cost_network2