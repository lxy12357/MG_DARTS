from torch.nn.parameter import Parameter

from operations import *
from utils import drop_path

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
    for name, index_input, index_output, kernel_num in zip(op_names, indices_input, indices_output, kernel_nums):
      stride = 2 if reduction and index_input < 2 and index_output >= 2 else 1
      if 'sep_conv' in name:
        op = OPS[name](kernel_num[0],kernel_num[1],kernel_num[2], stride, True)
      elif 'dil_conv' in name:
        op = OPS[name](kernel_num[0], kernel_num[1], stride, True)
      else:
        op = OPS[name](kernel_num[0], stride, True)

      self._ops += [op]
    self._indices_input = indices_input
    self._indices_output = indices_output
    self._concat = list(set(indices_output))
    self.multiplier = len(self._concat)

  def forward(self, s0, s1, drop_prob, mask=[]):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(4):
      s=0
      for j in range(len(self._indices_output)):
        if self._indices_output[j]==(i+2):
          h=states[self._indices_input[j]]
          op=self._ops[j]
          if mask!=[] and (isinstance(op, SepConv) or isinstance(op, DilConv)):
            h = op(h, mask[j])
          else:
            h=op(h)
          if self.training and drop_prob > 0.:
            if not isinstance(op, Identity):
              h = drop_path(h, drop_prob)
          s=s+h
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

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
    C_curr = stem_multiplier*C
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
        C_curr * cell.multiplier
      if i in [self.stage1_end+1, self.stage2_end+1]:
        C_curr = C_curr * 2
        reduction = True
      else:
        reduction = False
      concat=eval("genotype.%s" % "concat")
      cell = Cell(genotype1, concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.kernel_num_out
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev
    self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.initialize_masks()

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob, mask=self._masks_w[i])
      if i == 2 * self._layers // 3:
        logits_aux = self.auxiliary_head(s1)
        # print(logits_aux.shape)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux

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
