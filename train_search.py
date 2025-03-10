import gc
import os
import sys
import time
import glob

import numpy as np
import torch
from torch.nn.utils import prune

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from operations import *
import random

from genotypes import *

from prune import Unstructured, global_unstructured, unstructured

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--searched_epochs', type=int, default=25, help='num of searched epochs')
# parser.add_argument('--learning_rate_omega', type=float, default=[0.005,0.01,0.02,0.01], help='learning rate for omega')
parser.add_argument('--learning_rate_omega', type=float, default=[0.01,0.01,0.01,0.01], help='learning rate for omega')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=44, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--learning_rate_alpha', type=float, default=[0.5,1,2,1], help='learning rate for alpha')
parser.add_argument('--learning_rate_alpha', type=float, default=[1,1,1,1], help='learning rate for alpha')
parser.add_argument('--learning_rate_alpha_kernel', type=float, default=[1,1,1,1], help='learning rate for alpha')
parser.add_argument('--weight_decay_alpha', type=float, default=0, help='weight decay for alpha')
# parser.add_argument('--learning_rate_alpha', type=float, default=[6e-4,6e-4,6e-4,6e-4], help='learning rate for alpha')
# parser.add_argument('--weight_decay_alpha', type=float, default=1e-3, help='weight decay for alpha')
parser.add_argument('--eta_min', type=float, default=0.01, help='eta min')
parser.add_argument('--eta_max', type=float, default=0.05, help='eta max')
parser.add_argument('--pruning_n0_1', type=int, default=1, help='pruning velocity')
parser.add_argument('--pruning_n01', type=int, default=1, help='pruning velocity')
parser.add_argument('--lambda0', type=float, default=1e-5, help='lambda0')
parser.add_argument('--c0', type=float, default=2.0, help='c0')
parser.add_argument('--mu', type=float, default=0, help='the mu parameter')
parser.add_argument('--reg_flops', type=float, default=1, help='reg for FLOPs')
parser.add_argument('--min_flops', type=float, default=0.1, help='min FLOPs')
parser.add_argument('--base_flops', type=float, default=0, help='base FLOPs')
parser.add_argument('--auto_augment', action='store_false', default=True, help='whether autoaugment is used')
parser.add_argument('--stable_round', type=float, default=3, help='number of rounds for stability')

parser.add_argument('--max_flops_lambda1', type=float, default=100, help='max flops lambda')
parser.add_argument('--max_flops_lambda2', type=float, default=100, help='max flops lambda')
parser.add_argument('--max_flops_ratio', type=float, default=0.5, help='eta max ratio')
# parser.add_argument('--max_flops_lambda', type=float, default=1, help='max flops lambda')
parser.add_argument('--max_stable_round', type=float, default=3, help='max stable round')
# parser.add_argument('--max_stable_round', type=float, default=5, help='max stable round')
parser.add_argument('--eta_max_delta', type=float, default=0.01, help='eta max delta')
# parser.add_argument('--max_eta_max', type=float, default=0.1, help='max eta max')
# parser.add_argument('--max_eta_max', type=float, default=0.05, help='max eta max')
# parser.add_argument('--eta_max_raw', type=float, default=0.3, help='eta max')
parser.add_argument('--eta_max_raw', type=float, default=100, help='eta max')
parser.add_argument('--pruning_n_thre1', type=int, default=12, help='pruning velocity')
parser.add_argument('--pruning_n_thre2', type=int, default=12, help='pruning velocity')
parser.add_argument('--kernel_thre', type=float, default=0.05, help='eta max')
parser.add_argument('--weight_thre', type=float, default=0.05, help='eta max')
parser.add_argument('--kernel_ratio', type=float, default=0.2, help='eta max')
parser.add_argument('--weight_ratio', type=float, default=0.2, help='eta max')
parser.add_argument('--initial_epoch_num', type=int, default=5, help='initial epoch num')
parser.add_argument('--initial_epoch_num_stage', type=int, default=5, help='initial epoch num')
parser.add_argument('--search_epoch_num', type=int, default=28, help='initial epoch num')
parser.add_argument('--kernel_epoch_num', type=int, default=10, help='initial epoch num')
parser.add_argument('--weight_epoch_num', type=int, default=15, help='initial epoch num')
parser.add_argument('--stop_epoch_num', type=int, default=3, help='stop epoch num')
parser.add_argument('--regrow_thre', type=int, default=50, help='regrow threshold')
parser.add_argument('--regrow_ratio', type=float, default=0.2, help='regrow ratio')

args, unparsed = parser.parse_known_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def prune_op(model, args, eta_max, pruning_n0, stage_index=0,step=False,max=0):
    MAX = 10000.0
    EPS = 1e-8
    pruned_ops = 0
    for x in range(pruning_n0):
        pruning_cell = 0
        pruning_edge = 0
        pruning_op = 0
        pruning_w = MAX
        pruning_w_op = MAX
        start = 0
        end = args.layers
        if stage_index==1:
            end = model.stage1_end+1
        elif stage_index==2:
            # start = model.stage1_end+1
            end = model.stage2_end+1
        elif stage_index==3:
            # start = model.stage2_end + 1
            end = args.layers
        # elif stage_index==4:
        #     start = model.stage3_end + 1
        #     end = args.layers
        # for cell_id in range(start,end):
        #     # cell_weights1 = torch.sigmoid(model.arch_parameters()[cell_id][0])
        #     # edge_id = 0
        #     # while edge_id<2:
        #     #     edge_weights = cell_weights1[edge_id]
        #     #     weight_sum = 0
        #     #     if edge_id == 0:
        #     #         weight_sum = cell_weights1[0:2].sum()
        #     #     elif edge_id == 1:
        #     #         weight_sum = cell_weights1[2:4].sum()
        #     #     op_id = 0
        #     #     for w_op in edge_weights:
        #     #         w_normalized = w_op / weight_sum
        #     #         if w_normalized > args.eta_min:
        #     #             if w_normalized < pruning_w:
        #     #                 pruning_cell = cell_id
        #     #                 pruning_edge = edge_id-2
        #     #                 pruning_op = op_id
        #     #                 pruning_w = w_normalized
        #     #                 pruning_w_op = w_op
        #     #         elif EPS < w_normalized <= args.eta_min:
        #     #             pruned_ops += 1
        #     #             logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e', cell_id, edge_id-2, op_id,
        #     #                          w_normalized, w_op)
        #     #             model._arch_parameters[cell_id][0].data[edge_id][op_id] -= MAX
        #     #             weight_sum -= w_op
        #     #         op_id += 1
        #     #     edge_id += 1
        #     # cell_weights2 = torch.sigmoid(model.arch_parameters()[cell_id][1])
        #     cell_weights2 = torch.sigmoid(model.arch_parameters()[cell_id])
        #     edge_id = 0
        #     while edge_id < 14:
        #         edge_weights = cell_weights2[edge_id]
        #         # weight_sum = 0
        #         if edge_id == 0:
        #         # if edge_id < 2:
        #             weight_sum = cell_weights2[0:2].sum()
        #         elif edge_id == 2:
        #         # elif edge_id < 5:
        #             weight_sum = cell_weights2[2:5].sum()
        #         elif edge_id == 5:
        #         # elif edge_id < 9:
        #             weight_sum = cell_weights2[5:9].sum()
        #         elif edge_id == 9:
        #         # else:
        #             weight_sum = cell_weights2[9:14].sum()
        #         op_id = 0
        #         for w_op in edge_weights:
        #             w_normalized = w_op / weight_sum
        #             # if cell_id==0:
        #             #     logging.info(w_op)
        #             #     logging.info(w_normalized)
        #             if w_normalized > args.eta_min:
        #                 if w_normalized < pruning_w:
        #                     pruning_cell = cell_id
        #                     pruning_edge = edge_id
        #                     pruning_op = op_id
        #                     pruning_w = w_normalized
        #                     pruning_w_op = w_op
        #             # elif EPS < w_normalized <= args.eta_min and w_op <= args.eta_max_raw:
        #             elif model._masks[cell_id].data[edge_id][op_id]==1 and w_normalized <= args.eta_min and w_op <= args.eta_max_raw:
        #                 pruned_ops += 1
        #                 # logging.info('************Pruning (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e*************', cell_id,
        #                 #              edge_id, op_id,
        #                 #              w_normalized, w_op)
        #                 # logging.info('sum_weight %e',weight_sum)
        #                 logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e', cell_id, edge_id, op_id,
        #                              w_normalized,w_op)
        #                 # model._arch_parameters[cell_id][1].data[edge_id][op_id] -= MAX
        #                 model._arch_parameters[cell_id].data[edge_id][op_id] -= MAX
        #                 model._masks[cell_id].data[edge_id][op_id] = 0
        #                 weight_sum -= w_op
        #             op_id += 1
        #         edge_id += 1
        for i in range(start,end):
            # logging.info(torch.Tensor(model._masks[i])==0)
            indices = (torch.Tensor(model._masks[i])==0).nonzero()
            cell_weights2 = torch.sigmoid(model.arch_parameters()[i])
            if i == 0:
                weight_sum = cell_weights2[0:2].sum()
            elif i == 2:
                weight_sum = cell_weights2[2:5].sum()
            elif i == 5:
                weight_sum = cell_weights2[5:9].sum()
            elif i == 9:
                weight_sum = cell_weights2[9:14].sum()
            for item in indices:
                if not isinstance(model.cells[i]._ops[item[0]]._ops[item[1]], Zero):
                    edge_weights = cell_weights2[item[0]]
                    w_normalized = edge_weights[item[1]] / weight_sum
                    model._arch_parameters[i].data[item[0]][item[1]] -= MAX
            # if pruning_w > eta_max or pruning_w_op > args.eta_max_raw:
            #     logging.info('*****Too large to prune (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e', pruning_cell, pruning_edge,
            #                  pruning_op, pruning_w,pruning_w_op)
            #     pass
            # else:
                    pruned_ops += 1
                    logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e', i, item[0],
                                 item[1], w_normalized, edge_weights[item[1]])
            # if pruning_edge<0:
            #     model._arch_parameters[pruning_cell][0].data[pruning_edge+2][pruning_op] -= MAX
            # else:
            #     model._arch_parameters[pruning_cell][1].data[pruning_edge][pruning_op] -= MAX
                    stride = 2 if model.cells[i].reduction and item[0] in [0, 1, 2, 3, 5, 6, 9, 10] else 1
                    model.cells[i]._ops[item[0]]._ops[item[1]] = Zero(stride)
                    # if step:
                    #     genotype = model.genotype()
                    #     logging.info('genotype = %s', genotype)
                    if max!=0:
                        if pruned_ops==max:
                            break
            if max != 0:
                if pruned_ops == max:
                    break
        if max != 0:
            if pruned_ops == max:
                break
    return pruned_ops


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.eta_min, args.reg_flops,
                    args.mu)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # utils.save(model, os.path.join(args.save, 'weights_init.pt'))

    # logging.info(model.arch_parameters()[0].dtype)

    # torch.save(model._arch_parameters.to(torch.device('cpu')), os.path.join(args.save, 'arch_param.pth'))
    # model._arch_parameters = torch.load(os.path.join(args.save, 'arch_param.pth'))

    # stage1_bn = ()
    # stage1_weight = ()
    # for i in range(model.stage1_end + 1):
    #     for j in range(len(model.cells[i]._ops)):
    #         for k in range(len(model.cells[i]._ops[j]._ops)):
    #             if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
    #                 stage1_bn = stage1_bn + ((model.cells[i]._ops[j]._ops[k].op[3], 'weight',model._thresholds[i][j][k]),
    #                                          (model.cells[i]._ops[j]._ops[k].op[7], 'weight',model._thresholds[i][j][k]),)
    #                 stage1_weight = stage1_weight + (
    #                     (model.cells[i]._ops[j]._ops[k].op[1], 'weight',model._thresholds[i][j][k]),
    #                     (model.cells[i]._ops[j]._ops[k].op[2], 'weight',model._thresholds[i][j][k]),
    #                     (model.cells[i]._ops[j]._ops[k].op[5], 'weight',model._thresholds[i][j][k]),
    #                     (model.cells[i]._ops[j]._ops[k].op[6], 'weight',model._thresholds[i][j][k]),)
    #             elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
    #                 stage1_bn = stage1_bn + ((model.cells[i]._ops[j]._ops[k].op[3], 'weight',model._thresholds[i][j][k]),)
    #                 stage1_weight = stage1_weight + (
    #                     (model.cells[i]._ops[j]._ops[k].op[1], 'weight',model._thresholds[i][j][k]),
    #                     (model.cells[i]._ops[j]._ops[k].op[2], 'weight',model._thresholds[i][j][k]),)
    # # global_unstructured(
    # #     stage1_bn,
    # #     pruning_method=Unstructured,
    # #     amount=0,
    # #     mode=0
    # # )
    # # global_unstructured(
    # #     stage1_weight,
    # #     pruning_method=Unstructured,
    # #     amount=0,
    # #     mode=0
    # # )
    # for item in stage1_bn:
    #     unstructured(item[0], item[1], amount=0, mode=0)
    # for item in stage1_weight:
    #     unstructured(item[0], item[1], amount=0, mode=0)

    # model._arch_parameters = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'arch_param3_26.npy'), allow_pickle=True)
    # model._kernel_parameters = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'kernel_param3_26.npy'), allow_pickle=True)
    # model._masks = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'mask3_26.npy'), allow_pickle=True)
    # model._masks_k = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'mask_k3_26.npy'), allow_pickle=True)
    # model._masks_w = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'mask_w3_26.npy'), allow_pickle=True)
    # model._thresholds = np.load(
    #     os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'threshold3_26.npy'), allow_pickle=True)
    # model.update_arch()
    # # # model._initialize_masks()

    # model.update_kernel_num(3)
    # model.prune_kernel_update(3)
    # # model.prune_kernel(3)
    # model = model.cuda()

    # utils.load(model, os.path.join("/root/autodl-tmp/search-EXP-20230904-122803", 'weights3_26.pt'))


    # for i in range(model._layers):
    #     for j in range(len(model.cells[0]._ops)):
    #         for k in range(len(model.cells[0]._ops[j]._ops)):
    #             if isinstance(model.cells[0]._ops[j]._ops[k], FactorizedReduce):
    #                 if model._masks_k[i][j+1][k]==[]:
    #                     model._masks_k[i][j + 1][k].append(torch.ones(model.cells[i]._ops[j]._ops[k].c_out).cuda())
    #                 if model._kernel_parameters[i][j+1][k]==[]:
    #                     model._kernel_parameters[i][j + 1][k].append(Variable(torch.zeros(model.cells[i]._ops[j]._ops[k].c_out).cuda(),
    #                                  requires_grad=True))

    # /ubda/home/21041193r/NAS/
    # /ubda/home/16904288r/liuxiaoyun/
    # model._arch_parameters = np.load(
    #     os.path.join("/ubda/home/21041193r/NAS/search-EXP-20280925-005616", 'arch_param1.py'), allow_pickle=True)
    # model._initialize_masks()
    # model._reinitialize_alphas()
    # utils.load(model, os.path.join('/ubda/home/21041193r/NAS/search-EXP-20281017-021733', 'weights_init.pt'))
    # utils.load(model, os.path.join('/ubda/home/16904288r/liuxiaoyun/search-EXP-20281128-170135', 'weights_init.pt'))

    arch_para = []
    stage1_arch_para = []
    stage1_kernel_para = []
    stage1_para = []
    stage1_thre_alpha = []
    # stage1_thre_kernel = [{"params": model._thresholds[0][1][1][1]}]
    # stage1_thre = [{"params": model._thresholds[0][1][1][2]}]
    stage1_para.append({"params": model.stem.parameters()})
    for i in range(model.stage1_end + 1):
        stage1_arch_para.append({"params": model.arch_parameters()[i]})
        stage1_para.append({"params": model.cells[i].parameters()})
        # stage1_thre.append({"params": torch.Tensor(model._thresholds[i])})
        stage1_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
        stage1_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
        stage1_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
        stage1_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
        for j in range(len(model.cells[i]._ops)):
            for k in range(len(model.cells[i]._ops[j]._ops)):
                stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                for l in range(len(model._kernel_parameters[i][j + 1][k])):
                    stage1_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
    arch_para.extend(stage1_arch_para)
    stage1_para.append({"params": model.auxiliary_head1.parameters()})
    stage2_arch_para = stage1_arch_para
    stage2_kernel_para = stage1_kernel_para
    stage2_para = stage1_para
    stage2_thre_alpha = stage1_thre_alpha
    # stage2_thre = stage1_thre
    # stage2_thre_kernel = stage1_thre_kernel
    for i in range(model.stage1_end + 1, model.stage2_end + 1):
        stage2_arch_para.append({"params": model.arch_parameters()[i]})
        stage2_para.append({"params": model.cells[i].parameters()})
        # stage2_thre.append({"params": torch.Tensor(model._thresholds[i])})
        stage2_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
        stage2_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
        stage2_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
        stage2_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
        for j in range(len(model.cells[i]._ops)):
            for k in range(len(model.cells[i]._ops[j]._ops)):
                stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                for l in range(len(model._kernel_parameters[i][j + 1][k])):
                    stage2_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
    arch_para.extend(stage2_arch_para)
    stage2_para.append({"params": model.auxiliary_head2.parameters()})
    stage3_arch_para = stage2_arch_para
    stage3_kernel_para = stage2_kernel_para
    stage3_para = stage2_para
    stage3_thre_alpha = stage2_thre_alpha
    # stage3_thre_kernel = stage2_thre_kernel
    # stage3_thre = stage2_thre
    for i in range(model.stage2_end + 1, model._layers):
        stage3_arch_para.append({"params": model.arch_parameters()[i]})
        stage3_para.append({"params": model.cells[i].parameters()})
        # stage3_thre.append({"params": torch.Tensor(model._thresholds[i])})
        stage3_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
        stage3_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
        stage3_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
        stage3_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
        for j in range(len(model.cells[i]._ops)):
            for k in range(len(model.cells[i]._ops[j]._ops)):
                stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                for l in range(len(model._kernel_parameters[i][j + 1][k])):
                    stage3_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
    arch_para.extend(stage3_arch_para)
    stage3_para.append({"params": model.global_pooling.parameters()})
    stage3_para.append({"params": model.classifier.parameters()})

    optimizer_alpha1 = torch.optim.SGD(
        stage1_arch_para,
        args.learning_rate_alpha[0],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_alpha2 = torch.optim.SGD(
        stage2_arch_para,
        args.learning_rate_alpha[1],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_alpha3 = torch.optim.SGD(
        stage3_arch_para,
        args.learning_rate_alpha[2],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)

    optimizer_omega1 = torch.optim.SGD(
        stage1_para,
        args.learning_rate_omega[0],
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_omega2 = torch.optim.SGD(
        stage2_para,
        args.learning_rate_omega[1],
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_omega3 = torch.optim.SGD(
        stage3_para,
        args.learning_rate_omega[2],
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer_thre1_alpha = torch.optim.SGD(
        stage1_thre_alpha,
        0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_thre2_alpha = torch.optim.SGD(
        stage2_thre_alpha,
        0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_thre3_alpha = torch.optim.SGD(
        stage3_thre_alpha,
        0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)

    optimizer_kernel_alpha1 = torch.optim.SGD(
        stage1_kernel_para,
        args.learning_rate_alpha_kernel[0],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_kernel_alpha2 = torch.optim.SGD(
        stage2_kernel_para,
        args.learning_rate_alpha_kernel[0],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_kernel_alpha3 = torch.optim.SGD(
        stage3_kernel_para,
        args.learning_rate_alpha_kernel[0],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    min_flops = []
    min_flops_ratio = [0.5,0.4,0.3]

    for index in range(3):
        arch_para = []
        stage1_arch_para = []
        stage1_kernel_para = []
        stage1_para = []
        stage1_thre_alpha = []
        stage1_para.append({"params": model.stem.parameters()})
        for i in range(model.stage1_end + 1):
            stage1_arch_para.append({"params": model.arch_parameters()[i]})
            stage1_para.append({"params": model.cells[i].parameters()})
            # stage1_thre.append({"params": torch.Tensor(model._thresholds[i])})
            stage1_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
            stage1_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
            stage1_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
            stage1_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
            for j in range(len(model.cells[i]._ops)):
                for k in range(len(model.cells[i]._ops[j]._ops)):
                    stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                    stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                    stage1_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                    for l in range(len(model._kernel_parameters[i][j + 1][k])):
                        stage1_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
        arch_para.extend(stage1_arch_para)
        stage1_para.append({"params": model.auxiliary_head1.parameters()})
        stage2_arch_para = stage1_arch_para
        stage2_kernel_para = stage1_kernel_para
        stage2_para = stage1_para
        stage2_thre_alpha = stage1_thre_alpha
        # stage2_thre = stage1_thre
        # stage2_thre_kernel = stage1_thre_kernel
        for i in range(model.stage1_end + 1, model.stage2_end + 1):
            stage2_arch_para.append({"params": model.arch_parameters()[i]})
            stage2_para.append({"params": model.cells[i].parameters()})
            # stage2_thre.append({"params": torch.Tensor(model._thresholds[i])})
            stage2_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
            stage2_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
            stage2_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
            stage2_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
            for j in range(len(model.cells[i]._ops)):
                for k in range(len(model.cells[i]._ops[j]._ops)):
                    stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                    stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                    stage2_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                    for l in range(len(model._kernel_parameters[i][j + 1][k])):
                        stage2_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
        arch_para.extend(stage2_arch_para)
        stage2_para.append({"params": model.auxiliary_head2.parameters()})
        stage3_arch_para = stage2_arch_para
        stage3_kernel_para = stage2_kernel_para
        stage3_para = stage2_para
        stage3_thre_alpha = stage2_thre_alpha
        # stage3_thre_kernel = stage2_thre_kernel
        # stage3_thre = stage2_thre
        for i in range(model.stage2_end + 1, model._layers):
            stage3_arch_para.append({"params": model.arch_parameters()[i]})
            stage3_para.append({"params": model.cells[i].parameters()})
            # stage3_thre.append({"params": torch.Tensor(model._thresholds[i])})
            stage3_thre_alpha.append({"params": model._thresholds[i][0][0][1]})
            stage3_thre_alpha.append({"params": model._thresholds[i][0][1][1]})
            stage3_kernel_para.append({"params": model._kernel_parameters[i][0][0][0]})
            stage3_kernel_para.append({"params": model._kernel_parameters[i][0][1][0]})
            for j in range(len(model.cells[i]._ops)):
                for k in range(len(model.cells[i]._ops[j]._ops)):
                    stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][0]})
                    stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][1]})
                    stage3_thre_alpha.append({"params": model._thresholds[i][j + 1][k][2], 'lr': 0.001})
                    for l in range(len(model._kernel_parameters[i][j + 1][k])):
                        stage3_kernel_para.append({"params": model._kernel_parameters[i][j + 1][k][l]})
        arch_para.extend(stage3_arch_para)
        stage3_para.append({"params": model.global_pooling.parameters()})
        stage3_para.append({"params": model.classifier.parameters()})

        optimizer_alpha1 = torch.optim.SGD(
            stage1_arch_para,
            args.learning_rate_alpha[0],
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)
        optimizer_alpha3 = torch.optim.SGD(
            stage3_arch_para,
            args.learning_rate_alpha[2],
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)

        optimizer_omega1 = torch.optim.SGD(
            stage1_para,
            args.learning_rate_omega[0],
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizer_omega3 = torch.optim.SGD(
            stage3_para,
            args.learning_rate_omega[2],
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        optimizer_thre1_alpha = torch.optim.SGD(
            stage1_thre_alpha,
            0.1,
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)
        optimizer_thre2_alpha = torch.optim.SGD(
            stage2_thre_alpha,
            0.1,
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)
        optimizer_thre3_alpha = torch.optim.SGD(
            stage3_thre_alpha,
            0.1,
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)

        optimizer_kernel_alpha1 = torch.optim.SGD(
            stage1_kernel_para,
            args.learning_rate_alpha_kernel[0],
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)
        optimizer_kernel_alpha2 = torch.optim.SGD(
            stage2_kernel_para,
            args.learning_rate_alpha_kernel[0],
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)
        optimizer_kernel_alpha3 = torch.optim.SGD(
            stage3_kernel_para,
            args.learning_rate_alpha_kernel[0],
            momentum=args.momentum,
            weight_decay=args.weight_decay_alpha)

        stage_index = index+1
        # stage_index = 3
        if stage_index==1:
            optimizer_alpha = optimizer_alpha1
            optimizer_kernel_alpha = optimizer_kernel_alpha1
            optimizer_omega = optimizer_omega1
            # optimizer_thre = optimizer_thre1
            # optimizer_thre_kernel = optimizer_thre1_kernel
            optimizer_thre_alpha = optimizer_thre1_alpha
            end = model.stage1_end+1
        elif stage_index==2:
            optimizer_alpha = optimizer_alpha2
            optimizer_kernel_alpha = optimizer_kernel_alpha2
            optimizer_omega = optimizer_omega2
            # optimizer_thre = optimizer_thre2
            # optimizer_thre_kernel = optimizer_thre2_kernel
            optimizer_thre_alpha = optimizer_thre2_alpha
            pre_end = model.stage1_end+1
            end = model.stage2_end + 1
        elif stage_index == 3:
            optimizer_alpha = optimizer_alpha3
            optimizer_kernel_alpha = optimizer_kernel_alpha3
            optimizer_omega = optimizer_omega3
            # optimizer_thre = optimizer_thre3
            # optimizer_thre_kernel = optimizer_thre3_kernel
            optimizer_thre_alpha = optimizer_thre3_alpha
            pre_end = model.stage2_end + 1
            end = model._layers

        model._reinitialize_threshold()

        current_flops = model.current_flops(stage_index)
        logging.info('stage init model flops %e', current_flops)
        min_flops.append(current_flops*min_flops_ratio[index])

        epoch = 0
        flops_lambda = 0
        flops_lambda_delta = args.lambda0
        finished = False
        t = 0
        add_sparsity = False
        eta_max = args.eta_max
        best_acc = 0
        stop_epoch = 0
        can_prune = True
        prune_op_sum = 0
        prune_kernel_sum = 0
        prune_weight_sum = 0
        initial_train_epoch = args.initial_epoch_num_stage
        start = 0
        for epoch in range(start, initial_train_epoch):
            # logging.info("kernel num: " + str(model._kernel_num[0][0][1]))
            # logging.info("weight: " + str(model.cells[0]._ops[0]._ops[1].op[1].weight.shape))
            epoch_start = time.time()
            lr = args.learning_rate_omega[3]
            logging.info('initial epoch %d lr %e flops_weight %e', epoch, lr, flops_lambda)
            model.drop_path_prob = 0
            train_acc, train_obj_acc, train_obj_flops = train_local(train_queue, model, criterion, optimizer_alpha, optimizer_kernel_alpha, optimizer_omega, optimizer_thre_alpha,flops_lambda, stage_index, add_sparsity,
                                                                    only_train=True)
            logging.info('train_acc %f', train_acc)
            logging.info('train_obj_acc %e train_obj_flops %e', train_obj_acc, train_obj_flops)
            epoch_duration = time.time() - epoch_start
            logging.info('epoch time: %ds.', epoch_duration)
            np.save(os.path.join(args.save, 'arch_param'+str(stage_index)+'.npy'), model._arch_parameters)
            np.save(os.path.join(args.save, 'kernel_param' + str(stage_index) + '.npy'), model._kernel_parameters)
            np.save(os.path.join(args.save, 'mask' + str(stage_index) + '.npy'), model._masks)
            utils.save(model, os.path.join(args.save, 'weights'+str(stage_index)+'.pt'))
            np.save(os.path.join(args.save, 'mask_k' + str(stage_index) + '.npy'), model._masks_k)
            np.save(os.path.join(args.save, 'mask_w' + str(stage_index) + '.npy'), model._masks_w)
            np.save(os.path.join(args.save, 'threshold' + str(stage_index) + '.npy'), model._thresholds)
            # torch.save(optimizer_alpha.state_dict, os.path.join(args.save, 'optimizer_alpha'+str(stage_index)+'.pt'))
            # torch.save(optimizer_omega.state_dict, os.path.join(args.save, 'optimizer_omega' + str(stage_index) + '.pt'))


        epoch = 0
        max_flops = args.max_flops_lambda1
        model.drop_path_prob = 0
        while not finished:
            epoch_start = time.time()
            lr = args.learning_rate_omega[index]
            freeze_partial = False
            if epoch % 2 == 0 and epoch != 0:
                freeze_mask = False
                # if epoch % 6!=0:
                #     freeze_partial = True
            else:
                freeze_mask = True
            logging.info('epoch %d lr %e flops_weight %e', epoch, lr, flops_lambda)
            train_acc, train_obj_acc, train_obj_flops = train_local(train_queue, model, criterion, optimizer_alpha, optimizer_kernel_alpha, optimizer_omega,
                                                                    optimizer_thre_alpha, flops_lambda, stage_index, add_sparsity,freeze_mask=freeze_mask,freeze_partial=freeze_partial)
            # logging.info(model._kernel_parameters)
            logging.info('train_acc %f', train_acc)

            # model.prune_kernel(stage_index)
            # model.cuda()

            if train_acc>best_acc:
                best_acc = train_acc
            epoch_duration = time.time() - epoch_start
            logging.info('epoch time: %ds.', epoch_duration)

            if epoch % 2 == 0 and epoch!=0:
                can_prune = True
                if can_prune:
                    pruning_epoch = prune_op(model, args, eta_max, args.pruning_n0_1, stage_index)
                else:
                    pruning_epoch = 0
                prune_op_sum += pruning_epoch

                current_flops = model.current_flops(stage_index)
                current_flops += args.base_flops
                logging.info('current model flops %e', current_flops)

                if train_obj_acc < train_obj_flops:
                    max_flops = flops_lambda
                else:
                    max_flops = 100
                if can_prune:
                    if pruning_epoch >= args.pruning_n_thre1*2:
                        flops_lambda_delta = args.lambda0
                        flops_lambda = flops_lambda / args.c0
                    else:
                        if flops_lambda == max_flops:
                            pass
                        else:
                            flops_lambda_delta = flops_lambda_delta * (args.c0**2)
                            flops_lambda = flops_lambda + flops_lambda_delta
                if flops_lambda > max_flops:
                    flops_lambda = max_flops
                #     eta_max = args.max_eta_max
                #     t = 0
                # if current_flops < min_flops[index]:
                #     finished = True
                if pruning_epoch == 0:
                    t = t + 1
                else:
                    # if t > args.stable_round:
                    genotype = model.genotype()
                    logging.info('genotype = %s', genotype)
                    t = 0

                valid_num = 0
                total_num = 0
                valid_num_stage = 0
                total_num_stage = 0
                for i in range(end):
                    logging.info("cell " + str(i) + " thre: " + str(model._thresholds[i]))
                    valid_num_layer = 0
                    total_num_layer = 0
                    for j in range(len(model.cells[i]._ops)):
                        for k in range(len(model.cells[i]._ops[j]._ops)):
                            if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                                if j == 0:
                                    logging.info(
                                        "kernel_para: " + str(F.sigmoid(model._kernel_parameters[i][j + 1][k][1])))
                                    # logging.info(
                                    #     "kernel_orig: " + str(model.cells[i]._ops[j]._ops[k].op[3].weight))
                                    logging.info("kernel_mask: " + str(model._masks_k[i][j + 1][k][1]))
                                    # logging.info(
                                    #     "thre_kernel: " + str(model._thresholds[i][j][k][1]))
                                    # logging.info(
                                    #     "thre_weight: " + str(model._thresholds[i][j][k][2]))
                                for l in range(len(model._masks_k[i][j + 1][k])//2):
                                    valid_num_layer += torch.sum(model._masks_k[i][j + 1][k][2*l+1])
                                    total_num_layer += model._masks_k[i][j + 1][k][2*l+1].nelement()
                            elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                                for l in range(len(model._masks_k[i][j + 1][k])//2):
                                    valid_num_layer += torch.sum(model._masks_k[i][j + 1][k][2*l+1])
                                    total_num_layer += model._masks_k[i][j + 1][k][2*l+1].nelement()
                    logging.info(
                        "Cell bn " + str(i) + " Sparsity: " + str(
                            100. * float(valid_num_layer) / float(total_num_layer)))
                    valid_num_stage += valid_num_layer
                    total_num_stage += total_num_layer
                    valid_num += valid_num_layer
                    total_num += total_num_layer
                    if i == model.stage1_end:
                        logging.info("Stage bn 1 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i == model.stage2_end:
                        logging.info("Stage bn 2 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i == model._layers - 1:
                        logging.info("Stage bn 3 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                logging.info("Total bn " + str(stage_index) + " Sparsity: " + str(
                    100. * float(valid_num) / float(total_num)))

                valid_num = 0
                total_num = 0
                valid_num_stage = 0
                total_num_stage = 0
                for i in range(end):
                    valid_num_layer = 0
                    total_num_layer = 0
                    for j in range(len(model.cells[i]._ops)):
                        for k in range(len(model.cells[i]._ops[j]._ops)):
                            if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][2])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][3])
                                total_num_layer += model._masks_w[i][j][k][0].nelement()
                                total_num_layer += model._masks_w[i][j][k][1].nelement()
                                total_num_layer += model._masks_w[i][j][k][2].nelement()
                                total_num_layer += model._masks_w[i][j][k][3].nelement()
                            elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                                total_num_layer += model._masks_w[i][j][k][0].nelement()
                                total_num_layer += model._masks_w[i][j][k][1].nelement()
                    logging.info(
                        "Cell " + str(i) + " Sparsity: " + str(100. * float(valid_num_layer) / float(total_num_layer)))
                    valid_num_stage += valid_num_layer
                    total_num_stage += total_num_layer
                    valid_num += valid_num_layer
                    total_num += total_num_layer
                    if i == model.stage1_end:
                        logging.info("Stage 1 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i == model.stage2_end:
                        logging.info("Stage 2 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i == model._layers - 1:
                        logging.info("Stage 3 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                logging.info("Total " + str(stage_index) + " Sparsity: " + str(
                    100. * float(valid_num) / float(total_num)))

                model.prune_kernel(stage_index)
                model = model.cuda()

                np.save(os.path.join(args.save, 'arch_param' + str(stage_index) + '_'+str(epoch)+ '.npy'),
                        model._arch_parameters)
                np.save(os.path.join(args.save, 'kernel_param' + str(stage_index) + '_'+str(epoch)+ '.npy'), model._kernel_parameters)
                np.save(os.path.join(args.save, 'mask' + str(stage_index) + '_'+str(epoch)+ '.npy'), model._masks)
                utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '_'+str(epoch)+ '.pt'))
                np.save(os.path.join(args.save, 'mask_k' + str(stage_index) + '_'+str(epoch)+ '.npy'), model._masks_k)
                np.save(os.path.join(args.save, 'mask_w' + str(stage_index) + '_'+str(epoch)+ '.npy'), model._masks_w)
                np.save(os.path.join(args.save, 'threshold' + str(stage_index) + '_'+str(epoch)+ '.npy'), model._thresholds)

                logging.info("after: ")
                current_flops = model.current_flops(stage_index)
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                valid_num = 0
                total_num = 0
                valid_num_stage = 0
                total_num_stage = 0
                for i in range(end):
                    logging.info("cell " + str(i) + " thre: " + str(model._thresholds[i]))
                    valid_num_layer = 0
                    total_num_layer = 0
                    for j in range(len(model.cells[i]._ops)):
                        for k in range(len(model.cells[i]._ops[j]._ops)):
                            if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                                if j == 0:
                                    logging.info("kernel_para: "+str(F.sigmoid(model._kernel_parameters[i][j+1][k][1])))
                                    # logging.info(
                                    #     "kernel_orig: " + str(model.cells[i]._ops[j]._ops[k].op[3].weight))
                                    logging.info("kernel_mask: "+str(model._masks_k[i][j+1][k][1]))
                                    # logging.info(
                                    #     "thre_kernel: " + str(model._thresholds[i][j][k][1]))
                                    # logging.info(
                                    #     "thre_weight: " + str(model._thresholds[i][j][k][2]))
                                for l in range(len(model._masks_k[i][j + 1][k])//2):
                                    valid_num_layer += torch.sum(model._masks_k[i][j + 1][k][2*l+1])
                                    total_num_layer += model._masks_k[i][j + 1][k][2*l+1].nelement()
                            elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                                for l in range(len(model._masks_k[i][j + 1][k])//2):
                                    valid_num_layer += torch.sum(model._masks_k[i][j + 1][k][2*l+1])
                                    total_num_layer += model._masks_k[i][j + 1][k][2*l+1].nelement()
                    logging.info(
                        "Cell bn " + str(i) + " Sparsity: " + str(100. * float(valid_num_layer) / float(total_num_layer)))
                    valid_num_stage += valid_num_layer
                    total_num_stage += total_num_layer
                    valid_num += valid_num_layer
                    total_num += total_num_layer
                    if i==model.stage1_end:
                        logging.info("Stage bn 1 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i==model.stage2_end:
                        logging.info("Stage bn 2 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i==model._layers-1:
                        logging.info("Stage bn 3 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                logging.info("Total bn " + str(stage_index) + " Sparsity: " + str(
                    100. * float(valid_num) / float(total_num)))
                logging.info('current flops: ' + str(current_flops))
                if current_flops < min_flops[index]:
                    finished = True
                else:
                    finished = False

                current_flops = model.current_flops(stage_index)
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                # logging.info('prune weight num: ' + str(prune_weight_num))
                valid_num = 0
                total_num = 0
                valid_num_stage = 0
                total_num_stage = 0
                for i in range(end):
                    valid_num_layer = 0
                    total_num_layer = 0
                    for j in range(len(model.cells[i]._ops)):
                        for k in range(len(model.cells[i]._ops[j]._ops)):
                            if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][2])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][3])
                                total_num_layer += model._masks_w[i][j][k][0].nelement()
                                total_num_layer += model._masks_w[i][j][k][1].nelement()
                                total_num_layer += model._masks_w[i][j][k][2].nelement()
                                total_num_layer += model._masks_w[i][j][k][3].nelement()
                            elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                                valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                                total_num_layer += model._masks_w[i][j][k][0].nelement()
                                total_num_layer += model._masks_w[i][j][k][1].nelement()
                    logging.info("Cell "+str(i)+" Sparsity: "+str(100. * float(valid_num_layer) / float(total_num_layer)))
                    valid_num_stage += valid_num_layer
                    total_num_stage += total_num_layer
                    valid_num += valid_num_layer
                    total_num += total_num_layer
                    if i==model.stage1_end:
                        logging.info("Stage 1 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i==model.stage2_end:
                        logging.info("Stage 2 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                    if i==model._layers-1:
                        logging.info("Stage 3 Sparsity: " + str(
                            100. * float(valid_num_stage) / float(total_num_stage)))
                        valid_num_stage = 0
                        total_num_stage = 0
                logging.info("Total " + str(stage_index) + " Sparsity: " + str(
                    100. * float(valid_num) / float(total_num)))
                logging.info('current flops: ' + str(current_flops))
                if current_flops < min_flops[index]:
                    finished = True

                # if epoch % 6==0 and epoch!=0 and finished==False:
                #     logging.info('recover epoch %d', 0)
                #     train_acc, train_obj_acc, train_obj_flops = train_local(train_queue, model, criterion, optimizer_alpha,
                #                                                             optimizer_kernel_alpha, optimizer_omega,
                #                                                             optimizer_thre_alpha, 0, stage_index,
                #                                                             False,
                #                                                             only_train=True)
                #     logging.info('train_acc %f', train_acc)
                #     logging.info('train_obj_acc %e train_obj_flops %e', train_obj_acc, train_obj_flops)

            epoch += 1
            # finished = True
            logging.info('prune op sum %d', prune_op_sum)
            np.save(os.path.join(args.save, 'arch_param' + str(stage_index) + '.npy'),
                    model._arch_parameters)
            np.save(os.path.join(args.save, 'kernel_param' + str(stage_index) + '.npy'), model._kernel_parameters)
            np.save(os.path.join(args.save, 'mask' + str(stage_index) + '.npy'), model._masks)
            utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '.pt'))
            np.save(os.path.join(args.save, 'mask_k' + str(stage_index) + '.npy'), model._masks_k)
            np.save(os.path.join(args.save, 'mask_w' + str(stage_index) + '.npy'), model._masks_w)
            np.save(os.path.join(args.save, 'threshold' + str(stage_index) + '.npy'), model._thresholds)
            torch.save(optimizer_alpha.state_dict,
                       os.path.join(args.save, 'optimizer_alpha' + str(stage_index) + '.pt'))
            torch.save(optimizer_omega.state_dict,
                       os.path.join(args.save, 'optimizer_omega' + str(stage_index) + '.pt'))


        model.prune_kernel(stage_index)
        model = model.cuda()
        np.save(os.path.join(args.save, 'arch_param' + str(stage_index) + '.npy'),
                model._arch_parameters)
        np.save(os.path.join(args.save, 'kernel_param' + str(stage_index) + '.npy'), model._kernel_parameters)
        np.save(os.path.join(args.save, 'mask' + str(stage_index) + '.npy'), model._masks)
        utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '.pt'))
        np.save(os.path.join(args.save, 'mask_k' + str(stage_index) + '.npy'), model._masks_k)
        np.save(os.path.join(args.save, 'mask_w' + str(stage_index) + '.npy'), model._masks_w)
        np.save(os.path.join(args.save, 'threshold' + str(stage_index) + '.npy'), model._thresholds)

        if stage_index>1:
            epoch = 0
            logging.info('regrow epoch %d', epoch)
            train_acc, train_obj_acc, train_obj_flops = train_local(train_queue, model, criterion, optimizer_alpha,optimizer_kernel_alpha,
                                                                    optimizer_omega, optimizer_thre_alpha, flops_lambda, stage_index, False,freeze_mask=True)
            logging.info('train_acc %f', train_acc)
            regrow_sum, op_list = regrow(model,int(prune_op_sum*args.regrow_ratio),optimizer_alpha,stage_index)
            logging.info('regrow num: %d', regrow_sum)

        if stage_index==1 or stage_index==2:
            model.prune_kernel(stage_index+1)
        else:
            model.prune_kernel(stage_index)
        model = model.cuda()

    optimizer_alpha_all = torch.optim.SGD(
        stage3_arch_para,
        args.learning_rate_alpha[3],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_kernel_alpha_all = torch.optim.SGD(
        stage3_kernel_para,
        args.learning_rate_alpha_kernel[3],
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_omega_all = torch.optim.SGD(
        model.parameters(),
        args.learning_rate_omega[3],
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_thre_alpha_all = torch.optim.SGD(
        stage3_thre_alpha,
        0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)

    epoch = 0
    flops_lambda = 0
    flops_lambda_delta = args.lambda0
    finished = False
    t = 0
    eta_max = args.eta_max
    prune_epoch_sum = 0
    last_acc = 0
    stop_epoch = 0
    can_prune = True
    initial_train_epoch = args.initial_epoch_num
    init_flops, current_flops1, current_flops2 = model.current_flops()
    min_flops_round = []
    epoch = 0
    for i in range(initial_train_epoch):
        epoch_start = time.time()
        lr = args.learning_rate_omega[3]
        logging.info('initial epoch %d lr %e flops_weight %e', epoch, lr, flops_lambda)
        model.drop_path_prob = 0
        train_acc, train_obj_acc, train_obj_flops  = train(train_queue, model, criterion, optimizer_alpha_all, optimizer_kernel_alpha_all, optimizer_omega_all, optimizer_thre_alpha_all,
                                     flops_lambda,only_train=True)
        logging.info('train_acc %f', train_acc)
        logging.info('train_obj_acc %e train_obj_flops %e', train_obj_acc, train_obj_flops)
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)
        np.save(os.path.join(args.save, 'arch_param.npy'), model._arch_parameters)
        np.save(os.path.join(args.save, 'kernel_param.npy'), model._kernel_parameters)
        np.save(os.path.join(args.save, 'mask.npy'), model._masks)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        np.save(os.path.join(args.save, 'mask_k.npy'), model._masks_k)
        np.save(os.path.join(args.save, 'mask_w.npy'), model._masks_w)
        np.save(os.path.join(args.save, 'threshold.npy'), model._thresholds)
        torch.save(optimizer_alpha_all.state_dict, os.path.join(args.save, 'optimizer_alpha.pt'))
        torch.save(optimizer_omega_all.state_dict, os.path.join(args.save, 'optimizer_omega.pt'))
        epoch += 1

    epoch = 0
    flops_lambda = 1e-4
    max_flops = args.max_flops_lambda2
    while not finished:
        epoch_start = time.time()
        lr = args.learning_rate_omega[3]
        model.drop_path_prob = 0
        logging.info('epoch %d lr %e flops_weight %e', epoch, lr, flops_lambda)
        freeze_partial = False
        if epoch % 2 == 0 and epoch != 0:
            freeze_mask = False
            # if epoch%6!=0 and epoch!=24:
            #     freeze_partial = True
        else:
            freeze_mask = True
        train_acc, train_obj_acc, train_obj_flops = train(train_queue, model, criterion, optimizer_alpha_all, optimizer_kernel_alpha_all, optimizer_omega_all, optimizer_thre_alpha_all,
                                     flops_lambda, freeze_mask=freeze_mask,freeze_partial=freeze_partial)
        logging.info('train_acc %f', train_acc)
        logging.info('train_obj_acc %e train_obj_flops %e', train_obj_acc,train_obj_flops)
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)

        if epoch % 2==0 and epoch!=0:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            pruning_epoch = 0
            while True:
                prune_num = prune_op(model, args, 100, 1, step=True, max=1)
                pruning_epoch += prune_num
                current_flops, current_flops1, current_flops2 = model.current_flops()
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                if prune_num == 0 or current_flops < args.min_flops:
                    break
            prune_epoch_sum += pruning_epoch
            current_flops, current_flops1, current_flops2 = model.current_flops()
            current_flops += args.base_flops
            logging.info('current model flops %e', current_flops)
            if train_obj_acc < args.max_flops_ratio * train_obj_flops:
                max_flops = flops_lambda
            else:
                max_flops = 100
        # if can_prune:
            if pruning_epoch >= args.pruning_n_thre2*2:
                flops_lambda_delta = args.lambda0
                flops_lambda = flops_lambda / args.c0
            else:
                if flops_lambda == max_flops:
                    pass
                else:
                    flops_lambda_delta = flops_lambda_delta * args.c0
                    flops_lambda = flops_lambda + flops_lambda_delta
            if flops_lambda > max_flops:
                flops_lambda = max_flops
            if current_flops < args.min_flops:
                finished = True
            if pruning_epoch == 0:
                t = t + 1
            else:
                # if t > args.stable_round:
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                logging.info('prune sum: %d', prune_epoch_sum)
                t = 0


        # if epoch % 2==0:
            model.prune_kernel()
            model = model.cuda()
            np.save(os.path.join(args.save, 'arch_param_'+str(epoch)+'.npy'), model._arch_parameters)
            np.save(os.path.join(args.save, 'kernel_param_'+str(epoch)+'.npy'), model._kernel_parameters)
            np.save(os.path.join(args.save, 'mask_'+str(epoch)+'.npy'), model._masks)
            utils.save(model, os.path.join(args.save, 'weights_'+str(epoch)+'.pt'))
            np.save(os.path.join(args.save, 'mask_k_'+str(epoch)+'.npy'), model._masks_k)
            np.save(os.path.join(args.save, 'mask_w_'+str(epoch)+'.npy'), model._masks_w)
            np.save(os.path.join(args.save, 'threshold_'+str(epoch)+'.npy'), model._thresholds)
            valid_num = 0
            total_num = 0
            for i in range(model._layers):
                valid_num_layer = 0
                total_num_layer = 0
                for j in range(len(model.cells[i]._ops)):
                    for k in range(len(model.cells[i]._ops[j]._ops)):
                        if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                            valid_num_layer += torch.sum(model._masks_k[i][j+1][k][1])
                            valid_num_layer += torch.sum(model._masks_k[i][j+1][k][3])
                            total_num_layer += model._masks_k[i][j+1][k][1].nelement()
                            total_num_layer += model._masks_k[i][j+1][k][3].nelement()
                        elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                            valid_num_layer += torch.sum(model._masks_k[i][j+1][k][1])
                            total_num_layer += model._masks_k[i][j+1][k][1].nelement()
                if total_num_layer!=0:
                    logging.info(
                        "Cell bn " + str(i) + " Sparsity: " + str(100. * float(valid_num_layer) / float(total_num_layer)))
                else:
                    logging.info(
                        "Cell bn " + str(i) + " Sparsity: no conv")
                valid_num += valid_num_layer
                total_num += total_num_layer
            current_flops, current_flops1, current_flops2 = model.current_flops()
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            logging.info('current flops: ' + str(current_flops))

            valid_num = 0
            total_num = 0
            for i in range(model._layers):
                valid_num_layer = 0
                total_num_layer = 0
                for j in range(len(model.cells[i]._ops)):
                    for k in range(len(model.cells[i]._ops[j]._ops)):
                        if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][2])
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][3])
                            total_num_layer += model._masks_w[i][j][k][0].nelement()
                            total_num_layer += model._masks_w[i][j][k][1].nelement()
                            total_num_layer += model._masks_w[i][j][k][2].nelement()
                            total_num_layer += model._masks_w[i][j][k][3].nelement()
                        elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][0])
                            valid_num_layer += torch.sum(model._masks_w[i][j][k][1])
                            total_num_layer += model._masks_w[i][j][k][0].nelement()
                            total_num_layer += model._masks_w[i][j][k][1].nelement()
                if total_num_layer != 0:
                    logging.info(
                        "Cell " + str(i) + " Sparsity: " + str(100. * float(valid_num_layer) / float(total_num_layer)))
                else:
                    logging.info(
                        "Cell " + str(i) + " Sparsity: no conv")
                valid_num += valid_num_layer
                total_num += total_num_layer
            # logging.info("Stage " + str(stage_index) + " Sparsity: " + str(
            #     100. * float(valid_num) / float(total_num)))
            current_flops, current_flops1, current_flops2 = model.current_flops()
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            logging.info('current flops: ' + str(current_flops))
            if current_flops < args.min_flops:
                finished = True

            # if epoch%6==0 and epoch!=0:
            #     logging.info('recover epoch %d', 0)
            #     train_acc, train_obj_acc, train_obj_flops = train(train_queue, model, criterion, optimizer_alpha_all,
            #                                                       optimizer_kernel_alpha_all, optimizer_omega_all,
            #                                                       optimizer_thre_alpha_all,
            #                                                       0, only_train=True)
            #     logging.info('train_acc %f', train_acc)
            #     logging.info('train_obj_acc %e train_obj_flops %e', train_obj_acc, train_obj_flops)

        epoch += 1

def get_momentum_for_weight(optimizer, weight):
    # logging.info('optimizer: '+str(optimizer.state[weight]))
    if 'exp_avg' in optimizer.state[weight]:
        adam_m1 = optimizer.state[weight]['exp_avg']
        adam_m2 = optimizer.state[weight]['exp_avg_sq']
        grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
    elif 'momentum_buffer' in optimizer.state[weight]:
        grad = optimizer.state[weight]['momentum_buffer']
    # logging.info('grad: ' + str(grad))
    # else:
    #     grad = torch.zeros(weight.shape)
    return grad

def regrow(model, regrow_num, optimizer, stage_index=0):
    importance_ops_list,magnitude_ops_list,grad_ops_list = [],[],[]

    # stage-wise
    # importance_stage, magnitude_stage, grad_stage = [], [], []
    # split = [0,model.stage1_end+1,model.stage2_end+1,model._layers]
    # for i in range(len(split)-1):
    #     importance_ops,grad_ops,magnitude_ops = [],[],[]
    #     for j in range(len(PRIMITIVES)):
    #         sum_grad = 0
    #         num_grad = 0
    #         sum_magnitude = 0
    #         for k in range(split[i],split[i+1]):
    #             # print(model._arch_parameters[k][:,j])
    #             grad = get_momentum_for_weight(optimizer,model._arch_parameters[k])
    #             grad = grad[:,j]
    #             sum_grad += torch.abs(grad[model._masks[k][:,j].bool()]).sum()
    #             # logging.info('grad: ' + str(grad))
    #             num_grad += model._masks[k][:,j].sum()
    #             sum_magnitude += torch.abs(torch.sigmoid(model._arch_parameters[k][:,j][model._masks[k][:,j].bool()])).sum()
    #         if num_grad==0:
    #             mean_grad = 0
    #             mean_magnitude = 0
    #             tmp = 0
    #         else:
    #             mean_grad = sum_grad/num_grad
    #             # logging.info("mean_grad: "+str(num_grad.data))
    #             mean_magnitude = sum_magnitude/(split[i+1]-split[i])*model._masks[0].shape[0]
    #             # logging.info("mean_magnitude: " + str(mean_magnitude))
    #             # print((split[i+1]-split[i])*model._masks[0].shape[0])
    #             # tmp = mean_grad+mean_magnitude
    #             # logging.info("tmp: " + str(tmp))
    #         # logging.info('mean grad: '+str(mean_grad))
    #         # logging.info('mean magnitude: ' + str(mean_magnitude))
    #         # importance_ops.append(tmp)
    #         grad_ops.append(mean_grad)
    #         magnitude_ops.append(mean_magnitude)

    # layer-wise
    importance_layer, magnitude_layer, grad_layer = [], [], []
    if stage_index==1:
        end = model.stage1_end + 1
    elif stage_index==2:
        end = model.stage2_end+1
        # end = model.stage1_end + 1
    else:
        end = model._layers
        # end = model.stage2_end + 1

    # sum = 0
    # num = 0
    # for i in range(end):
    #     for j in range(len(model.cells[i]._ops)):
    #         for k in range(len(model.cells[i]._ops[j]._ops)):
    #             if isinstance(model.cells[i]._ops[j]._ops[k], SepConv):
    #                 sum += (model.cells[i]._ops[j]._ops[k].op[3].weight * model._masks_k[i][j][k][0]).sum() + (
    #                             model.cells[i]._ops[j]._ops[k].op[7].weight * model._masks_k[i][j][k][1]).sum()
    #                 num += model._masks_k[i][j][k][0].sum() + model._masks_k[i][j][k][1].sum()
    #             elif isinstance(model.cells[i]._ops[j]._ops[k], DilConv):
    #                 sum += (model.cells[i]._ops[j]._ops[k].op[3].weight * model._masks_k[i][j][k][0]).sum()
    #                 num += model._masks_k[i][j][k][0].sum()
    # avg = sum / num

    for i in range(end):
        importance_ops, grad_ops, magnitude_ops = [], [], []
        min_mean_grad = 100000
        min_mean_magnitude = 100000
        for j in range(len(PRIMITIVES)):
            # print(model._arch_parameters[k][:,j])
            grad = get_momentum_for_weight(optimizer, model._arch_parameters[i])
            grad = grad[:, j]
            sum_grad = torch.abs(grad[model._masks[i][:, j].bool()]).sum()
            # logging.info('grad: ' + str(grad))
            num_grad = model._masks[i][:, j].sum()
            sum_magnitude = torch.abs(
                torch.sigmoid(model._arch_parameters[i][:, j][model._masks[i][:, j].bool()])).sum()
            if num_grad == 0:
                mean_grad = 0
                mean_magnitude = 0
                tmp = 0
            else:
                mean_grad = sum_grad / num_grad
                if mean_grad < min_mean_grad:
                    min_mean_grad = mean_grad
                # logging.info("mean_grad: "+str(num_grad.data))
                # mean_magnitude = sum_magnitude / model._masks[0].shape[0]
                mean_magnitude = sum_magnitude / num_grad
                if mean_magnitude < min_mean_magnitude:
                    min_mean_magnitude = mean_magnitude
                # logging.info("mean_magnitude: " + str(mean_magnitude))
                # print((split[i+1]-split[i])*model._masks[0].shape[0])
                # tmp = mean_grad+mean_magnitude
                # logging.info("tmp: " + str(tmp))
            # logging.info('mean grad: '+str(mean_grad))
            # logging.info('mean magnitude: ' + str(mean_magnitude))
            # importance_ops.append(tmp)
            grad_ops.append(mean_grad)
            magnitude_ops.append(mean_magnitude)

        for j in range(len(PRIMITIVES)):
            if grad_ops[j]==0:
                grad_ops[j] = min_mean_grad/2
                magnitude_ops[j] = min_mean_magnitude/2

        # importance_ops = torch.Tensor(importance_ops)
        # logging.info('raw ops importance before divide: ' + str(importance_ops))
        # importance_ops = (importance_ops/importance_ops.sum())
        # logging.info('raw ops importance after divide: ' + str(importance_ops))
        # importance_ops_list.append(importance_ops)
        magnitude_ops = torch.Tensor(magnitude_ops)
        sum_magnitude = magnitude_ops.sum()
        # logging.info('raw ops magnitude before divide: ' + str(magnitude_ops))
        magnitude_ops = (magnitude_ops / magnitude_ops.sum())
        # logging.info('raw ops magnitude after divide: ' + str(magnitude_ops))
        magnitude_ops_list.append(magnitude_ops)
        grad_ops = torch.Tensor(grad_ops)
        sum_grad = grad_ops.sum()
        # logging.info('raw ops grad before divide: ' + str(grad_ops))
        grad_ops = (grad_ops / grad_ops.sum())
        # logging.info('raw ops grad after divide: ' + str(grad_ops))
        grad_ops_list.append(grad_ops)
        importance_ops = 0.5 * magnitude_ops + 0.5 * grad_ops
        # importance_ops = grad_ops
        # logging.info('raw ops importance after divide: ' + str(importance_ops))
        importance_ops_list.append(importance_ops)

        # stage-wise
        # sum_grad = 0
        # num_grad = 0
        # sum_magnitude = 0
        # for k in range(split[i],split[i+1]):
        #     grad = get_momentum_for_weight(optimizer,model._arch_parameters[k])
        #     sum_grad += torch.abs(grad[model._masks[k].bool()]).sum()
        #     num_grad += model._masks[k].sum()
        #     sum_magnitude += torch.abs(torch.sigmoid(model._arch_parameters[k][model._masks[k].bool()])).sum()
        # mean_grad = sum_grad / num_grad
        # mean_magnitude = sum_magnitude / (split[i + 1] - split[i]) * model._masks[0].shape[0] * model._masks[0].shape[1]
        # grad_stage.append(mean_grad)
        # magnitude_stage.append(mean_magnitude)

        # layer-wise
        # sum_grad = grad_ops.sum()
        num_grad = model._masks[i].sum()
        # sum_magnitude = magnitude_ops.sum()
        mean_grad = sum_grad/num_grad
        # mean_magnitude = sum_magnitude/model._masks[0].shape[0]*model._masks[0].shape[1]
        mean_magnitude = sum_magnitude / num_grad
        grad_layer.append(mean_grad)
        magnitude_layer.append(mean_magnitude)

    # logging.info('raw stage importance: ' + str(importance_stage))
    # importance_stage = torch.Tensor(importance_stage)
    # importance_stage = importance_stage / importance_stage.sum()
    logging.info('raw ops importance: ' + str(importance_ops_list))
    logging.info('raw ops magnitude: ' + str(magnitude_ops_list))
    logging.info('raw ops grad: ' + str(grad_ops_list))

    # stage-wise
    # grad_stage = torch.Tensor(grad_stage)
    # grad_stage = grad_stage / grad_stage.sum()
    # magnitude_stage = torch.Tensor(magnitude_stage)
    # magnitude_stage = magnitude_stage / magnitude_stage.sum()
    # importance_stage = 0.5*grad_stage+0.5*magnitude_stage
    # # importance_stage = magnitude_stage
    # logging.info('raw stage importance: ' + str(importance_stage))
    # logging.info('raw stage magnitude: ' + str(magnitude_stage))
    # logging.info('raw stage grad: ' + str(grad_stage))

    # layer-wise
    grad_layer = torch.Tensor(grad_layer)
    grad_layer = grad_layer / grad_layer.sum()
    magnitude_layer = torch.Tensor(magnitude_layer)
    magnitude_layer = magnitude_layer / magnitude_layer.sum()
    importance_layer = 0.5*grad_layer+0.5*magnitude_layer
    # importance_layer = grad_layer
    logging.info('raw layer importance: ' + str(importance_layer))
    logging.info('raw layer magnitude: ' + str(magnitude_layer))
    logging.info('raw layer grad: ' + str(grad_layer))

    regrow_prob = torch.rand(regrow_num,)
    logging.info('regrow prob: '+str(regrow_prob))
    importance_sum = 0
    regrow_sum = 0

    # stage-wise
    # for i in range(len(split)-1):
    #     importance_ops_list[i] = importance_stage[i].item()*importance_ops_list[i]
    #     for j in range(len(PRIMITIVES)):
    #         importance_sum_pre = importance_sum
    #         importance_sum = importance_sum + importance_ops_list[i][j]
    #         importance_ops_list[i][j] = importance_sum
    #         # logging.info('sum1: >'+str(importance_sum_pre)+', '+str((regrow_prob>=importance_sum_pre).sum()))
    #         # logging.info('sum2: >' + str(importance_sum) + ', ' + str((regrow_prob >= importance_sum).sum()))
    #         regrow_num_tmp = (regrow_prob>=importance_sum_pre).sum() - (regrow_prob>=importance_sum).sum()
    #         if regrow_num_tmp>0:
    #             logging.info('stage importance: ' + str(importance_stage))
    #             logging.info('ops importance: ' + str(importance_ops_list))
    #             # logging.info('position (stage, ops): '+str(i)+', '+str(j))
    #             regrow_position_tmp1 = torch.nonzero(regrow_prob>=importance_sum_pre)
    #             regrow_position_tmp2 = torch.nonzero(regrow_prob>=importance_sum)
    #             regrow_position_tmp = [item for item in regrow_position_tmp1 if item not in regrow_position_tmp2]
    #             # logging.info('position (regrow_prob): ' + str(regrow_position_tmp))
    #             for item in regrow_position_tmp:
    #                 empty_num = 0
    #                 empty_index = []
    #                 for k in range(split[i], split[i + 1]):
    #                     empty_num += (model._masks[k][:, j] == 0).sum()
    #                     empty_index_tmp = torch.nonzero(model._masks[k][:, j] == 0).cpu().numpy().tolist()
    #                     for item2 in empty_index_tmp:
    #                         item2.append(j)
    #                         item2.append(k)
    #                     empty_index.extend(empty_index_tmp)
    #                 if empty_num == 0:
    #                     break
    #                 # logging.info('position (empty): ' + str(empty_index))
    #                 prob_list = torch.arange(start=0, end=empty_num.item(), step=1 / empty_num.item()).cuda()
    #                 regrow_prob_tmp = (regrow_prob[item]-importance_sum_pre)/(importance_sum-importance_sum_pre)
    #                 # logging.info('revised regrow prob: ' + str(regrow_prob[item]))
    #                 position1 = torch.nonzero((regrow_prob_tmp.cuda()-prob_list)<1/empty_num)
    #                 position2 = torch.nonzero((regrow_prob_tmp.cuda()-prob_list)<0)
    #                 position = [items for items in position1 if items not in position2]
    #                 # logging.info('position: '+str(position))
    #                 model._arch_parameters[empty_index[position[0]][2]][empty_index[position[0]][0], empty_index[position[0]][1]].data.fill_(0)
    #                 model._masks[empty_index[position[0]][2]][empty_index[position[0]][0], empty_index[position[0]][1]] = 1
    #                 logging.info('Regrowing (cell, edge, op) = (%d, %d, %d): ', empty_index[position[0]][2], empty_index[position[0]][0], empty_index[position[0]][1])
    #                 regrow_sum += 1
    # logging.info('stage importance: '+str(importance_stage))
    # logging.info('ops importance: '+str(importance_ops_list))

    # layer-wise
    model.update_kernel_num(stage_index)
    if stage_index == 1:
        end = model.stage1_end + 1
    elif stage_index == 2:
        end = model.stage2_end+1
        # end = model.stage1_end + 1
    else:
        end = model._layers
        # end = model.stage2_end + 1
    op_list = []
    for i in range(end):
        importance_ops_list[i] = importance_layer[i].item()*importance_ops_list[i]
        for j in range(len(PRIMITIVES)):
            importance_sum_pre = importance_sum
            importance_sum = importance_sum + importance_ops_list[i][j]
            importance_ops_list[i][j] = importance_sum
            # logging.info('sum1: >'+str(importance_sum_pre)+', '+str((regrow_prob>=importance_sum_pre).sum()))
            # logging.info('sum2: >' + str(importance_sum) + ', ' + str((regrow_prob >= importance_sum).sum()))
            regrow_num_tmp = (regrow_prob>=importance_sum_pre).sum() - (regrow_prob>=importance_sum).sum()
            if regrow_num_tmp>0:
                logging.info('layer importance: ' + str(importance_layer))
                logging.info('ops importance: ' + str(importance_ops_list))
                # logging.info('position (stage, ops): '+str(i)+', '+str(j))
                regrow_position_tmp1 = torch.nonzero(regrow_prob>=importance_sum_pre)
                regrow_position_tmp2 = torch.nonzero(regrow_prob>=importance_sum)
                regrow_position_tmp = [item for item in regrow_position_tmp1 if item not in regrow_position_tmp2]
                # logging.info('position (regrow_prob): ' + str(regrow_position_tmp))
                for item in regrow_position_tmp:
                    empty_num = (model._masks[i][:, j] == 0).sum()
                    empty_index = torch.nonzero(model._masks[i][:, j] == 0).cpu().numpy().tolist()
                    for item2 in empty_index:
                        item2.append(j)
                        item2.append(i)
                    if empty_num == 0:
                        break
                    # logging.info('position (empty): ' + str(empty_index))
                    prob_list = torch.arange(start=0, end=empty_num.item(), step=1 / empty_num.item()).cuda()
                    regrow_prob_tmp = (regrow_prob[item] - importance_sum_pre) / (importance_sum - importance_sum_pre)
                    # logging.info('revised regrow prob: ' + str(regrow_prob[item]))
                    position1 = torch.nonzero((regrow_prob_tmp.cuda() - prob_list) < 1 / empty_num)
                    position2 = torch.nonzero((regrow_prob_tmp.cuda() - prob_list) < 0)
                    position = [items for items in position1 if items not in position2]
                    # logging.info('position: '+str(position))
                    MAX = 10000.0
                    cell_id = empty_index[position[0]][2]
                    edge_id = empty_index[position[0]][0]
                    op_id = empty_index[position[0]][1]
                    model._arch_parameters[cell_id].data[edge_id][op_id] += MAX
                    model._masks[cell_id][
                        edge_id, op_id] = 1
                    stride = 2 if model.cells[cell_id].reduction and edge_id in [0,1,2,3,5,6,9,10] else 1
                    C = model.cells[cell_id].C
                    if edge_id in [0,2,5,9]:
                        step_start = 0
                    elif edge_id in [1,3,6,10]:
                        step_start = 1
                    elif edge_id in [4,7,11]:
                        step_start = 2
                    elif edge_id in [8,12]:
                        step_start = 3
                    else:
                        step_start = 4
                    if edge_id in [0,1]:
                        step_end = 2
                    elif edge_id in [2,3,4]:
                        step_end = 3
                    elif edge_id in [5,6,7,8]:
                        step_end = 4
                    elif edge_id in [9,10,11,12,13]:
                        step_end = 5
                    logging.info('Regrowing (cell, edge, op) = (%d, %d, %d): ', cell_id,
                                 edge_id, op_id)
                    op_list.append([cell_id, edge_id, op_id])

                    if 'sep_conv' in PRIMITIVES[op_id]:
                        model.cells[cell_id]._ops[edge_id]._ops[op_id] = OPS[
                            PRIMITIVES[op_id]](int(model._kernel_sum[cell_id][step_start]), int(model._kernel_sum[cell_id][step_start]),
                                           int(model._kernel_sum[cell_id][step_end]), stride, True).cuda()
                        model._masks_k[cell_id][edge_id+1][op_id] = []
                        model._masks_k[cell_id][
                            edge_id+1][op_id].append([])
                        model._masks_k[cell_id][
                            edge_id+1][op_id].append(torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape[0]).cuda())
                        model._masks_k[cell_id][
                            edge_id + 1][op_id].append([])
                        model._masks_k[cell_id][
                            edge_id + 1][op_id].append(
                            torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[6].weight.shape[0]).cuda())
                        model._kernel_parameters[cell_id][edge_id + 1][op_id] = []
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append([])
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append(
                            Variable(torch.zeros(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape[0]).cuda(),
                                     requires_grad=True))
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append([])
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append(
                            Variable(torch.zeros(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[6].weight.shape[0]).cuda(),
                                     requires_grad=True))
                        model._masks_w[cell_id][edge_id][op_id] = []
                        model._masks_w[cell_id][edge_id][op_id].append(torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[1].weight.shape).cuda())
                        model._masks_w[cell_id][edge_id][op_id].append(torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape).cuda())
                        model._masks_w[cell_id][edge_id][op_id].append(torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[5].weight.shape).cuda())
                        model._masks_w[cell_id][edge_id][op_id].append(torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[6].weight.shape).cuda())
                        # model.cells[cell_id]._ops[edge_id]._ops[op_id].op[3].weight.data.fill_(0.5)
                        # model.cells[cell_id]._ops[edge_id]._ops[op_id].op[7].weight.data.fill_(0.5)
                    elif 'dil_conv' in PRIMITIVES[op_id]:
                        try:
                            model.cells[cell_id]._ops[edge_id]._ops[op_id] = OPS[
                                PRIMITIVES[op_id]](int(model._kernel_sum[cell_id][step_start]),
                                                   int(model._kernel_sum[cell_id][step_end]), stride, True).cuda()
                        except Exception as e:
                            logging.info(model._kernel_num[cell_id][edge_id][op_id])
                        model._masks_k[cell_id][edge_id+1][op_id] = []
                        model._masks_k[cell_id][
                            edge_id+1][op_id].append([])
                        model._masks_k[cell_id][
                            edge_id + 1][op_id].append(
                            torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape[0]).cuda())
                        model._kernel_parameters[cell_id][edge_id + 1][op_id] = []
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append([])
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append(
                            Variable(torch.zeros(
                                model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape[0]).cuda(),
                                     requires_grad=True))
                        model._masks_w[cell_id][edge_id][op_id] = []
                        model._masks_w[cell_id][edge_id][op_id].append(
                            torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[1].weight.shape).cuda())
                        model._masks_w[cell_id][edge_id][op_id].append(
                            torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape).cuda())
                        # model.cells[cell_id]._ops[edge_id]._ops[op_id].op[3].weight.data.fill_(0.5)
                    elif 'skip' in PRIMITIVES[op_id] and stride!=1:
                        model.cells[cell_id]._ops[edge_id]._ops[op_id] = FactorizedReduce(int(model._kernel_sum[cell_id][step_start]),
                                                                         int(model._kernel_sum[cell_id][step_start]),
                                                                         affine=True).cuda()
                        model._masks_k[cell_id][edge_id + 1][op_id] = []
                        model._masks_k[cell_id][
                            edge_id + 1][op_id].append(
                            torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].c_out).cuda())
                        model._kernel_parameters[cell_id][edge_id + 1][op_id] = []
                        model._kernel_parameters[cell_id][edge_id + 1][op_id].append(
                            Variable(torch.zeros(
                                model.cells[cell_id]._ops[edge_id]._ops[op_id].c_out).cuda(),
                                     requires_grad=True))
                    else:
                        model.cells[cell_id]._ops[edge_id]._ops[op_id] = OPS[PRIMITIVES[op_id]](
                            int(model._kernel_sum[cell_id][step_start]),stride,
                            affine=True).cuda()
                    model.cells[cell_id]._ops[edge_id]._ops[op_id].reset()
                    regrow_sum += 1

    logging.info('layer importance: '+str(importance_layer))
    logging.info('ops importance: '+str(importance_ops_list))

    # layer-wise no ops importance
    # importance_layer = importance_layer.cpu().numpy().tolist()
    # op_list = []
    # for i in range(end):
    #     importance_sum_pre = importance_sum
    #     importance_sum = importance_sum + importance_layer[i]
    #     importance_layer[i] = importance_sum
    #     # logging.info('sum1: >'+str(importance_sum_pre)+', '+str((regrow_prob>=importance_sum_pre).sum()))
    #     # logging.info('sum2: >' + str(importance_sum) + ', ' + str((regrow_prob >= importance_sum).sum()))
    #     regrow_num_tmp = (regrow_prob>=importance_sum_pre).sum() - (regrow_prob>=importance_sum).sum()
    #     if regrow_num_tmp>0:
    #         logging.info('layer importance: ' + str(importance_layer))
    #         # logging.info('position (stage, ops): '+str(i)+', '+str(j))
    #         regrow_position_tmp1 = torch.nonzero(regrow_prob>=importance_sum_pre)
    #         regrow_position_tmp2 = torch.nonzero(regrow_prob>=importance_sum)
    #         regrow_position_tmp = [item for item in regrow_position_tmp1 if item not in regrow_position_tmp2]
    #         # logging.info('position (regrow_prob): ' + str(regrow_position_tmp))
    #         for item in regrow_position_tmp:
    #             empty_num = (model._masks[i] == 0).sum()
    #             empty_index = torch.nonzero(model._masks[i] == 0).cpu().numpy().tolist()
    #             for item2 in empty_index:
    #                 item2.append(i)
    #             if empty_num == 0:
    #                 break
    #             # logging.info('position (empty): ' + str(empty_index))
    #             prob_list = torch.arange(start=0, end=empty_num.item(), step=1 / empty_num.item()).cuda()
    #             regrow_prob_tmp = (regrow_prob[item] - importance_sum_pre) / (importance_sum - importance_sum_pre)
    #             # logging.info('revised regrow prob: ' + str(regrow_prob[item]))
    #             position1 = torch.nonzero((regrow_prob_tmp.cuda() - prob_list) < 1 / empty_num)
    #             position2 = torch.nonzero((regrow_prob_tmp.cuda() - prob_list) < 0)
    #             position = [items for items in position1 if items not in position2]
    #             # logging.info('position: '+str(position))
    #             MAX = 10000.0
    #             cell_id = empty_index[position[0]][2]
    #             edge_id = empty_index[position[0]][0]
    #             op_id = empty_index[position[0]][1]
    #             model._arch_parameters[cell_id].data[edge_id][op_id] += MAX
    #             model._masks[cell_id][edge_id, op_id] = 1
    #             stride = 2 if model.cells[cell_id].reduction and edge_id in [0,1,2,3,5,6,9,10] else 1
    #             C = model.cells[cell_id].C
    #             model.cells[cell_id]._ops[edge_id]._ops[op_id] = OPS[PRIMITIVES[op_id]](C, stride, True).cuda()
    #             model.cells[cell_id]._ops[edge_id]._ops[op_id].reset()
    #             logging.info('Regrowing (cell, edge, op) = (%d, %d, %d): ', cell_id,
    #                          edge_id, op_id)
    #             # if isinstance(model.cells[empty_index[position[0]][2]]._ops[empty_index[position[0]][0]]._ops[empty_index[position[0]][1]], SepConv) or isinstance(
    #             #         model.cells[empty_index[position[0]][2]]._ops[empty_index[position[0]][0]]._ops[empty_index[position[0]][1]], DilConv):
    #             #     logging.info("("+str(empty_index[position[0]][2])+","+str(empty_index[position[0]][0])+","+str(empty_index[position[0]][1])+"), has mask: " + str(hasattr(model.cells[empty_index[position[0]][2]]._ops[empty_index[position[0]][0]]._ops[empty_index[position[0]][1]].op[1], "weight_mask")))
    #             op_list.append([cell_id, edge_id, op_id])
    #
    #             if isinstance(model.cells[cell_id]._ops[edge_id]._ops[
    #                               op_id], SepConv):
    #                 model._masks_k[cell_id][edge_id][op_id] = []
    #                 model._masks_k[cell_id][
    #                     edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[3].weight.shape).cuda())
    #                 model._masks_k[cell_id][
    #                     edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[7].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id] = []
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[1].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[5].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[6].weight.shape).cuda())
    #             if isinstance(model.cells[cell_id]._ops[edge_id]._ops[op_id], DilConv):
    #                 model._masks_k[cell_id][edge_id][op_id] = []
    #                 model._masks_k[cell_id][
    #                     edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[3].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id] = []
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[1].weight.shape).cuda())
    #                 model._masks_w[cell_id][edge_id][op_id].append(
    #                     torch.ones(model.cells[cell_id]._ops[edge_id]._ops[op_id].op[2].weight.shape).cuda())
    #
    #             # if isinstance(model.cells[empty_index[position[0]][2]]._ops[empty_index[position[0]][0]]._ops[
    #             #                   empty_index[position[0]][1]], SepConv) or isinstance(
    #             #     model.cells[empty_index[position[0]][2]]._ops[empty_index[position[0]][0]]._ops[
    #             #         empty_index[position[0]][1]], DilConv):
    #             #     for num in range(len(model._masks_k[empty_index[position[0]][2]][
    #             #                              empty_index[position[0]][0], empty_index[position[0]][1]])):
    #             #         model._masks_k[empty_index[position[0]][2]][
    #             #             empty_index[position[0]][0]][empty_index[position[0]][1]][num].data.fill_(1)
    #             #     for num in range(len(model._masks_w[empty_index[position[0]][2]][
    #             #                              empty_index[position[0]][0], empty_index[position[0]][1]])):
    #             #         model._masks_w[empty_index[position[0]][2]][
    #             #             empty_index[position[0]][0]][empty_index[position[0]][1]][num].data.fill_(1)
    #
    #             regrow_sum += 1

    # print(importance_ops_list)
    # thre_sum = 0
    # for i in range(model._layers):
    #     thre_masks.append(torch.zeros(model._masks[0].shape[0], model._masks[0].shape[1]))
    #     # new_masks.append(torch.rand(model._masks[0].shape[0], model._masks[0].shape[1]))
    #     # print(new_masks)
    #     for j in range(len(PRIMITIVES)):
    #             # threshold_masks[k][:, j] = torch.full((threshold_masks[k].shape[0],),importance_ops_list[i][j])
    #         for k in range(len(split)-1):
    #             if i>=split[k] and i<split[k+1]:
    #                 stage_index=k
    #         thre_sum += +importance_ops_list[stage_index][j]
    #         thre_masks[i][:, j] = thre_sum
            # new_masks[i][:,j] = new_masks[i][:,j] < importance_ops_list[stage_index][j]

    # new_masks = new_masks < threshold_masks
    # logging.info(new_masks)

    # regrow_sum = 0
    # for i in range(model._layers):
    #     regrow_num = ((new_masks[i].cuda()-model._masks[i])==1).sum()
    #     regrow_sum += regrow_num
    #     for item in torch.nonzero((new_masks[i].cuda()-model._masks[i])==1):
    #         logging.info('Regrowing (cell, edge, op) = (%d, %d, %d): ', i, item[0], item[1])
    #     model._arch_parameters[i][(new_masks[i].cuda()-model._masks[i])==1].data.fill_(0)
    #     # logging.info(new_masks[i].cuda()-model._masks[i]==1)
    #     # model.cells[i].parameters()[new_masks[i].cuda()-model._masks[i])==1].data.fill_()
    #     model._masks[i][(new_masks[i].cuda()-model._masks[i])==1] = 1
    # logging.info(model._masks)
    return regrow_sum, op_list

def train(train_queue, model, criterion, optimizer_alpha,optimizer_kernel_alpha,optimizer_omega,optimizer_thre_alpha,flops_lambda,only_train=False,freeze_mask=True,freeze_partial=False):
# def train(train_queue, model, criterion, optimizer_alpha,optimizer_alpha1,optimizer_alpha2,optimizer_alpha3, optimizer_omega,optimizer_omega1,optimizer_omega2,optimizer_omega3,flops_lambda, stage1_lambda, stage2_lambda, stage3_lambda):
    objs = utils.AvgrageMeter()
    objs_acc = utils.AvgrageMeter()
    objs_flops = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    update_mask = False
    update_partial = False
    for step, (input, target) in enumerate(train_queue):
        if step==len(train_queue)-1 and freeze_mask==False:
            update_mask = True
            if freeze_partial:
                update_partial = True
        n = input.size(0)
        # get a random minibatch from the search queue with replacement
        # with torch.autograd.set_detect_anomaly(True):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        # torch.autograd.set_detect_anomaly(True)

        try:
            logits1, logits2, logits3, loss_flops, loss_thre = model(input,update_mask=update_mask,update_partial=update_partial)
        except Exception as e:
            logging.info("batch: "+str(step))
            logits1, logits2, logits3, loss_flops, loss_thre = model(input, update_mask=update_mask,update_partial=update_partial, log=True)
        # loss_aux1 = criterion(logits1, target)
        # loss_aux2 = criterion(logits2, target)
        loss_final = criterion(logits3, target)

        if only_train==False:
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.zero_grad()
            # optimizer_thre.zero_grad()
            # optimizer_thre_kernel.zero_grad()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.zero_grad()
        # loss = 0.2*loss_aux1 + 0.4*loss_aux2 + loss_final
        # loss += flops_lambda * loss_flops
        loss = loss_final + flops_lambda * (loss_flops+loss_thre)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if only_train == False:
            optimizer_alpha.step()
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.step()
            optimizer_kernel_alpha.zero_grad()
            # optimizer_thre.step()
            # optimizer_thre.zero_grad()
            # optimizer_thre_kernel.step()
            # optimizer_thre_kernel.zero_grad()
            optimizer_thre_alpha.step()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.step()
        optimizer_omega.zero_grad()

        prec1, correct = utils.accuracy(logits3, target, topk=(1,))
        objs.update(loss.data.item(), n)
        objs_acc.update(loss_final.data.item(), n)
        objs_flops.update(flops_lambda * loss_flops.data.item(), n)
        top1.update(prec1[0].data.item(), n)
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)
        torch.cuda.empty_cache()
        # break
    return top1.avg, objs_acc.avg, objs_flops.avg

def train_local(train_queue, model, criterion, optimizer_alpha, optimizer_kernel_alpha, optimizer_omega, optimizer_thre_alpha, flops_lambda, stage_index, add_sparsity, only_train=False,freeze_mask=True,freeze_partial=False):
    objs = utils.AvgrageMeter()
    objs_acc = utils.AvgrageMeter()
    objs_flops = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    # logits1_all,logits2_all,logits3_all, target_all = [], [],[], []
    model.train()
    update_mask = False
    update_partial = False
    for step, (input, target) in enumerate(train_queue):
        if step == len(train_queue) - 1 and freeze_mask == False:
            update_mask = True
            if freeze_partial:
                update_partial = True
        n = input.size(0)
        # get a random minibatch from the search queue with replacement
        # with torch.autograd.set_detect_anomaly(True):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        avg1 = torch.full(model.arch_parameters()[0].shape, 0.5).cuda()
        # avg2 = torch.full(model.arch_parameters()[0][1].shape, 0.5).cuda()
        loss_sparsity = 0
        if stage_index==1:
            logits1, loss_flops, loss_thre = model(input,stage_index,update_mask=update_mask,gradual_discretization = False)
            # final_logits, logits1, loss_flops = model(input,stage_index,gradual_discretization = False)
            logits = logits1
            # logits1_all.extend(logits1.cpu().detach().numpy())
            if add_sparsity:
                for i in range(model.stage1_end+1):
                    arch_param1 = torch.sigmoid(model.arch_parameters()[i])
                    # arch_param2 = torch.sigmoid(model.arch_parameters()[i][1])
                    loss_sparsity += - (F.mse_loss(arch_param1,avg1))
        elif stage_index==2:
            logits1,logits2, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False)
            # final_logits, logits1, logits2, loss_flops = model(input, stage_index, gradual_discretization=False)
            logits = logits2
            # logits1_all.extend(logits1.cpu().detach().numpy())
            # logits2_all.extend(logits2.cpu().detach().numpy())
            if add_sparsity:
                for i in range(model.stage1_end+1,model.stage2_end+1):
                    arch_param1 = torch.sigmoid(model.arch_parameters()[i])
                    # arch_param2 = torch.sigmoid(model.arch_parameters()[i][1])
                    loss_sparsity += - (F.mse_loss(arch_param1,avg1) )
        elif stage_index==3:
            try:
                logits1, logits2,logits3, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False)
            except Exception as e:
                logging.info("batch: " + str(step))
                logits1, logits2, logits3, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False, log=True)
            logits = logits3
            # logits1_all.extend(logits1.cpu().detach().numpy())
            # logits2_all.extend(logits2.cpu().detach().numpy())
            # logits3_all.extend(logits3.cpu().detach().numpy())
            if add_sparsity:
                for i in range(model.stage2_end + 1, model._layers):
                    arch_param1 = torch.sigmoid(model.arch_parameters()[i])
                    # arch_param2 = torch.sigmoid(model.arch_parameters()[i][1])
                    loss_sparsity += - (F.mse_loss(arch_param1,avg1))
        # target_all.extend(target)
        loss_aux = criterion(logits, target)
        # if stage_index!=3:
        #     loss_final = criterion(final_logits, target)
        if only_train==False:
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.zero_grad()
            # optimizer_thre.zero_grad()
            # optimizer_thre_kernel.zero_grad()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.zero_grad()
        # if stage_index != 3:
        #     loss = loss_aux + loss_final + flops_lambda * loss_flops
        # else:
        loss = loss_aux + flops_lambda * (loss_flops+loss_thre)
        if add_sparsity:
            loss += 1*loss_sparsity
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if only_train == False:
            optimizer_alpha.step()
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.step()
            optimizer_kernel_alpha.zero_grad()
            # optimizer_thre.step()
            # optimizer_thre.zero_grad()
            # optimizer_thre_kernel.step()
            # optimizer_thre_kernel.zero_grad()
            optimizer_thre_alpha.step()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.step()
        optimizer_omega.zero_grad()

        prec1, correct = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.data.item(), n)
        objs_acc.update(loss_aux.data.item(), n)
        objs_flops.update(flops_lambda * loss_flops.data.item(), n)
        top1.update(prec1[0].data.item(), n)
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)
            # current_flops = model.current_flops(stage_index)
            # logging.info('current flops:'+str(current_flops))
            # logging.info("thre: " + str(model._thresholds[0][0]))
        torch.cuda.empty_cache()
        # if step==2:
        # break
    return top1.avg, objs_acc.avg, objs_flops.avg
    # if stage_index == 1:
    #     return top1.avg, objs.avg, logits1_all, target_all
    # elif stage_index == 2:
    #     return top1.avg, objs.avg, logits1_all, logits2_all, target_all
    # elif stage_index == 3:
    #     return top1.avg, objs.avg, logits1_all, logits2_all, logits3_all, target_all

def train_omega(train_queue, model, criterion, optimizer_omega, stage_index):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top1_initial = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        # get a random minibatch from the search queue with replacement
        # with torch.autograd.set_detect_anomaly(True):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        if stage_index == 1:
            logits, loss_flops = model(input,stage_index,gradual_discretization = True,discretization_include_current=True)
            # logits_initial, loss_flops_initial = model(input, stage_index, gradual_discretization=True)
        elif stage_index == 2:
            logits1, logits, loss_flops = model(input, stage_index,gradual_discretization = True,discretization_include_current=True)
            # logits1_initial, logits_initial, loss_flops_initial = model(input, stage_index, gradual_discretization=True)
        elif stage_index==3:
            logits1, logits2,logits, loss_flops = model(input, stage_index,gradual_discretization = True,discretization_include_current=True)
            # logits1_initial, logits2_initial, logits_initial, loss_flops_initial = model(input, stage_index, gradual_discretization=True)
        loss = criterion(logits, target)
        optimizer_omega.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_omega.step()
        optimizer_omega.zero_grad()

        prec1, correct = utils.accuracy(logits, target, topk=(1,))
        # prec1_initial, correct_initial = utils.accuracy(logits_initial, target, topk=(1,))
        objs.update(loss.data.item(), n)
        top1.update(prec1[0].data.item(), n)
        # top1_initial.update(prec1_initial[0].data.item(), n)
        if step % args.report_freq == 0:
            # logging.info('train_initial %03d %f', step, top1_initial.avg)
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)
        torch.cuda.empty_cache()
    return top1.avg,top1_initial.avg, objs.avg

# def infer(valid_queue, model, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.eval()
#
#     for step, (input, target) in enumerate(valid_queue):
#         input = Variable(input, volatile=True).cuda()
#         target = Variable(target, volatile=True).cuda()
#         logits, logits_aux1, logits_aux2 = model(input)
#         loss = criterion(logits, target)
#         loss_aux1 = criterion(logits_aux1, target)
#         loss_aux2 = criterion(logits_aux2, target)
#         loss += loss_aux1 + loss_aux2
#         prec1_1, correct = utils.accuracy(logits_aux1, target, topk=(1,))
#         prec11, correct = utils.accuracy(logits_aux2, target, topk=(1,))
#         prec1_3, correct = utils.accuracy(logits, target, topk=(1,))
#         prec1 = (prec1_1 + prec11 + prec1_3) / 3
#         # prec5 = (prec5_1 + prec51 + prec5_3) / 3
#         n = input.size(0)
#         objs.update(loss.data[0], n)
#         top1.update(prec1.data[0], n)
#         # top5.update(prec5.data[0], n)
#         if step % args.report_freq == 0:
#             logging.info('valid %03d %e %f %f %f %f', step, objs.avg, top1.avg)
#
#     return top1.avg, objs.avg

def infer(valid_queue, model, criterion,stage_index=0):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda()
            if stage_index == 1:
                logits, loss_flops,loss_thre = model(input, stage_index, gradual_discretization=True,discretization_include_current=True)
            elif stage_index == 2:
                logits1, logits, loss_flops,loss_thre = model(input, stage_index, gradual_discretization=True,discretization_include_current=True)
            elif stage_index == 3:
                logits1, logits2, logits, loss_flops,loss_thre = model(input, stage_index, gradual_discretization=True,discretization_include_current=True)
            else:
                logits1, logits2, logits, loss_flops,loss_thre = model(input)
            loss = criterion(logits, target)
            prec1, correct = utils.accuracy(logits, target, topk=(1,))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1[0].data.item(), n)
            # top5.update(prec5.data[0], n)
            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()