import os
import sys
import time
import glob

import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import utils
import argparse
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model_search import Network
from operations import *

from genotypes import *

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
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
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
    pruned_ops = 0
    for x in range(pruning_n0):
        start = 0
        end = args.layers
        if stage_index==1:
            end = model.stage1_end+1
        elif stage_index==2:
            end = model.stage2_end+1
        elif stage_index==3:
            end = args.layers
        for i in range(start,end):
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

                    pruned_ops += 1
                    logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e raw_weight %e', i, item[0],
                                 item[1], w_normalized, edge_weights[item[1]])
                    stride = 2 if model.cells[i].reduction and item[0] in [0, 1, 2, 3, 5, 6, 9, 10] else 1
                    model.cells[i]._ops[item[0]]._ops[item[1]] = Zero(stride)
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

    arch_para = []
    stage1_arch_para = []
    stage1_kernel_para = []
    stage1_para = []
    stage1_thre_alpha = []
    stage1_para.append({"params": model.stem.parameters()})
    for i in range(model.stage1_end + 1):
        stage1_arch_para.append({"params": model.arch_parameters()[i]})
        stage1_para.append({"params": model.cells[i].parameters()})
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
    for i in range(model.stage1_end + 1, model.stage2_end + 1):
        stage2_arch_para.append({"params": model.arch_parameters()[i]})
        stage2_para.append({"params": model.cells[i].parameters()})
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
    for i in range(model.stage2_end + 1, model._layers):
        stage3_arch_para.append({"params": model.arch_parameters()[i]})
        stage3_para.append({"params": model.cells[i].parameters()})
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
    min_flops_ratio = [0.3, 0.3, 0.3]

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
        for i in range(model.stage1_end + 1, model.stage2_end + 1):
            stage2_arch_para.append({"params": model.arch_parameters()[i]})
            stage2_para.append({"params": model.cells[i].parameters()})
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
        for i in range(model.stage2_end + 1, model._layers):
            stage3_arch_para.append({"params": model.arch_parameters()[i]})
            stage3_para.append({"params": model.cells[i].parameters()})
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

        stage_index = index+1
        if stage_index==1:
            optimizer_alpha = optimizer_alpha1
            optimizer_kernel_alpha = optimizer_kernel_alpha1
            optimizer_omega = optimizer_omega1
            optimizer_thre_alpha = optimizer_thre1_alpha
            end = model.stage1_end+1
        elif stage_index==2:
            optimizer_alpha = optimizer_alpha2
            optimizer_kernel_alpha = optimizer_kernel_alpha2
            optimizer_omega = optimizer_omega2
            optimizer_thre_alpha = optimizer_thre2_alpha
            end = model.stage2_end + 1
        elif stage_index == 3:
            optimizer_alpha = optimizer_alpha3
            optimizer_kernel_alpha = optimizer_kernel_alpha3
            optimizer_omega = optimizer_omega3
            optimizer_thre_alpha = optimizer_thre3_alpha
            end = model._layers

        model._reinitialize_threshold()

        current_flops = model.current_flops(stage_index)
        logging.info('stage init model flops %e', current_flops)
        min_flops.append(current_flops*min_flops_ratio[index])

        flops_lambda = 0
        flops_lambda_delta = args.lambda0
        finished = False
        t = 0
        add_sparsity = False
        eta_max = args.eta_max
        prune_op_sum = 0
        initial_train_epoch = args.initial_epoch_num_stage
        start = 0
        for epoch in range(start, initial_train_epoch):
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
            torch.save(model._arch_parameters, os.path.join(args.save, 'arch_param' + str(stage_index) + '_init.npy'))
            torch.save(model._kernel_parameters, os.path.join(args.save, 'kernel_param' + str(stage_index) + '_init.npy'))
            torch.save(model._masks, os.path.join(args.save, 'mask' + str(stage_index) + '_init.npy'))
            utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '_init.pt'))
            torch.save(model._masks_k, os.path.join(args.save, 'mask_k' + str(stage_index) + '_init.npy'))
            torch.save(model._masks_w, os.path.join(args.save, 'mask_w' + str(stage_index) + '_init.npy'))
            torch.save(model._thresholds, os.path.join(args.save, 'threshold' + str(stage_index) + '_init.npy'))

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
            logging.info('train_acc %f', train_acc)

            # model.prune_kernel(stage_index)
            # model.cuda()

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
                    if pruning_epoch >= args.pruning_n_thre1*4:
                        flops_lambda_delta = args.lambda0
                        flops_lambda = flops_lambda / args.c0
                        # pass
                    elif pruning_epoch < args.pruning_n_thre1*2:
                        if flops_lambda == max_flops:
                            pass
                        else:
                            flops_lambda_delta = flops_lambda_delta * (args.c0**2)
                            flops_lambda = flops_lambda + flops_lambda_delta
                if flops_lambda > max_flops:
                    flops_lambda = max_flops

                if pruning_epoch == 0:
                    t = t + 1
                else:
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

                # model.prune_kernel(stage_index)
                # model = model.cuda()

                torch.save(model._arch_parameters, os.path.join(args.save, 'arch_param' + str(stage_index) + '_' + str(epoch) + '.npy'))
                torch.save(model._kernel_parameters, os.path.join(args.save, 'kernel_param' + str(stage_index) + '_' + str(epoch) + '.npy'))
                torch.save(model._masks, os.path.join(args.save, 'mask' + str(stage_index) + '_' + str(epoch) + '.npy'))
                utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '_' + str(epoch) + '.pt'))
                torch.save(model._masks_k, os.path.join(args.save, 'mask_k' + str(stage_index) + '_' + str(epoch) + '.npy'))
                torch.save(model._masks_w, os.path.join(args.save, 'mask_w' + str(stage_index) + '_' + str(epoch) + '.npy'))
                torch.save(model._thresholds, os.path.join(args.save, 'threshold' + str(stage_index) + '_' + str(epoch) + '.npy'))

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
                                    logging.info("kernel_mask: "+str(model._masks_k[i][j+1][k][1]))
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
            logging.info('prune op sum %d', prune_op_sum)

        torch.save(model._arch_parameters, os.path.join(args.save, 'arch_param' + str(stage_index) + '.npy'))
        torch.save(model._kernel_parameters, os.path.join(args.save, 'kernel_param' + str(stage_index) + '.npy'))
        torch.save(model._masks, os.path.join(args.save, 'mask' + str(stage_index) + '.npy'))
        utils.save(model, os.path.join(args.save, 'weights' + str(stage_index) + '.pt'))
        torch.save(model._masks_k, os.path.join(args.save, 'mask_k' + str(stage_index) + '.npy'))
        torch.save(model._masks_w, os.path.join(args.save, 'mask_w' + str(stage_index) + '.npy'))
        torch.save(model._thresholds, os.path.join(args.save, 'threshold' + str(stage_index) + '.npy'))

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

    model._reinitialize_threshold()

    epoch = 0
    flops_lambda = 0
    flops_lambda_delta = args.lambda0
    finished = False
    t = 0
    eta_max = args.eta_max
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
        torch.save(model._arch_parameters, os.path.join(args.save, 'arch_param_init.npy'))
        torch.save(model._kernel_parameters, os.path.join(args.save, 'kernel_param_init.npy'))
        torch.save(model._masks, os.path.join(args.save, 'mask_init.npy'))
        utils.save(model, os.path.join(args.save, 'weights_init.pt'))
        torch.save(model._masks_k, os.path.join(args.save, 'mask_k_init.npy'))
        torch.save(model._masks_w, os.path.join(args.save, 'mask_w_init.npy'))
        torch.save(model._thresholds, os.path.join(args.save, 'threshold_init.npy'))
        epoch += 1

    epoch = 0
    flops_lambda = 0
    max_flops = args.max_flops_lambda2
    prune_epoch_sum = 0
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
            if pruning_epoch >= args.pruning_n_thre1 * 4:
                flops_lambda_delta = args.lambda0
                flops_lambda = flops_lambda / args.c0
                # pass
            elif pruning_epoch < args.pruning_n_thre1 * 2:
                if flops_lambda == max_flops:
                    pass
                else:
                    flops_lambda_delta = flops_lambda_delta * (args.c0 ** 2)
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

            model.prune_kernel()
            model = model.cuda()
            torch.save(model._arch_parameters, os.path.join(args.save, 'arch_param_' + str(epoch) + '.npy'))
            torch.save(model._kernel_parameters, os.path.join(args.save, 'kernel_param_' + str(epoch) + '.npy'))
            torch.save(model._masks, os.path.join(args.save, 'mask_' + str(epoch) + '.npy'))
            utils.save(model, os.path.join(args.save, 'weights_' + str(epoch) + '.pt'))
            torch.save(model._masks_k, os.path.join(args.save, 'mask_k_' + str(epoch) + '.npy'))
            torch.save(model._masks_w, os.path.join(args.save, 'mask_w_' + str(epoch) + '.npy'))
            torch.save(model._thresholds, os.path.join(args.save, 'threshold_' + str(epoch) + '.npy'))

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
            current_flops, current_flops1, current_flops2 = model.current_flops()
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            logging.info('current flops: ' + str(current_flops))
            if current_flops < args.min_flops:
                finished = True

        epoch += 1

def get_momentum_for_weight(optimizer, weight):
    # logging.info('optimizer: '+str(optimizer.state[weight]))
    if 'exp_avg' in optimizer.state[weight]:
        adam_m1 = optimizer.state[weight]['exp_avg']
        adam_m2 = optimizer.state[weight]['exp_avg_sq']
        grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
    elif 'momentum_buffer' in optimizer.state[weight]:
        grad = optimizer.state[weight]['momentum_buffer']
    return grad

def regrow(model, regrow_num, optimizer, stage_index=0):
    importance_ops_list,magnitude_ops_list,grad_ops_list = [],[],[]

    # layer-wise
    importance_layer, magnitude_layer, grad_layer = [], [], []
    if stage_index==1:
        end = model.stage1_end + 1
    elif stage_index==2:
        end = model.stage2_end+1
    else:
        end = model._layers

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
                mean_magnitude = sum_magnitude / num_grad
                if mean_magnitude < min_mean_magnitude:
                    min_mean_magnitude = mean_magnitude
            grad_ops.append(mean_grad)
            magnitude_ops.append(mean_magnitude)

        for j in range(len(PRIMITIVES)):
            if grad_ops[j]==0:
                grad_ops[j] = min_mean_grad/2
                magnitude_ops[j] = min_mean_magnitude/2

        magnitude_ops = torch.Tensor(magnitude_ops)
        sum_magnitude = magnitude_ops.sum()
        magnitude_ops = (magnitude_ops / magnitude_ops.sum())
        magnitude_ops_list.append(magnitude_ops)
        grad_ops = torch.Tensor(grad_ops)
        sum_grad = grad_ops.sum()
        grad_ops = (grad_ops / grad_ops.sum())
        grad_ops_list.append(grad_ops)
        importance_ops = 0.5 * magnitude_ops + 0.5 * grad_ops
        importance_ops_list.append(importance_ops)

        # layer-wise
        num_grad = model._masks[i].sum()
        mean_grad = sum_grad/num_grad
        mean_magnitude = sum_magnitude / num_grad
        grad_layer.append(mean_grad)
        magnitude_layer.append(mean_magnitude)

    logging.info('raw ops importance: ' + str(importance_ops_list))
    logging.info('raw ops magnitude: ' + str(magnitude_ops_list))
    logging.info('raw ops grad: ' + str(grad_ops_list))

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

    # layer-wise
    model.update_kernel_num(stage_index)
    if stage_index == 1:
        end = model.stage1_end + 1
    elif stage_index == 2:
        end = model.stage2_end+1
    else:
        end = model._layers
    op_list = []
    for i in range(end):
        importance_ops_list[i] = importance_layer[i].item()*importance_ops_list[i]
        for j in range(len(PRIMITIVES)):
            importance_sum_pre = importance_sum
            importance_sum = importance_sum + importance_ops_list[i][j]
            importance_ops_list[i][j] = importance_sum
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

    return regrow_sum, op_list

def train(train_queue, model, criterion, optimizer_alpha,optimizer_kernel_alpha,optimizer_omega,optimizer_thre_alpha,flops_lambda,only_train=False,freeze_mask=True,freeze_partial=False):
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

        if update_mask:
            log = True
            logging.info("log1 True")
        else:
            log = False
        try:
            logits1, logits2, logits3, loss_flops, loss_thre = model(input,update_mask=update_mask,update_partial=update_partial,log=log)
        except Exception as e:
            logging.info("batch: "+str(step))
            logits1, logits2, logits3, loss_flops, loss_thre = model(input, update_mask=update_mask,update_partial=update_partial, log=True)

        loss_final = criterion(logits3, target)

        if only_train==False:
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.zero_grad()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.zero_grad()
        loss = loss_final + flops_lambda * (loss_flops+loss_thre)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if only_train == False:
            optimizer_alpha.step()
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.step()
            optimizer_kernel_alpha.zero_grad()
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
    return top1.avg, objs_acc.avg, objs_flops.avg

def train_local(train_queue, model, criterion, optimizer_alpha, optimizer_kernel_alpha, optimizer_omega, optimizer_thre_alpha, flops_lambda, stage_index, add_sparsity, only_train=False,freeze_mask=True,freeze_partial=False):
    objs = utils.AvgrageMeter()
    objs_acc = utils.AvgrageMeter()
    objs_flops = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
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
        loss_sparsity = 0
        if stage_index==1:
            logits1, loss_flops, loss_thre = model(input,stage_index,update_mask=update_mask,gradual_discretization = False)
            logits = logits1
        elif stage_index==2:
            logits1,logits2, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False)
            logits = logits2
        elif stage_index==3:
            try:
                logits1, logits2,logits3, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False)
            except Exception as e:
                logging.info("batch: " + str(step))
                logits1, logits2, logits3, loss_flops, loss_thre = model(input, stage_index,update_mask=update_mask,update_partial=update_partial,gradual_discretization = False, log=True)
            logits = logits3

        loss_aux = criterion(logits, target)
        if only_train==False:
            optimizer_alpha.zero_grad()
            optimizer_kernel_alpha.zero_grad()
            optimizer_thre_alpha.zero_grad()
        optimizer_omega.zero_grad()
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
        torch.cuda.empty_cache()
    return top1.avg, objs_acc.avg, objs_flops.avg

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