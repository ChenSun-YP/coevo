import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os

import logging
from config import *
from utils import *

import time
import argparse
import sys
import datetime
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from spikingjelly.spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


from spikingjelly.spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.spikingjelly.activation_based.model import spiking_vgg

from spikingjelly.spikingjelly.activation_based.model import spiking_resnet
writer = SummaryWriter('runs/snn_experiment_1')


class SNN(nn.Module):
    #simple sequential network
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


if __name__ =="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # minist_net = torch.load('logs_from_spikingjelly/T100_b64_adam_lr0.001_amp/checkpoint_max.pth',map_location=device)

    #
    # model = SNN(tau=2.0)  # Your Model Class Here
    # model.load_state_dict(minist_net['net'])
    # summary(model, (1, 28, 28), batch_size=1, device="cpu")
    # net = model

    vgg = spiking_vgg.spiking_vgg16(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    summary(vgg,(3, 224, 224) , batch_size=1, device="cpu")
    total1 =[]
    total =0
    # print(vgg.features)
    # print(vgg.avgpool)
    # print(vgg.classifier)
    for layer in vgg.features:
        # print(layer)
        if isinstance(layer, nn.Conv2d):
            print('conv2d')
            CCEP_snn.generate_initial_pop(layer.shape)
        if isinstance(layer, neuron.IFNode):
            print(layer.shape)
    exit()
    for m in vgg.modules():
        print('---')
        print(m.classifier)
        print('---')

        if isinstance(m, nn.BatchNorm2d):
            total.append(m.weight.data.shape[0])
            total += m.weight.data.shape[0]
    print(total,total1)
    print(vgg.modules()[0])

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    conv2d_5 = vgg.conv2d
    # prune.random_unstructured(net,name="weight", amount=0.3) #频闭30%
    # prune.ln_structured(net, name="weight", amount=0.5, n=2, dim=0)
    mask = torch.ones(10, 784)
    mask[0][0] = 0
    prune.custom_from_mask(
        net, name='weight', mask=mask
    )

    print(list(net.named_parameters()))
    print('----------------------oo----')
    print(mask.shape)
    # mask = prune.BasePruningMethod.compute_mask()
    print("buffer", list(net.named_buffers()))

    print(net._forward_pre_hooks)

    # for hook in net._forward_pre_hooks.values():
    #     if hook._tensor_name == "weight":
    #        break
    # print(list(hook))
    #
    #args from config TODO
    parser = argparse.ArgumentParser(description='Pruning Neural Network by CCEA for CCEPSNN')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--dict_path', dest='dict_path', default='./save_model/', type=str)
    # parser.add_argument('-b', '--batch-size', default=256, type=int,
    #                     metavar='N', help='mini-batch size (default: 128)')
    # parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset', default='~/data/ImageNet', type=str)
    parser.add_argument('--dataset',
                        help='choose datasets ', default='ImageNet', type=str)
    parser.add_argument('--save_path',
                        help='path to log', default='./save', type=str)
    parser.add_argument('--random_seed', '-rd', default=random.randint(0, 2022), type=int,
                        help='seed for dataset split')
    parser.add_argument('--run_epoch', '-rp', default=1, type=int,
                        help='Outer Running epoch')
    parser.add_argument('--evolution_epoch', '-ep', default=10, type=int,
                        help='Evolution epoch')
    parser.add_argument('--ft_epoch', '-fp', default=60, type=int,
                        help='Fine-tune epoch')
    parser.add_argument('--filter_num', type=int, nargs='+',
                        help='Filter number of network, use in resume mode')
    parser.add_argument('--lr_milestone', type=int, nargs='+',
                        help='lr milestone for fine-tune')
    parser.add_argument('--resume', action='store_true',
                        help='Resume running')
    parser.add_argument('--keep', action='store_true',
                        help='Keep best or not')
    parser.add_argument('--finetune', action='store_true',
                        help='fintune after resume running')
    parser.add_argument('--mutation_rate', default=0.1, type=float,
                        help='Mutation rate in ccea')
    parser.add_argument('--pop_init_rate', default=0.8, type=float,
                        help='init pruning rate when init population')
    parser.add_argument('--prune_limitation', default=0.75, type=float,
                        help='prune limitation rate')
    parser.add_argument('--pop_size', default=5, type=int,
                        help='population size for CCEA')
    parser.add_argument('--valid_ratio', default=0.99, type=float,
                        help='population size for CCEA')

    parser.add_argument('--use_crossover', action='store_true',
                        help='use crossover in evolution')

    parser.add_argument('--crossover_rate', default=0.3, type=float,
                        help='crossover rate for CCEA')

    #
    #load args
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./dataset/minist',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')


    #load model
    #load data
    # 初始化数据加载器
    args = parser.parse_args()
    print(args)
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=int(args.b*0.8),
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    #form mask
    #start ea

    #==============

    #==============

    # set_logger(args)
    # model = get_model(args)
    # train_loader, valid_loader, test_loader = get_data(args)
    logger = logging.getLogger()

    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEP_snn.CCEP(model, train_loader, valid_loader, test_loader, args)
    # alg.run(args.run_epoch)
    alg.run(args.run_epoch)

    logger.info('done')

    #save mask
