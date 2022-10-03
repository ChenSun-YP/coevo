import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import CCEP_snn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import arch.resnet_cifar10 as cifar_models
from prefetch_generator1.prefetch_generator import BackgroundGenerator
# from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import arch.resnet_imagenet as imagenet_models
import arch.vgg_cifar10 as vggmodels
import pruning_utils
import torchvision.models.vgg as vgg_imagenet_models
import torchvision
import logging
import train
import os
import finetune
import CCEP
from thop import profile
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from CCEP_snn import CCEP_snn

from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_vgg
from spikingjelly.activation_based.layer import BatchNorm2d

from spikingjelly.activation_based import surrogate, neuron, functional
cifar_model_names = sorted(name for name in cifar_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and name.startswith("resnet")
                           and callable(cifar_models.__dict__[name]))

imagenet_model_names = sorted(name for name in imagenet_models.__dict__
                              if name.islower() and not name.startswith("__")
                              and name.startswith("resnet")
                              and callable(imagenet_models.__dict__[name]))
vgg_models_name = sorted(name for name in vggmodels.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(vggmodels.__dict__[name]))

model_names = cifar_model_names + imagenet_model_names + vgg_models_name

parser = argparse.ArgumentParser(description='Pruning Neural Network by CCEA')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--dict_path', dest='dict_path', default='./save_model/', type=str)
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--run_epoch', '-rp', default=10, type=int,
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

parser.add_argument('--use_crossover',action='store_true',
                    help='use crossover in evolution')

parser.add_argument('--crossover_rate', default=0.3, type=float,
                    help='crossover rate for CCEA')
