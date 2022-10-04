import torch
from config import *
from pruning_utils import *
from thop import profile
import copy
# import finetune
import random
import numpy as np
import logging
from train import validate
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os

import logging
# from config import *
# from utils import *

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
from spikingjelly.spikingjelly.activation_based.model import spiking_resnet




# modified CCEP for snn pruniing, do have training
class CCEPSNN:
    def __init__(self, model, train_loader, valid_loader, test_loader, args):
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.FILTER_NUM = []
        self.args = args
        self.model = model
        self.ori_model = model
        self.best_model_list = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = []
        self.parms = []
        self.FLOPS = []
        self.FILTER_NUMS = []
        if self.args:
            self.pop_size = self.args.pop_size
            self.pop_init_rate = self.args.pop_init_rate
            self.prune_limitation = self.args.prune_limitation
            self.mutation_rate = self.args.mutation_rate
            self.evolution_epoch = self.args.evolution_epoch
            if self.args.arch == 'vgg' :
                self.pruning_func = prune_VGG_group
            elif self.args.arch in {'resnet56','resnet110'}:
                self.pruning_func = prune_Resnet_group
            elif self.args.arch == 'resnet50':
                self.pruning_func = prune_Resnet_imagenet_group
            elif self.args.arch == 'resnet34':
                self.pruning_func = prune_Resnet34_group
            elif self.args.arch == 'spike_vgg':
                self.pruning_func = prune_spike_VGG_group
            else:
                raise NotImplementedError('Not implemented model')

    def fitness(self, test_model=None):
        if test_model:
            return validate(self.valid_loader, test_model, self.criterion, self.args, print_result=False)
        return validate(self.valid_loader, self.model, self.criterion, self.args, print_result=False)

    def generate_initial_pop(self, filter_num):
        p = []
        indiv = [i for i in range(filter_num)]
        p.append(indiv)
        for i in range(self.pop_size-1):
            indiv = []
            for j in range(filter_num):
                if random.random() < self.pop_init_rate:
                    indiv.append(j)
            if len(indiv) >= 1:
                if len(indiv) >= filter_num*self.prune_limitation:
                    p.append(indiv)
                else:
                    while len(indiv) < filter_num*self.prune_limitation:
                        new_filter = random.randint(0, filter_num - 1)
                        if new_filter not in indiv:
                            indiv.append(new_filter)
                    p.append(indiv)
            else:
                for j in range(filter_num):
                    indiv.append(j)
                p.append(indiv)
        p.sort()
        return p
    
    def crossover(self, indiv1, indiv2, filter_num):
        cross_point = random.randint(0, filter_num)
        gene1 = np.zeros(filter_num)
        gene2 = np.zeros(filter_num)
        for x in indiv1:
            gene1[x] = 1
        for x in indiv2:
            gene2[x] = 1
        offspring_gene1 = np.hstack((gene1[:cross_point], gene2[cross_point:]))
        offspring_gene2 = np.hstack((gene2[:cross_point], gene1[cross_point:]))
        offspring1 = [x for x in range(len(offspring_gene1)) if offspring_gene1[x] == 1]
        offspring2 = [x for x in range(len(offspring_gene2)) if offspring_gene2[x] == 1]
        return offspring1,offspring2

    def mutation(self, indiv, filter_num):
        temp_np = np.zeros((int(filter_num)))
        temp_np[indiv] = 1
        for i in range(filter_num):
            if random.random() < self.mutation_rate:
                if temp_np[i] == 0:
                    temp_np[i] = 1 - temp_np[i]
                elif np.sum(temp_np) >= (filter_num*self.prune_limitation):
                    temp_np[i] = 1 - temp_np[i]
        new_indiv = []
        for i in range(filter_num):
            if temp_np[i] == 1:
                new_indiv.append(i)
        if len(new_indiv) == 0:
            return indiv
        else:
            new_indiv.sort()
            return new_indiv

    def evoluiton_step(self, filter_num, deleted_stage_index, deleted_block_index=-1, delete_conv_index=-1):
        # for a block return a solution , which is suppose to be a mask in my xcase with the correct shape
        pop = self.generate_initial_pop(filter_num)
        logger = logging.getLogger()
        logger.info("Stage:{0} | block:{1} -{2} | filter_num:{3}\n".format(deleted_stage_index, deleted_block_index, delete_conv_index, filter_num))
        parent_fitness = []
        initial_fitness = self.fitness()
        logger.info(f"Initial fitness:{initial_fitness}")
        logger.info(f"Initial population")
        for i in range(self.pop_size):
            test_model = copy.deepcopy(self.model)
            if delete_conv_index != -1:
                test_model = self.pruning_func(test_model, deleted_stage_index, deleted_block_index, delete_conv_index, pop[i])
            elif deleted_block_index != -1:
                test_model = self.pruning_func(test_model, deleted_stage_index, deleted_block_index, pop[i])
            else:
                test_model = self.pruning_func(test_model, deleted_stage_index, pop[i])
            fitness_i = self.fitness(test_model)
            parent_fitness.append([i, fitness_i, pop[i], len(pop[i])])
            logger.info([i, fitness_i, [_ for _ in range(filter_num) if _ not in pop[i]], len(pop[i])])

        parent_fitness.sort(key=lambda x: (x[1], -x[3]), reverse=True)

        for i in range(self.evolution_epoch):
            child_fitness = []
            logger.info(f'Population at round {i}')
            if self.args.use_crossover:
                for j in range(0, self.pop_size):
                    if random.random() < self.args.crossover_rate:
                        rand1 = random.randint(0, self.pop_size - 1)
                        rand2 = random.randint(0, self.pop_size - 1)
                        while rand1 == rand2:
                            rand2 = random.randint(0, self.pop_size - 1)

                        parent1 = pop[rand1]
                        parent2 = pop[rand2]

                        child1, child2 = self.crossover(parent1, parent2, filter_num)
                        pop[rand1] = child1
                        pop[rand2] = child2
            for j in range(self.pop_size):
                parent = pop[random.randint(0,self.pop_size - 1)]
                child_indiv = self.mutation(parent, filter_num)
                test_model = copy.deepcopy(self.model)
                if delete_conv_index != -1:
                    test_model = self.pruning_func(test_model, deleted_stage_index, deleted_block_index,
                                                   delete_conv_index, child_indiv)
                elif deleted_block_index != -1:
                    test_model = self.pruning_func(test_model, deleted_stage_index, deleted_block_index, child_indiv)
                else:
                    test_model = self.pruning_func(test_model, deleted_stage_index,child_indiv)
                fitness_j = self.fitness(test_model)
                child_fitness.append([j, fitness_j, child_indiv, len(child_indiv)])

                logger.info([j, fitness_j, [_ for _ in range(filter_num) if _ not in child_indiv], len(child_indiv)])
            logger.info('\n\n')
            temp_list = []
            for j in range(len(parent_fitness)):
                temp_list.append(parent_fitness[j])
            for j in range(len(child_fitness)):
                temp_list.append(child_fitness[j])
            temp_list.sort(key = lambda x:(x[1], -x[3]), reverse=True)
            logger.info(f'Population at epoch {i}:')
            for j in range(self.pop_size):
                pop[j] = temp_list[j][2]
                parent_fitness[j] = temp_list[j]

                logger.info([parent_fitness[j][0], parent_fitness[j][1], [_ for _ in range(filter_num) if _ not in parent_fitness[j][2]],len(parent_fitness[j][2]) ])
            logger.info(f'\n\n')
            best_ind = None
            if self.args.keep==True:
                best_ind = parent_fitness[0]
            else:
                if len(parent_fitness[0][2]) != filter_num:
                    best_ind = parent_fitness[0]
                else:
                    best_ind = parent_fitness[1]
            logger.info(
                f'Best so far {best_ind[1]}, Initial fitness: {initial_fitness}, Filter now:{best_ind[3]}, Pruning ratio: {1 - best_ind[3] / filter_num}')
        logger.info(f'Pruned filters {[_ for _ in range(filter_num) if _ not in best_ind[2]]}')
        return best_ind[2]

    def check_model_profile(self):
        logger = logging.getLogger()
        if self.args.dataset == 'cifar10':
            model_input = torch.randn(1, 3, 32, 32)
        else:
            model_input = torch.randn(1, 3, 224, 224)
        if self.args.arch != 'vgg':
            i_flops, i_params = profile(self.ori_model.module, inputs=(model_input.cuda(),), verbose=False)
            logger.info("initial model: FLOPs: {0}, params: {1}".format(i_flops, i_params))
            p_flops, p_params = profile(self.model.module.cuda(), inputs=(model_input.cuda(),), verbose=False)
            logger.info("pruned model: FLOPs: {0}({1:.2f}%), params: {2}({3:.2f}%)".format(p_flops, (1 - p_flops / i_flops) * 100,
                                                                                     p_params,
                                                                                     (1 - p_params / i_params) * 100))
        else:
            if self.args.dataset == 'cifar10':
                i_flops, i_params = profile(self.ori_model, inputs=(model_input.cuda(),), verbose=False)
                logger.info("initial model: FLOPs: {0}, params: {1}".format(i_flops, i_params))
                p_flops, p_params = profile(self.model.cuda(), inputs=(model_input.cuda(),), verbose=False)
                logger.info("pruned model: FLOPs: {0}({1:.2f}%), params: {2}({3:.2f}%)".format(p_flops, (
                        1 - p_flops / i_flops) * 100,
                                                                                               p_params,
                                                                                               (
                                                                                                       1 - p_params / i_params) * 100))
            else:
                i_flops, i_params = profile(self.ori_model.module, inputs=(model_input.cuda(),), verbose=False)
                logger.info("initial model: FLOPs: {0}, params: {1}".format(i_flops, i_params))
                p_flops, p_params = profile(self.model.module.cuda(), inputs=(model_input.cuda(),), verbose=False)
                logger.info("pruned model: FLOPs: {0}({1:.2f}%), params: {2}({3:.2f}%)".format(p_flops, (
                        1 - p_flops / i_flops) * 100,
                                                                                               p_params,
                                                                                               (
                                                                                                       1 - p_params / i_params) * 100))
        self.model.cuda()
        return (1 - p_flops / i_flops) * 100, (1 - p_params / i_params) * 100
    def generate_mask(self,shape):

        mask = torch.ones(shape)
        return mask
    def run(self, run_epoch):
        self.model.cuda()
        logger = logging.getLogger()
        # fine_tune_method = finetune.fine_tune()
        FILTER_NUM = []
        mask = self.generate_mask()
        print(mask)
        BLOCK_NUM = []
        sol = []
        layers = []
        #init solution
        if self.args.arch in {'resnet34', 'resnet50'}:
            layers = ['layer1', 'layer2', 'layer3', 'layer4']
            BLOCK_NUM = [3, 4, 6, 3]
            if self.args.arch == 'resnet50':
                for layer in range(len(BLOCK_NUM)):
                    for i in range(BLOCK_NUM[layer]):
                        FILTER_NUM.append(64 * (2 ** layer))
                        FILTER_NUM.append(64 * (2 ** layer))
                        sol.append([])
                        sol.append([])
            else:
                for layer in range(len(BLOCK_NUM)):
                    for i in range(BLOCK_NUM[layer]):
                        FILTER_NUM.append(64 * (2 ** layer))
                        sol.append([])
        elif self.args.arch in {'resnet56', 'resnet110'}:
            layers = ['layer1', 'layer2', 'layer3']
            if self.args.arch == 'resnet56':
                BLOCK_NUM = [9, 9, 9]
            else:
                BLOCK_NUM = [18, 18, 18]
            for layer in range(len(BLOCK_NUM)):
                for i in range(BLOCK_NUM[layer]):
                    FILTER_NUM.append(16 * (2 ** layer))
                    sol.append([])
        elif self.args.arch in {'vgg'}: #vgg 16 ,13 cove layers, last 3 layers are not conv but all connected
            BLOCK_NUM = [1] * 13
            layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            FILTER_NUM = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            sol = []
            for i in range(13):
                sol.append([])
        elif self.args.arch in {'spiking_vgg'}: #vgg 16 ,13 cove layers, last 3 layers are not conv but all connected
            BLOCK_NUM = [1] * 13#TODO
            layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            FILTER_NUM = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            sol = []
            for i in range(13):
                sol.append([])
        #resume
        if self.args.resume:
            FILTER_NUM = self.args.filter_num
            pruned_model = torch.load(self.args.dict_path)
            logger.info(f'FILTER_NUM: {FILTER_NUM}')
            logger.info(f'Model now: {pruned_model}')
            self.model = copy.deepcopy(pruned_model)
            self.check_model_profile()
            # print(pruned_model)
            if self.args.finetune:
                optimizer = torch.optim.SGD(pruned_model.parameters(), 0.05, momentum=self.args.momentum,
                                            weight_decay=0.00004)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, last_epoch=-1)
                # self.model = fine_tune_method.basic_finetune(pruned_model, self.args.ft_epoch, self.train_loader,
                #                                              self.test_loader, self.criterion, optimizer, self.args,
                #                                              lr_scheduler, log_save=False)
                flops, params = self.check_model_profile()
                self.best_model_list.append(self.model)
                logger.info("epoch:{0} after fine-tune...".format(i))
                logger.info("Test set:")
                acc = validate(self.test_loader, self.model, self.criterion, self.args)
                self.acc.append(acc)
                self.FLOPS.append(flops)
                self.parms.append(params)
                logger.info("Valid set:")
                validate(self.valid_loader, self.model, self.criterion, self.args)
                save_path = f'{self.args.save_path}/{self.args.arch}_{self.args.dataset}_af.pth'
                torch.save(self.model, save_path)
            self.model = copy.deepcopy(pruned_model)
        self.check_model_profile()
        logger.info("Test set:")
        validate(self.test_loader, self.model, self.criterion, self.args)
        for i in range(run_epoch):
            cur_model = copy.deepcopy(self.model)
            logger.info(f'Outer Epoch: {i}')
            index = 0
            for layer in range(len(BLOCK_NUM)): # 13 blocks
                if self.args.arch == 'vgg': # vgg   13 blocks           0 1 2 3 4 5 6 7 8 9 10 11 12
                    sol[index] = self.evoluiton_step(FILTER_NUM[index], layers[layer])
                    index += 1
                    print('vgg')
                elif self.args.arch == 'spiking_vgg': # vgg   13 blocks           0 1 2 3 4 5 6 7 8 9 10 11 12
                    sol[index] = self.evoluiton_step(FILTER_NUM[index], layers[layer])
                    index += 1
                    print('vgg')
                # else:
                #     for block_index in range(BLOCK_NUM[layer]):
                #         if self.args.arch != 'resnet50':
                #             sol[index] = self.evoluiton_step(FILTER_NUM[index],layers[layer], block_index)
                #             index += 1
                #         else:
                #             sol[index] = self.evoluiton_step(FILTER_NUM[index], layers[layer], block_index, 0)
                #             index += 1
                #             sol[index] = self.evoluiton_step(FILTER_NUM[index], layers[layer], block_index, 1)
                #             index += 1

            index = 0
            for layer in range(len(BLOCK_NUM)):
                if self.args.arch == 'vgg':
                    self.pruning_func(cur_model, layers[layer], sol[index])
                    FILTER_NUM[index] = len(sol[index])
                    index += 1
                # else:
                #     for block_index in range(BLOCK_NUM[layer]):
                #         if self.args.arch != 'resnet50':
                #             self.pruning_func(cur_model, layers[layer], block_index, sol[index])
                #             FILTER_NUM[index] = len(sol[index])
                #             index += 1
                #         else:
                #             self.pruning_func(cur_model, layers[layer], block_index, 0, sol[index])
                #             FILTER_NUM[index] = len(sol[index])
                #             index += 1
                #             self.pruning_func(cur_model, layers[layer], block_index, 1, sol[index])
                #             FILTER_NUM[index] = len(sol[index])
                #             index += 1

            logger.info(f'FILTER_NUM {FILTER_NUM}')
            logger.info("epoch:{0} before fine-tune...".format(i))
            self.check_model_profile()
            self.FILTER_NUMS.append(FILTER_NUM[:])
            logger.info("Test set:")
            validate(self.test_loader, cur_model, self.criterion, self.args)
            save_path = f'{self.args.save_path}/{self.args.arch}_{self.args.dataset}_bf{i}.pth'
            torch.save(cur_model.state_dict(), save_path)
            if self.args.dataset == 'cifar10':
                optimizer = torch.optim.SGD(cur_model.parameters(), 0.1, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
            # else:
            #     optimizer = torch.optim.SGD(cur_model.parameters(), 0.01, momentum=self.args.momentum,
            #                                 weight_decay=self.args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestone,
                                                                    last_epoch=-1)
            # self.model = fine_tune_method.basic_finetune(cur_model, self.args.ft_epoch, self.train_loader, self.test_loader, self.criterion, optimizer, self.args,
            #                                 lr_scheduler)
            flops, params = self.check_model_profile()
            self.best_model_list.append(self.model)
            logger.info("epoch:{0} after fine-tune...".format(i))
            logger.info("Test set:")
            acc = validate(self.test_loader, self.model, self.criterion, self.args)
            self.acc.append(acc)
            self.FLOPS.append(flops)
            self.parms.append(params)
            logger.info("Valid set:")
            validate(self.valid_loader, self.model, self.criterion, self.args)
            save_path = f'{self.args.save_path}/{self.args.arch}_{self.args.dataset}_af{i}.pth'
            torch.save(self.model.state_dict(), save_path)
            logger.info(f'ACC:{self.acc}')
            logger.info(f'FLOPS:{self.FLOPS}')
            logger.info(f'Params:{self.parms}')
            for i in range(len(self.FILTER_NUMS)):
                logger.info(f'FILTER_NUM at epoch {i + 1}:{self.FILTER_NUMS[i]}')
        logger.info(f'ACC:{self.acc}')
        logger.info(f'FLOPS:{self.FLOPS}')
        logger.info(f'Params:{self.parms}')
        for i in range(len(self.FILTER_NUMS)):
            logger.info(f'FILTER_NUM at epoch {i+1}:{self.FILTER_NUMS[i]}')
        return self.best_model_list
