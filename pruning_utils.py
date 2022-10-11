import torch.nn.utils.prune as prune

import arch.resnet_cifar10 as resnet
import arch.resnet_imagenet as resnet_imagenet
import torch
import spikingjelly
import copy
from torchvision import models
from spikingjelly.activation_based.layer import Conv2d as Conv2dsnn
from spikingjelly.activation_based.layer import BatchNorm2d as BatchNorm2dsnn
from spikingjelly.activation_based.layer import Linear as Linearsnn
from spikingjelly.activation_based import surrogate, neuron, functional



def replace_block(stage_list, i, index, new_block):
    if i == index:
        return new_block
    return stage_list[i][1]


def random_prune(named_modules, layer_type, layer_key, prune_rate=0.1, dim=0):
    for n, module in named_modules:
        if isinstance(module, layer_type):
            if layer_key in n:
                prune.random_structured(module, name='weight', amount=prune_rate, dim=dim)
                prune.remove(module, 'weight')


def prune_Resnet(model, deleted_stage_index, deleted_block_index, deleted_filter_index, use_cuda=True):
    for n, module in model.module._modules.items():
        if n == deleted_stage_index:
            stage_list = list(module._modules.items())

    old_basic_block = stage_list[deleted_block_index][1]
    old_conv1_weight = old_basic_block._modules['conv1'].weight.data.cpu().numpy()
    old_bn1_weight = [old_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    old_conv2_weight = old_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_basic_block = resnet.New_BasicBlock(old_basic_block.in_planes, old_basic_block.neck_planes - 1,
                                            old_basic_block.planes, old_basic_block.stride, old_basic_block.option)

    new_conv1_weight = new_basic_block._modules['conv1'].weight.data.cpu().numpy()
    new_bn1_weight = [new_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_conv1_weight[:deleted_filter_index, :, :, :] = old_conv1_weight[:deleted_filter_index, :, :, :]
    new_conv1_weight[deleted_filter_index:, :, :, :] = old_conv1_weight[deleted_filter_index + 1:, :, :, :]
    for i in range(len(new_bn1_weight)):
        new_bn1_weight[i][:deleted_filter_index] = old_bn1_weight[i][:deleted_filter_index]
        new_bn1_weight[i][deleted_filter_index:] = old_bn1_weight[i][deleted_filter_index + 1:]
    new_conv2_weight[:, :deleted_filter_index, :, :] = old_conv2_weight[:, :deleted_filter_index, :, :]
    new_conv2_weight[:, deleted_filter_index:, :, :] = old_conv2_weight[:, deleted_filter_index + 1:, :, :]

    new_basic_block._modules['conv1'].weight.data = torch.from_numpy(new_conv1_weight)
    new_basic_block._modules['bn1'].weight.data = torch.from_numpy(new_bn1_weight[0])
    new_basic_block._modules['bn1'].bias.data = torch.from_numpy(new_bn1_weight[1])
    new_basic_block._modules['bn1'].running_mean.data = torch.from_numpy(new_bn1_weight[2])
    new_basic_block._modules['bn1'].running_var.data = torch.from_numpy(new_bn1_weight[3])
    new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)

    new_basic_block._modules['bn2'] = old_basic_block._modules['bn2']
    new_basic_block._modules['shortcut'] = old_basic_block._modules['shortcut']

    if use_cuda:
        new_basic_block.cuda()

    new_stage = torch.nn.Sequential(
        *(replace_block(stage_list, i, deleted_block_index, new_basic_block) for i in range(len(stage_list))))

    model.module.add_module(deleted_stage_index, new_stage)
    return model


def prune_Resnet34_group(model, deleted_stage_index, deleted_block_index, reserved_filter_group, use_cuda=True):
    for n, module in model.module._modules.items():
        if n == deleted_stage_index:
            stage_list = list(module._modules.items())

    old_basic_block = stage_list[deleted_block_index][1]
    old_conv1_weight = old_basic_block._modules['conv1'].weight.data.cpu().numpy()
    old_bn1_weight = [old_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    old_conv2_weight = old_basic_block._modules['conv2'].weight.data.cpu().numpy()

    if deleted_block_index == 0 and deleted_stage_index != 'layer1':
        new_basic_block = resnet_imagenet.New_BasicBlock(old_basic_block.in_planes, len(reserved_filter_group),
                                            old_basic_block.planes, old_basic_block.stride,old_basic_block._modules['downsample'])
    else:
        new_basic_block = resnet_imagenet.New_BasicBlock(old_basic_block.in_planes, len(reserved_filter_group),
                                                         old_basic_block.planes, old_basic_block.stride)

    new_conv1_weight = new_basic_block._modules['conv1'].weight.data.cpu().numpy()
    new_bn1_weight = [new_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_conv1_weight[:, :, :, :] = old_conv1_weight[reserved_filter_group, :, :, :]
    for i in range(len(new_bn1_weight)):
        new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
    new_conv2_weight[:, :, :, :] = old_conv2_weight[:, reserved_filter_group, :, :]

    new_basic_block._modules['conv1'].weight.data = torch.from_numpy(new_conv1_weight)
    new_basic_block._modules['bn1'].weight.data = torch.from_numpy(new_bn1_weight[0])
    new_basic_block._modules['bn1'].bias.data = torch.from_numpy(new_bn1_weight[1])
    new_basic_block._modules['bn1'].running_mean.data = torch.from_numpy(new_bn1_weight[2])
    new_basic_block._modules['bn1'].running_var.data = torch.from_numpy(new_bn1_weight[3])
    new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)

    new_basic_block._modules['bn2'] = old_basic_block._modules['bn2']

    if use_cuda:
        new_basic_block.cuda()

    new_stage = torch.nn.Sequential(
        *(replace_block(stage_list, i, deleted_block_index, new_basic_block) for i in range(len(stage_list))))

    model.module.add_module(deleted_stage_index, new_stage)
    return model


def prune_Resnet_group(model, deleted_stage_index, deleted_block_index, reserved_filter_group, use_cuda=True):
    for n, module in model.module._modules.items():
        if n == deleted_stage_index:
            stage_list = list(module._modules.items())

    old_basic_block = stage_list[deleted_block_index][1]
    old_conv1_weight = old_basic_block._modules['conv1'].weight.data.cpu().numpy()
    old_bn1_weight = [old_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    old_conv2_weight = old_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_basic_block = resnet.New_BasicBlock(old_basic_block.in_planes, len(reserved_filter_group),
                                            old_basic_block.planes, old_basic_block.stride, old_basic_block.option)

    new_conv1_weight = new_basic_block._modules['conv1'].weight.data.cpu().numpy()
    new_bn1_weight = [new_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_conv1_weight[:, :, :, :] = old_conv1_weight[reserved_filter_group, :, :, :]
    for i in range(len(new_bn1_weight)):
        new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
    new_conv2_weight[:, :, :, :] = old_conv2_weight[:, reserved_filter_group, :, :]

    new_basic_block._modules['conv1'].weight.data = torch.from_numpy(new_conv1_weight)
    new_basic_block._modules['bn1'].weight.data = torch.from_numpy(new_bn1_weight[0])
    new_basic_block._modules['bn1'].bias.data = torch.from_numpy(new_bn1_weight[1])
    new_basic_block._modules['bn1'].running_mean.data = torch.from_numpy(new_bn1_weight[2])
    new_basic_block._modules['bn1'].running_var.data = torch.from_numpy(new_bn1_weight[3])
    new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)

    new_basic_block._modules['bn2'] = old_basic_block._modules['bn2']
    new_basic_block._modules['shortcut'] = old_basic_block._modules['shortcut']

    if use_cuda:
        new_basic_block.cuda()

    new_stage = torch.nn.Sequential(
        *(replace_block(stage_list, i, deleted_block_index, new_basic_block) for i in range(len(stage_list))))

    model.module.add_module(deleted_stage_index, new_stage)
    return model


def copy_resnet(model, pruned_model,stage_index, block_index, filter_group, use_cuda=True):
    for n, module in pruned_model.module._modules.items():
        if n == stage_index:
            stage_list = list(module._modules.items())

    old_basic_block = stage_list[block_index][1]
    old_conv1_weight = old_basic_block._modules['conv1'].weight.data.cpu().numpy()
    old_bn1_weight = [old_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    old_conv2_weight = old_basic_block._modules['conv2'].weight.data.cpu().numpy()


    for n, module in model.module._modules.items():
        if n == stage_index:
            stage_list = list(module._modules.items())
    new_basic_block = stage_list[block_index][1]

    new_conv1_weight = new_basic_block._modules['conv1'].weight.data.cpu().numpy()
    new_bn1_weight = [new_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      new_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()

    new_conv1_weight[filter_group, :, :, :] = old_conv1_weight[:, :, :, :]
    for i in range(len(new_bn1_weight)):
        new_bn1_weight[i][filter_group] = old_bn1_weight[i]
    new_conv2_weight[:, filter_group, :, :] = old_conv2_weight[:, :, :, :]

    new_basic_block._modules['conv1'].weight.data = torch.from_numpy(new_conv1_weight)
    new_basic_block._modules['bn1'].weight.data = torch.from_numpy(new_bn1_weight[0])
    new_basic_block._modules['bn1'].bias.data = torch.from_numpy(new_bn1_weight[1])
    new_basic_block._modules['bn1'].running_mean.data = torch.from_numpy(new_bn1_weight[2])
    new_basic_block._modules['bn1'].running_var.data = torch.from_numpy(new_bn1_weight[3])
    new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)

    new_basic_block._modules['bn2'] = old_basic_block._modules['bn2']
    new_basic_block._modules['shortcut'] = old_basic_block._modules['shortcut']

    if use_cuda:
        new_basic_block.cuda()

    new_stage = torch.nn.Sequential(
        *(replace_block(stage_list, i, block_index, new_basic_block) for i in range(len(stage_list))))

    model.module.add_module(stage_list, new_stage)
    return model


def prune_VGG_group(model, deleted_layer_index, reserved_filter_group, bias=False, use_cuda=True):
    # print(model)
    def replace_layers(model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    conv_16_index = {0: 0, 1: 3, 2: 7, 3: 10, 4: 14, 5: 17, 6: 20, 7: 24, 8: 27, 9: 30, 10: 34, 11: 37, 12: 40}
    if deleted_layer_index != 12:
        # current conv
        _, old_conv1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index]]
        new_conv1 = torch.nn.Conv2d(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None))
        old_conv1_weights = old_conv1.weight.data.cpu().numpy()
        new_conv1_weights = new_conv1.weight.data.cpu().numpy()
        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        # conv bias
        if bias == True:
            old_conv1_bias = old_conv1.bias.data.cpu().numpy()
            new_conv1_bias = new_conv1.bias.data.cpu().numpy()
            new_conv1_bias = old_conv1_weights[reserved_filter_group]
            new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
        # following bn
        _, old_bn1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = torch.nn.BatchNorm2d(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        # next_conv
        _, old_conv2 = list(model.features._modules.items())[conv_16_index[deleted_layer_index + 1]]
        new_conv2 = torch.nn.Conv2d(in_channels=len(reserved_filter_group), out_channels=old_conv2.out_channels,
                                    kernel_size=old_conv2.kernel_size, stride=old_conv2.stride,
                                    padding=old_conv2.padding, dilation=old_conv2.dilation, groups=old_conv2.groups,
                                    bias=(old_conv2.bias is not None))
        old_conv2_weights = old_conv2.weight.data.cpu().numpy()
        new_conv2_weights = new_conv2.weight.data.cpu().numpy()
        new_conv2_weights[:, :, :, :] = old_conv2_weights[:, reserved_filter_group, :, :]
        new_conv2.weight.data = torch.from_numpy(new_conv2_weights)
        if bias == True:
            new_conv2.bias.data = old_conv2.bias.data

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()
            new_conv2.cuda()

        features = torch.nn.Sequential(
            *(replace_layers(model.features, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1,
                              conv_16_index[deleted_layer_index + 1]], \
                             [new_conv1, new_bn1, new_conv2]) for i, _ in enumerate(model.features)))

        del model.features
        model.features = features
        return model

    else:
        # last conv layer
        _, old_conv1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index]]
        new_conv1 = torch.nn.Conv2d(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None))
        old_conv1_weights = old_conv1.weight.data.cpu().numpy()
        new_conv1_weights = new_conv1.weight.data.cpu().numpy()
        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        if bias == True:
            new_conv1_bias = old_conv1_weights[reserved_filter_group]
            new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
            old_conv1_bias = old_conv1.bias.data.cpu().numpy()
            new_conv1_bias = new_conv1.bias.data.cpu().numpy()

        # following bn
        _, old_bn1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = torch.nn.BatchNorm2d(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()

        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1], \
                             [new_conv1, new_bn1]) for i, _ in enumerate(model.features)))

        # first linear layer
        old_conv_filter_num = old_conv1.out_channels
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module,Linearsnn):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")

        params_per_input_channel = old_linear_layer.in_features // old_conv1.out_channels
        # print(params_per_input_channel)
        # print(old_conv_filter_num, len(reserved_filter_group))
        new_linear_layer = \
            Linearsnn(old_linear_layer.in_features - params_per_input_channel * (
                        old_conv_filter_num - len(reserved_filter_group)),
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        # print(new_conv_filter_num)
        # print(params_per_input_channel)

        reserved_filter_group.sort(key=lambda a: a)
        # print(reserved_filter_group)

        for i in range(len(reserved_filter_group)):
            # print(i, reserved_filter_group[i])
            new_weights[:, i * params_per_input_channel:(i + 1) * params_per_input_channel] = \
                old_weights[:, reserved_filter_group[i] * params_per_input_channel:(reserved_filter_group[
                                                                                        i] + 1) * params_per_input_channel]
            # print(i)

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        model.classifier = classifier
        return model

def prune_spike_VGG_group(model, deleted_layer_index, reserved_filter_group, bias=False, use_cuda=True):
    def replace_layers(model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    conv_16_index = {0: 0, 1: 3, 2: 7, 3: 10, 4: 14, 5:17 , 6: 20, 7:24 , 8: 27, 9: 30, 10: 34, 11: 37, 12:40 }
    if deleted_layer_index != 12:
        # current conv
        _, old_conv1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index]]


        # new_conv1 = torch.nn.Conv2d(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
        #                             kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
        #                             padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
        #                             bias=(old_conv1.bias is not None))

        new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None), step_mode="s")

        old_conv1_weights = old_conv1.weight.data.cpu().numpy()

        new_conv1_weights = new_conv1.weight.data.cpu().numpy()

        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]

        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        # conv bias
        if bias == True:
            old_conv1_bias = old_conv1.bias.data.cpu().numpy()
            new_conv1_bias = new_conv1.bias.data.cpu().numpy()
            new_conv1_bias = old_conv1_weights[reserved_filter_group]
            new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
        # following bn
        _, old_bn1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        # next_conv
        _, old_conv2 = list(model.features._modules.items())[conv_16_index[deleted_layer_index + 1]]
        new_conv2 = Conv2dsnn(in_channels=len(reserved_filter_group), out_channels=old_conv2.out_channels,
                                    kernel_size=old_conv2.kernel_size, stride=old_conv2.stride,
                                    padding=old_conv2.padding, dilation=old_conv2.dilation, groups=old_conv2.groups,
                                    bias=(old_conv2.bias is not None))
        old_conv2_weights = old_conv2.weight.data.cpu().numpy()
        new_conv2_weights = new_conv2.weight.data.cpu().numpy()
        new_conv2_weights[:, :, :, :] = old_conv2_weights[:, reserved_filter_group, :, :]
        new_conv2.weight.data = torch.from_numpy(new_conv2_weights)
        if bias == True:
            new_conv2.bias.data = old_conv2.bias.data

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()
            new_conv2.cuda()

        features = torch.nn.Sequential(
            *(replace_layers(model.features, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1,
                              conv_16_index[deleted_layer_index + 1]], \
                             [new_conv1, new_bn1, new_conv2]) for i, _ in enumerate(model.features)))

        del model.features
        model.features = features
        return model

    else:
        # last conv layer
        _, old_conv1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index]]
        new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None))
        old_conv1_weights = old_conv1.weight.data.cpu().numpy()
        new_conv1_weights = new_conv1.weight.data.cpu().numpy()
        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        if bias == True:
            new_conv1_bias = old_conv1_weights[reserved_filter_group]
            new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
            old_conv1_bias = old_conv1.bias.data.cpu().numpy()
            new_conv1_bias = new_conv1.bias.data.cpu().numpy()

        # following bn
        _, old_bn1 = list(model.features._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()

        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1], \
                             [new_conv1, new_bn1]) for i, _ in enumerate(model.features)))

        # first linear layer
        old_conv_filter_num = old_conv1.out_channels
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, Linearsnn):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

        params_per_input_channel = old_linear_layer.in_features // old_conv1.out_channels
        # print(params_per_input_channel)
        # print(old_conv_filter_num, len(reserved_filter_group))
        new_linear_layer = \
            Linearsnn(old_linear_layer.in_features - params_per_input_channel * (
                        old_conv_filter_num - len(reserved_filter_group)),
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        # print(new_conv_filter_num)
        # print(params_per_input_channel)

        reserved_filter_group.sort(key=lambda a: a)
        # print(reserved_filter_group)

        for i in range(len(reserved_filter_group)):
            # print(i, reserved_filter_group[i])
            new_weights[:, i * params_per_input_channel:(i + 1) * params_per_input_channel] = \
                old_weights[:, reserved_filter_group[i] * params_per_input_channel:(reserved_filter_group[
                                                                                        i] + 1) * params_per_input_channel]
            # print(i)

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        model.classifier = classifier
        return model

def prune_Resnet_imagenet_group(model, deleted_stage_index, deleted_block_index, delete_conv_index,
                                reserved_filter_group, use_cuda=True):
    # print(len(reserved_filter_group))
    for n, module in model.module._modules.items():
        if n == deleted_stage_index:
            stage_list = list(module._modules.items())

    old_basic_block = stage_list[deleted_block_index][1]
    old_conv1_weight = old_basic_block._modules['conv1'].weight.data.cpu().numpy()
    old_bn1_weight = [old_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
    old_conv2_weight = old_basic_block._modules['conv2'].weight.data.cpu().numpy()
    old_bn2_weight = [old_basic_block._modules['bn2'].weight.data.cpu().numpy(), \
                      old_basic_block._modules['bn2'].bias.data.cpu().numpy(), \
                      old_basic_block._modules['bn2'].running_mean.data.cpu().numpy(), \
                      old_basic_block._modules['bn2'].running_var.data.cpu().numpy()]
    old_conv3_weight = old_basic_block._modules['conv3'].weight.data.cpu().numpy()

    if delete_conv_index == 0:

        inplanes = old_conv1_weight.shape[1]
        neckplane2 = old_conv2_weight.shape[0]
        outplane = old_conv3_weight.shape[0]
        new_basic_block = resnet_imagenet.New_Bottleneck(inplanes, len(reserved_filter_group), neckplane2,
                                                         outplane, old_basic_block.stride, old_basic_block.downsample)

        new_conv1_weight = new_basic_block._modules['conv1'].weight.data.cpu().numpy()
        new_bn1_weight = [new_basic_block._modules['bn1'].weight.data.cpu().numpy(), \
                          new_basic_block._modules['bn1'].bias.data.cpu().numpy(), \
                          new_basic_block._modules['bn1'].running_mean.data.cpu().numpy(), \
                          new_basic_block._modules['bn1'].running_var.data.cpu().numpy()]
        new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()

        new_conv1_weight[:, :, :, :] = old_conv1_weight[reserved_filter_group, :, :, :]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_conv2_weight[:, :, :, :] = old_conv2_weight[:, reserved_filter_group, :, :]

        new_basic_block._modules['conv1'].weight.data = torch.from_numpy(new_conv1_weight)
        new_basic_block._modules['bn1'].weight.data = torch.from_numpy(new_bn1_weight[0])
        new_basic_block._modules['bn1'].bias.data = torch.from_numpy(new_bn1_weight[1])
        new_basic_block._modules['bn1'].running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_basic_block._modules['bn1'].running_var.data = torch.from_numpy(new_bn1_weight[3])
        new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)

        new_basic_block._modules['bn2'] = old_basic_block._modules['bn2']
        new_basic_block._modules['conv3'] = old_basic_block._modules['conv3']
        new_basic_block._modules['bn3'] = old_basic_block._modules['bn3']
        # print(new_basic_block._modules['bn2'].bias)
        # print(old_basic_block._modules['bn2'].bias)
        if deleted_block_index == 0:
            new_basic_block._modules['downsample'] = old_basic_block._modules['downsample']

        # print(new_basic_block)
        # print(old_basic_block)
    else:
        inplanes = old_conv1_weight.shape[1]
        neckplane1 = old_conv1_weight.shape[0]
        outplane = old_conv3_weight.shape[0]
        new_basic_block = resnet_imagenet.New_Bottleneck(inplanes, neckplane1, len(reserved_filter_group),
                                                         outplane, old_basic_block.stride,
                                                         old_basic_block.downsample)

        new_conv2_weight = new_basic_block._modules['conv2'].weight.data.cpu().numpy()
        new_bn2_weight = [new_basic_block._modules['bn2'].weight.data.cpu().numpy(), \
                          new_basic_block._modules['bn2'].bias.data.cpu().numpy(), \
                          new_basic_block._modules['bn2'].running_mean.data.cpu().numpy(), \
                          new_basic_block._modules['bn2'].running_var.data.cpu().numpy()]
        new_conv3_weight = new_basic_block._modules['conv3'].weight.data.cpu().numpy()

        new_conv2_weight[:, :, :, :] = old_conv2_weight[reserved_filter_group, :, :, :]
        for i in range(len(new_bn2_weight)):
            new_bn2_weight[i] = old_bn2_weight[i][reserved_filter_group]
        new_conv3_weight[:, :, :, :] = old_conv3_weight[:, reserved_filter_group, :, :]

        new_basic_block._modules['conv2'].weight.data = torch.from_numpy(new_conv2_weight)
        new_basic_block._modules['bn2'].weight.data = torch.from_numpy(new_bn2_weight[0])
        new_basic_block._modules['bn2'].bias.data = torch.from_numpy(new_bn2_weight[1])
        new_basic_block._modules['bn2'].running_mean.data = torch.from_numpy(new_bn2_weight[2])
        new_basic_block._modules['bn2'].running_var.data = torch.from_numpy(new_bn2_weight[3])
        new_basic_block._modules['conv3'].weight.data = torch.from_numpy(new_conv3_weight)

        new_basic_block._modules['bn1'] = old_basic_block._modules['bn1']
        new_basic_block._modules['conv1'] = old_basic_block._modules['conv1']
        new_basic_block._modules['bn3'] = old_basic_block._modules['bn3']
        if deleted_block_index == 0:
            new_basic_block._modules['downsample'] = old_basic_block._modules['downsample']
    if use_cuda:
        new_basic_block.cuda()

    new_stage = torch.nn.Sequential(
        *(replace_block(stage_list, i, deleted_block_index, new_basic_block) for i in range(len(stage_list))))

    model.module.add_module(deleted_stage_index, new_stage)
    return model


# def prune_csnn(model, deleted_layer_index, reserved_filter_group, bias=False, use_cuda=True):
#     def replace_layers(model, i, indexes, layers):
#         if i in indexes:
#             return layers[indexes.index(i)]
#         return model[i]
#
#     conv_16_index = {0: 0, 1: 4}
#     if deleted_layer_index != 1:
#         # current conv
#         _, old_conv1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index]]
#
#
#         # new_conv1 = torch.nn.Conv2d(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
#         #                             kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
#         #                             padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
#         #                             bias=(old_conv1.bias is not None))
#
#         new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
#                                     kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
#                                     padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
#                                     bias=(old_conv1.bias is not None), step_mode="m")
#
#         old_conv1_weights = old_conv1.weight.data.cpu().numpy()
#
#         new_conv1_weights = new_conv1.weight.data.cpu().numpy()
#
#         new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
#
#         new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
#         # # conv bias
#         # if bias == True:
#         #     old_conv1_bias = old_conv1.bias.data.cpu().numpy()
#         #     new_conv1_bias = new_conv1.bias.data.cpu().numpy()
#         #     new_conv1_bias = old_conv1_weights[reserved_filter_group]
#         #     new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
#         # following bn
#         _, old_bn1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index] + 1]
#         old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
#                           old_bn1.bias.data.cpu().numpy(), \
#                           old_bn1.running_mean.data.cpu().numpy(), \
#                           old_bn1.running_var.data.cpu().numpy()]
#         new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
#         new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
#                           new_bn1.bias.data.cpu().numpy(), \
#                           new_bn1.running_mean.data.cpu().numpy(), \
#                           new_bn1.running_var.data.cpu().numpy()]
#         for i in range(len(new_bn1_weight)):
#             new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
#         new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
#         new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
#         new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
#         new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])
#
#         # next_conv
#         _, old_conv2 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index + 1]]
#         new_conv2 = Conv2dsnn(in_channels=len(reserved_filter_group), out_channels=old_conv2.out_channels,
#                                     kernel_size=old_conv2.kernel_size, stride=old_conv2.stride,
#                                     padding=old_conv2.padding, dilation=old_conv2.dilation, groups=old_conv2.groups,
#                                     bias=(old_conv2.bias is not None), step_mode="m")
#         old_conv2_weights = old_conv2.weight.data.cpu().numpy()
#         new_conv2_weights = new_conv2.weight.data.cpu().numpy()
#         new_conv2_weights[:, :, :, :] = old_conv2_weights[:, reserved_filter_group, :, :]
#         new_conv2.weight.data = torch.from_numpy(new_conv2_weights)
#         if bias == True:
#             new_conv2.bias.data = old_conv2.bias.data
#
#         if use_cuda:
#             new_conv1.cuda()
#             new_bn1.cuda()
#             new_conv2.cuda()
#
#         features = torch.nn.Sequential(
#             *(replace_layers(model.conv_fc, i,
#                              [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1,
#                               conv_16_index[deleted_layer_index + 1]], \
#                              [new_conv1, new_bn1, new_conv2]) for i, _ in enumerate(model.conv_fc)))
#
#         del model.conv_fc
#         model.conv_fc = features
#         functional.set_step_mode(model,step_mode='m')
#
#         return model
#
#     else:
#         # last conv layer
#         _, old_conv1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index]]
#         new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
#                                     kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
#                                     padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
#                                     bias=(old_conv1.bias is not None))
#         old_conv1_weights = old_conv1.weight.data.cpu().numpy()
#         new_conv1_weights = new_conv1.weight.data.cpu().numpy()
#         new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
#         new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
#         if bias == True:
#             new_conv1_bias = old_conv1_weights[reserved_filter_group]
#             new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
#             old_conv1_bias = old_conv1.bias.data.cpu().numpy()
#             new_conv1_bias = new_conv1.bias.data.cpu().numpy()
#
#         # following bn
#         _, old_bn1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index] + 1]
#         old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
#                           old_bn1.bias.data.cpu().numpy(), \
#                           old_bn1.running_mean.data.cpu().numpy(), \
#                           old_bn1.running_var.data.cpu().numpy()]
#         new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
#         new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
#                           new_bn1.bias.data.cpu().numpy(), \
#                           new_bn1.running_mean.data.cpu().numpy(), \
#                           new_bn1.running_var.data.cpu().numpy()]
#         for i in range(len(new_bn1_weight)):
#             new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
#         new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
#         new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
#         new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
#         new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])
#
#         if use_cuda:
#             new_conv1.cuda()
#             new_bn1.cuda()
#
#         model.conv_fc = torch.nn.Sequential(
#             *(replace_layers(model.conv_fc, i,
#                              [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1], \
#                              [new_conv1, new_bn1]) for i, _ in enumerate(model.conv_fc)))
#
#         # first linear layer
#         old_conv_filter_num = old_conv1.out_channels
#         layer_index = 0
#         old_linear_layer = None
#         for _, module in model.conv_fc._modules.items():
#             if isinstance(module, Linearsnn):
#                 old_linear_layer = module
#                 break
#             layer_index = layer_index + 1
#
#         if old_linear_layer is None:
#             raise BaseException("No linear layer found in classifier")
#
#         params_per_input_channel = old_linear_layer.in_features // old_conv1.out_channels
#         # print(params_per_input_channel)
#         # print(old_conv_filter_num, len(reserved_filter_group))
#         new_linear_layer = \
#             Linearsnn(old_linear_layer.in_features - params_per_input_channel * (
#                         old_conv_filter_num - len(reserved_filter_group)),
#                             old_linear_layer.out_features)
#
#         old_weights = old_linear_layer.weight.data.cpu().numpy()
#         new_weights = new_linear_layer.weight.data.cpu().numpy()
#         # print(new_conv_filter_num)
#         # print(params_per_input_channel)
#
#         reserved_filter_group.sort(key=lambda a: a)
#         # print(reserved_filter_group)
#
#         for i in range(len(reserved_filter_group)):
#             # print(i, reserved_filter_group[i])
#             new_weights[:, i * params_per_input_channel:(i + 1) * params_per_input_channel] = \
#                 old_weights[:, reserved_filter_group[i] * params_per_input_channel:(reserved_filter_group[
#                                                                                         i] + 1) * params_per_input_channel]
#             # print(i)
#
#         # new_linear_layer.bias.data = old_linear_layer.bias.data
#
#         new_linear_layer.weight.data = torch.from_numpy(new_weights)
#         if use_cuda:
#             new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()
#
#         conv_fc = torch.nn.Sequential(
#             *(replace_layers(model.conv_fc, i, [layer_index], \
#                              [new_linear_layer]) for i, _ in enumerate(model.conv_fc)))
#         model.conv_fc = conv_fc #fixme all classifer
#         # classifier = torch.nn.Sequential(
#         #     *(replace_layers(model.classifier, i, [layer_index], \
#         #                      [new_linear_layer]) for i, _ in enumerate(model.classifier)))
#         # model.classifier = classifier #origninal code
#         functional.set_step_mode(model,step_mode='m')
#         return model
def prune_csnn(model, deleted_layer_index, reserved_filter_group, bias=False, use_cuda=True):
    def replace_layers(model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    conv_16_index = {0: 0, 1: 4}
    if deleted_layer_index != 1:
        # current conv
        _, old_conv1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index]]


        # new_conv1 = torch.nn.Conv2d(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
        #                             kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
        #                             padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
        #                             bias=(old_conv1.bias is not None))

        new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None), step_mode="m")

        old_conv1_weights = old_conv1.weight.data.cpu().numpy()

        new_conv1_weights = new_conv1.weight.data.cpu().numpy()

        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]

        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        # # conv bias
        # if bias == True:
        #     old_conv1_bias = old_conv1.bias.data.cpu().numpy()
        #     new_conv1_bias = new_conv1.bias.data.cpu().numpy()
        #     new_conv1_bias = old_conv1_weights[reserved_filter_group]
        #     new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
        # following bn
        _, old_bn1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        # next_conv
        _, old_conv2 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index + 1]]
        new_conv2 = Conv2dsnn(in_channels=len(reserved_filter_group), out_channels=old_conv2.out_channels,
                                    kernel_size=old_conv2.kernel_size, stride=old_conv2.stride,
                                    padding=old_conv2.padding, dilation=old_conv2.dilation, groups=old_conv2.groups,
                                    bias=(old_conv2.bias is not None), step_mode="m")
        old_conv2_weights = old_conv2.weight.data.cpu().numpy()
        new_conv2_weights = new_conv2.weight.data.cpu().numpy()
        new_conv2_weights[:, :, :, :] = old_conv2_weights[:, reserved_filter_group, :, :]
        new_conv2.weight.data = torch.from_numpy(new_conv2_weights)
        if bias == True:
            new_conv2.bias.data = old_conv2.bias.data

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()
            new_conv2.cuda()

        features = torch.nn.Sequential(
            *(replace_layers(model.conv_fc, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1,
                              conv_16_index[deleted_layer_index + 1]], \
                             [new_conv1, new_bn1, new_conv2]) for i, _ in enumerate(model.conv_fc)))

        del model.conv_fc
        model.conv_fc = features
        functional.set_step_mode(model,step_mode='m')

        return model

    else:
        # last conv layer
        _, old_conv1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index]]
        new_conv1 = Conv2dsnn(in_channels=old_conv1.in_channels, out_channels=len(reserved_filter_group),
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, dilation=old_conv1.dilation, groups=old_conv1.groups,
                                    bias=(old_conv1.bias is not None))
        old_conv1_weights = old_conv1.weight.data.cpu().numpy()
        new_conv1_weights = new_conv1.weight.data.cpu().numpy()
        new_conv1_weights[:, :, :, :] = old_conv1_weights[reserved_filter_group, :, :, :]
        new_conv1.weight.data = torch.from_numpy(new_conv1_weights)
        if bias == True:
            new_conv1_bias = old_conv1_weights[reserved_filter_group]
            new_conv1.bias.data = torch.from_numpy(new_conv1_bias)
            old_conv1_bias = old_conv1.bias.data.cpu().numpy()
            new_conv1_bias = new_conv1.bias.data.cpu().numpy()

        # following bn
        _, old_bn1 = list(model.conv_fc._modules.items())[conv_16_index[deleted_layer_index] + 1]
        old_bn1_weight = [old_bn1.weight.data.cpu().numpy(), \
                          old_bn1.bias.data.cpu().numpy(), \
                          old_bn1.running_mean.data.cpu().numpy(), \
                          old_bn1.running_var.data.cpu().numpy()]
        new_bn1 = BatchNorm2dsnn(len(reserved_filter_group))
        new_bn1_weight = [new_bn1.weight.data.cpu().numpy(), \
                          new_bn1.bias.data.cpu().numpy(), \
                          new_bn1.running_mean.data.cpu().numpy(), \
                          new_bn1.running_var.data.cpu().numpy()]
        for i in range(len(new_bn1_weight)):
            new_bn1_weight[i] = old_bn1_weight[i][reserved_filter_group]
        new_bn1.weight.data = torch.from_numpy(new_bn1_weight[0])
        new_bn1.bias.data = torch.from_numpy(new_bn1_weight[1])
        new_bn1.running_mean.data = torch.from_numpy(new_bn1_weight[2])
        new_bn1.running_var.data = torch.from_numpy(new_bn1_weight[3])

        if use_cuda:
            new_conv1.cuda()
            new_bn1.cuda()

        model.conv_fc = torch.nn.Sequential(
            *(replace_layers(model.conv_fc, i,
                             [conv_16_index[deleted_layer_index], conv_16_index[deleted_layer_index] + 1], \
                             [new_conv1, new_bn1]) for i, _ in enumerate(model.conv_fc)))

        # first linear layer
        old_conv_filter_num = old_conv1.out_channels
        layer_index = 0
        old_linear_layer = None
        for _, module in model.conv_fc._modules.items():
            if isinstance(module, Linearsnn):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

        params_per_input_channel = old_linear_layer.in_features // old_conv1.out_channels
        # print(params_per_input_channel)
        # print(old_conv_filter_num, len(reserved_filter_group))
        new_linear_layer = \
            Linearsnn(old_linear_layer.in_features - params_per_input_channel * (
                        old_conv_filter_num - len(reserved_filter_group)),
                            old_linear_layer.out_features,bias=False)
        # old_weights2 =old_linear_layer.weight.data

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        # print(new_conv_filter_num)
        # print(params_per_input_channel)

        reserved_filter_group.sort(key=lambda a: a)
        # print(reserved_filter_group)

        for i in range(len(reserved_filter_group)):
            # print(i, reserved_filter_group[i])
            new_weights[:, i * params_per_input_channel:(i + 1) * params_per_input_channel] = \
                old_weights[:, reserved_filter_group[i] * params_per_input_channel:(reserved_filter_group[
                                                                                        i] + 1) * params_per_input_channel]
            # print(i)

        # new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        conv_fc = torch.nn.Sequential(
            *(replace_layers(model.conv_fc, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.conv_fc)))
        model.conv_fc = conv_fc #fixme all classifer
        # classifier = torch.nn.Sequential(
        #     *(replace_layers(model.classifier, i, [layer_index], \
        #                      [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        # model.classifier = classifier #origninal code
        functional.set_step_mode(model,step_mode='m')
        return model

