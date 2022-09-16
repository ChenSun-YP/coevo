import argparse
import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import arch.resnet_cifar10 as resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp_Resnet56_222', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%m-%d %H:%M')


    fh = logging.FileHandler(
        f'{args.save_dir}/pruning_{args.arch}_{time.strftime("%m-%d", time.localtime())}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading chechpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("best_prec1: ", best_prec1)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], last_epoch = -1)

    if args.arch in ['resnet1202', 'resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
            print(param_group['lr'])
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion, args)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1, 
                            }, is_best, file_name = os.path.join(args.save_dir,'best_model.th'))

        if epoch > 0 and epoch % args.save_every == 0:
            # sd = os.path.join(args.save_dir,'model.th')
            save_checkpoint({   'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec1': best_prec1, }
                                , is_best, file_name = os.path.join(args.save_dir,'checkpoint.th'))
        # sd = os.path.join(args.save_dir,'model.th')
        save_checkpoint({'state_dict': model.state_dict(),
                        'best_prec1': best_prec1, 
                        }, is_best, file_name = os.path.join(args.save_dir,'model.th'))


def validate(test_loader, model, criterion, args, print_result = True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    logger = logging.getLogger()
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            labels = labels.cuda()
            inputs_var = inputs.cuda()
            labels_var = labels.cuda()

            output = model(inputs_var).float()
            loss = criterion(output, labels_var).float()

            prec1 = accuracy(output.data, labels)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and print_result == True:
                logger = logging.getLogger()
                logger.info('Test [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f}({top1.avg:.3f})'.format(i, len(test_loader), batch_time = batch_time, loss = losses, top1 = top1)
                            )
        if print_result:
            logger.info('Valid error:'+str('%.3f' % top1.avg))

    # print(losses.avg) 
    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    logger = logging.getLogger()
    model.train()

    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = labels.cuda()
        inputs_var = inputs.cuda()
        labels_var = labels

        output = model(inputs_var)
        loss = criterion(output, labels_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, labels)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            logger.info(  'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time = batch_time, data_time = data_time, loss = losses, top1 = top1)
                    )
    logger.info(' *Prec@1 {top1.avg:.3f}'.format(top1 = top1))


def save_checkpoint(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        # print('val',self.val)


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.reshape(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
