from config import *
import sys
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
model_names = cifar_model_names + imagenet_model_names + vgg_models_name + ['vgg_imagenet']


def get_data(args):
    trainloader, validloader, testloader = None, None, None
    if args.dataset == 'ImageNet':
        traindir = os.path.join(args.data, 'train')

        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(
            traindir,
            transform_train
        )
        train_dataset2 = datasets.ImageFolder(
            traindir,
            transform_test
        )
        test_dataset = datasets.ImageFolder(
            valdir,
            transform_test
        )
        train_set, valid_set = torch.utils.data.random_split(train_dataset2,
                                                             [int(args.valid_ratio * len(train_dataset2)),
                                                              len(train_dataset2) - int((args.valid_ratio) * len(train_dataset2))]
                                                             ,
                                                             generator=torch.Generator().manual_seed(args.random_seed))
        trainloader = DataLoaderX(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.workers, pin_memory=True)
        validloader = DataLoaderX(valid_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

        testloader = DataLoaderX(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset = datasets.CIFAR10(root='./data_cifar10', train=True, transform=transform_train, download=True)
        trainset2 = datasets.CIFAR10(root='./data_cifar10', train=True, transform=transform_test, download=True)
        train_set, valid_set = torch.utils.data.random_split(trainset2,
                                                             [int(args.valid_ratio * len(trainset)), len(trainset)-int((args.valid_ratio) * len(trainset))])
        trainloader = DataLoaderX(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.workers, pin_memory=True)
        validloader = DataLoaderX(valid_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

        testset = datasets.CIFAR10(root='./data_cifar10', train=False, transform=transform_test)
        testloader = DataLoaderX(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError('Not supported data set')
    return trainloader, validloader, testloader


def get_model(args):
    if args.arch in imagenet_model_names:
        model = torch.nn.DataParallel(imagenet_models.__dict__[args.arch](pretrained=True))
    elif args.arch == 'spike_vgg':
        if args.dataset == 'cifar10':
            # model = vggmodels.__dict__[args.arch](depth=16)

            model = spiking_vgg.spiking_vgg16_bn(pretrained=True, spiking_neuron=neuron.IFNode, norm_layer=BatchNorm2d,
                                                 surrogate_function=surrogate.ATan(), detach_reset=True)
            # model = spiking_vgg._spiking_vgg('vgg16_bn', 'D', True, True, True, BatchNorm2d, neuron.IFNode,
            #                                  surrogate_function=surrogate.ATan(), detach_reset=True,data_path =args.data_path)
            # print(type(model))
            #
            # model_dict = torch.load(args.dict_path)['state_dict']
            # model.load_state_dict(model_dict)
        else:
            model =spiking_vgg.spiking_vgg11(pretrained=True, spiking_neuron=neuron.IFNode,
                                      surrogate_function=surrogate.ATan(), detach_reset=True)
    elif args.arch in cifar_model_names:
        model = torch.nn.DataParallel(cifar_models.__dict__[args.arch]()) #FIX
        model.load_state_dict(torch.load(args.dict_path)['state_dict'])
    elif args.arch == 'vgg':
        if args.dataset == 'cifar10':
            model = vggmodels.__dict__[args.arch](depth=16)

            model_dict = torch.load(args.dict_path)['state_dict']
            model.load_state_dict(model_dict)
        else:
            model = torch.nn.DataParallel(vgg_imagenet_models.__dict__['vgg16_bn'](pretrained=True))

    else:
        raise NotImplementedError('Not supported architecture')
    model.cuda()
    return model


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_logger(args):
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%m-%d %H:%M')
    c = ''
    if args.use_crossover:
        c = 'c'
    args.save_path = args.save_path + '/' + f'pruning_{args.dataset}_{args.arch}_{c}_' + time.strftime("%m_%d_%H_%M_%S", time.localtime()) + f'_{args.valid_ratio}_{args.pop_init_rate}_{args.ft_epoch}_{args.random_seed}'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fh = logging.FileHandler(
        f'{args.save_path}/pruning_{args.dataset}_{args.arch}_{c}_{time.strftime("%m-%d", time.localtime())}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("PyThon  version : {}".format(sys.version.replace('\n', ' ')))
    logger.info("PyTorch version : {}".format(torch.__version__))
    logger.info("cuDNN   version : {}".format(torch.backends.cudnn.version()))
    logger.info("Vision  version : {}".format(torchvision.__version__))
    logger.info(f"Network: {args.arch}")
    logger.info(f"Dataset: {args.dataset}")
    if args.dataset != 'ImageNet':
        logger.info(f"Original Model: {args.dict_path}")
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Fine-tune after pruning: {args.ft_epoch}')
    logger.info(f'Fine-tune lr milestone: {args.lr_milestone}')
    logger.info(f"Running Epoch: {args.run_epoch}")
    logger.info(f"Population Init Rate: {args.pop_init_rate}")
    logger.info(f"Prune limitation: {args.prune_limitation}")
    logger.info(f"Evolution Round: {args.evolution_epoch}")
    logger.info(f'Random seed:{args.random_seed}')
    logger.info(f'Population Size: {args.pop_size}')
    logger.info(f'Using crossover: {args.use_crossover}')
    if args.use_crossover:
        logger.info(f'Crossover rate: {args.crossover_rate}')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



