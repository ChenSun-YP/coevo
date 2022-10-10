import logging
import spikingjelly.activation_based.examples.conv_fashion_mnist as s
from config import *

from utils import *
from CCEP_snn import CCEPSNN

from spikingjelly.activation_based.examples.conv_fashion_mnist import CSNN
from spikingjelly.activation_based.model.train_classify import Trainer

def main():

    args = parser.parse_args()

    set_logger(args)



    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)
    net.to(args.device)
    train_dataset2 = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    train_set, valid_set = torch.utils.data.random_split(train_dataset2,
                                                         [int(args.valid_ratio * len(train_dataset2)),
                                                          len(train_dataset2) - int(
                                                              (args.valid_ratio) * len(train_dataset2))])
    test_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    valid_loader= torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    # train_loader, valid_loader, test_loader = get_data(args)
    # writer_test = SummaryWriter(log_dir, flush_secs=600, purge_step=net.epochs) TODO
    # writer_train = SummaryWriter(log_dir, flush_secs=600, purge_step=net.train_times)
    #
    # model = spiking_vgg._spiking_vgg('vgg16_bn', 'D', True, True, True, BatchNorm2d, neuron.IFNode,
    #                                  surrogate_function=surrogate.ATan(), detach_reset=True)


    #    vgg = get_model(args)
    # x = Trainer()
    # dataset, dataset_test, train_sampler, test_sampler = x.load_ImageNet(args=args)
    # model = get_model(args)

    checkpoint = torch.load("./logs/T4_b128_sgd_lr0.1_c128_amp_cupy/checkpoint_max.pth", map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = s.F.one_hot(label, 10).float()
            out_fr = net(img)
            loss = s.F.mse_loss(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    test_time = time.time()
    test_loss /= test_samples
    test_acc /= test_samples
    print('test_acc', test_acc)

    # writer.add_scalar('test_loss', test_loss, epoch)
    # writer.add_scalar('test_acc', test_acc, epoch)

    model = net


    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEPSNN(model, train_loader, valid_loader, test_loader, args)

    alg.run(args.run_epoch)


if __name__ == '__main__':
    main()
# python -m main_snn_2.py -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8
