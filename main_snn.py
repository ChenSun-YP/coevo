import logging

from config import *
from utils import *


def main():
    args = parser.parse_args()
    set_logger(args)

    model =spiking_vgg._spiking_vgg('vgg16_bn', 'D', True, True, True, BatchNorm2d, neuron.IFNode,  surrogate_function=surrogate.ATan(), detach_reset=True)
    # model = spiking_vgg.spiking_vgg16_bn(pretrained=True, spiking_neuron=neuron.IFNode,norm_layer=BatchNorm2d,
    #                                   surrogate_function=surrogate.ATan(), detach_reset=True)
    # train_loader, valid_loader, test_loader = get_data(args)
    # writer_test = SummaryWriter(log_dir, flush_secs=600, purge_step=net.epochs) TODO
    # writer_train = SummaryWriter(log_dir, flush_secs=600, purge_step=net.train_times)


    vgg = model
<<<<<<< HEAD
    #    vgg = get_model(args)
=======
    # vgg = get_model(args)
>>>>>>> b6c5ae54464d627319e92403776a6d9024b0b87e

    x = []
    i=0
    for layer in vgg.features:
        print(type(layer),i)
        i+=1
        # if isinstance(layer, nn.Conv2d):
        #     print(layer.weight.shape)
        # elif isinstance(layer, nn.MaxPool2d):
        #     print(layer)






    exit()
    for m in vgg.modules():
        print('---')
        print(m.classifier)
        print('---')


    print(vgg.modules()[0])
    exit()

    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEP_snn(model, train_loader, valid_loader, test_loader, args)

    alg.run(args.run_epoch)


if __name__ == '__main__':
    main()
