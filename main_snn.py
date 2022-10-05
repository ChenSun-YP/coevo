import logging

from config import *

from utils import *
from CCEP_snn import CCEPSNN


from spikingjelly.spikingjelly.activation_based.model.train_classify import Trainer

def main():
    args = parser.parse_args()
    set_logger(args)

     # model = spiking_vgg.spiking_vgg16_bn(pretrained=True, spiking_neuron=neuron.IFNode,norm_layer=BatchNorm2d,
    #                                   surrogate_function=surrogate.ATan(), detach_reset=True)
    train_loader, valid_loader, test_loader = get_data(args)
    # writer_test = SummaryWriter(log_dir, flush_secs=600, purge_step=net.epochs) TODO
    # writer_train = SummaryWriter(log_dir, flush_secs=600, purge_step=net.train_times)
    #
    # model = spiking_vgg._spiking_vgg('vgg16_bn', 'D', True, True, True, BatchNorm2d, neuron.IFNode,
    #                                  surrogate_function=surrogate.ATan(), detach_reset=True)


    #    vgg = get_model(args)
    # x = Trainer()
    # dataset, dataset_test, train_sampler, test_sampler = x.load_ImageNet(args=args)
    model = get_model(args)



    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEPSNN(model, train_loader, valid_loader, test_loader, args)

    alg.run(args.run_epoch)


if __name__ == '__main__':
    main()
