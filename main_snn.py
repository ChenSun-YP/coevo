import logging

from config import *
from utils import *


def main():
    args = parser.parse_args()
    set_logger(args)
    model = spiking_vgg.spiking_vgg11(pretrained=True, spiking_neuron=neuron.IFNode,
                                      surrogate_function=surrogate.ATan(), detach_reset=True)
    train_loader, valid_loader, test_loader = get_data(args)
    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEP_snn.CCEP(model, train_loader, valid_loader, test_loader, args)
    alg.run(args.run_epoch)


if __name__ == '__main__':
    main()