import logging

from config import *
from utils import *


def main():
    args = parser.parse_args()
    set_logger(args)
    model = get_model(args)
    train_loader, valid_loader, test_loader = get_data(args)
    logger = logging.getLogger()
    logger.info("=> Model : {}".format(model))

    logger = logging.getLogger()
    logger.info('START PRUNING:')
    alg = CCEP.CCEP(model, train_loader, valid_loader, test_loader, args)
    alg.run(args.run_epoch)

    
if __name__ == '__main__':
    main()