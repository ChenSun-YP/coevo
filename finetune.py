
import time
from utils import AverageMeter
from utils import accuracy
import train
import copy
import logging



class fine_tune:
    def __init__(self):
        pass

    def finetune_epochstep(self, model, train_loader, criterion, optimizer, print_freq=200):
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

            if i % print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f}({top1.avg:.3f})'.format(
                    '*', i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1)
                )
        logger.info('* Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return model


    def basic_finetune(self, model, epoch, train_loader, test_loader, criterion, optimizer, args, lr_scheduler=None):
        logger = logging.getLogger()
        best_pre = 0
        for i in range(epoch):
            logger.info("Finetune epoch: [{:d}/{:d}]".format(i, epoch))
            logger.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            self.finetune_epochstep(model, train_loader, criterion, optimizer)
            pre = train.validate(test_loader, model, criterion, args, print_result=True)
            print("---")
            if pre > best_pre:
                best_pre = pre
                best_model = copy.deepcopy(model)
            if lr_scheduler is not None:
                lr_scheduler.step()
        return best_model

