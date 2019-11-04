import io
import itertools
import logging
import sys

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TqdmToLogger(io.StringIO):
    def __init__(self, logger, level=None):
        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def get_tqdm_pbar(it, logger, level, **kwargs):
    if 'mininterval' not in kwargs:
        kwargs['mininterval'] = 1
    if 'miniters' not in kwargs:
        kwargs['miniters'] = 50
    # FIXME: once I figure out how to get this to work with logging...
    return tqdm.tqdm(it, **kwargs)
    tqdm_out = TqdmToLogger(logger, level=level)
    return tqdm.tqdm(it, file=tqdm_out, **kwargs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pytorch_accuracy(output, target, topk=(1,), bce=False):
    if bce:
        output = F.sigmoid(output)
        output = np.squeeze(output.cpu().numpy())
        target = np.squeeze(target.cpu().numpy())
        return [[np.mean((output > 0.5) == target) * 100]]
    else:
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_metric(output, target, metric):
    if metric == 'topk':
        return pytorch_accuracy(output, target)
    elif metric == 'bce':
        return pytorch_accuracy(output, target, bce=True)
    elif metric == 'l2':
        return np.sqrt((output - target) ** 2)
    elif metric == 'l1':
        return np.sum(np.abs(output - target))
    elif callable(metric):
        return metric(output, target)
    else:
        raise NotImplementedError

def train_epoch(train_loader, model, criterion, optimizer, epoch, metric, silent=False):
    model.train()
    model.cuda()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = get_tqdm_pbar(train_loader, logger, logging.DEBUG)
    for inp, target in pbar:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if not silent:
            pbar.set_description('loss: %2.4f, acc: %2.1f' % (losses.avg, top1.avg))
        input_var = torch.autograd.Variable(inp)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if metric == 'bce':
            target_var = target_var.float()
            if target_var.size() != output.size():
                target_var = target_var.view(output.size())
        loss = criterion(output, target_var)

        prec1 = compute_metric(output.data, target, metric)
        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1[0][0], inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()

def val_epoch(val_loader, model, criterion, metric):
    model.eval()
    model.cuda()
    losses = AverageMeter()
    top1 = AverageMeter()
    for i, (inp, target) in enumerate(val_loader):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(inp, requires_grad=False)
        target_var = torch.autograd.Variable(target, requires_grad=False)

        with torch.no_grad():
            output = model(input_var)
        if metric == 'bce':
            target_var = target_var.float()
            if target_var.size() != output.size():
                target_var = target_var.view(output.size())
        loss = criterion(output, target_var)
        prec1 = compute_metric(output.data, target, metric)

        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1[0][0], inp.size(0))

    return losses.avg, top1.avg

# TODO: should possibly make this into a class, a la torchsample
# scheduler_arg: 'loss' | 'epoch'
# metric: 'bce' | 'topk' | 'l2' | 'l1' | None
def trainer(model, criterion, optimizer, scheduler,
            loaders, # (train_loader, val_loader)
            nb_epochs=50,
            patience=5, save_every=5,
            weight_ckpt_name='weight-epoch{epoch:02d}.t7', weight_best_name='weight.best.t7',
            scheduler_arg='loss',
            return_best=False,
            metric='topk',
            silent=False):
    model.cuda()
    train_loader, val_loader = loaders

    best_loss = (float('Inf'), -1)
    best_acc = (0, -1)
    last_update = -1
    pbar = get_tqdm_pbar(range(nb_epochs), logger, logging.DEBUG)
    for epoch in pbar:
        if scheduler_arg == 'epoch':
            scheduler.step(epoch)
        train_epoch(train_loader, model, criterion, optimizer, epoch, metric, silent=silent)

        val_loss, val_acc = val_epoch(val_loader, model, criterion, metric)
        if not silent:
            pbar.set_description('val loss: %2.4f, val acc: %2.1f' % (val_loss, val_acc))

        if val_loss < best_loss[0]:
            best_loss = (val_loss, epoch)
            last_update = epoch
        if val_acc > best_acc[0]:
            best_acc = (val_acc, epoch)
            last_update = epoch
            torch.save({'state_dict': model.state_dict(), 'acc': val_acc},  weight_best_name)

        if epoch % save_every == 0 and weight_ckpt_name is not None:
            weight_fname = weight_ckpt_name.format(epoch=epoch)
            torch.save({'state_dict': model.state_dict()}, weight_fname)

        if epoch - last_update > patience:
            break

        if scheduler_arg == 'loss':
            scheduler.step(val_loss)
    pbar.close()
    print('', file=sys.stderr)
    sys.stderr.flush()
    logger.debug('Best loss: ' + str(best_loss))
    logger.debug('Best acc: ' + str(best_acc))
    if return_best:
        model.load_state_dict(torch.load(model_best_name)['state_dict'])
    return best_acc[0], best_loss[0]
