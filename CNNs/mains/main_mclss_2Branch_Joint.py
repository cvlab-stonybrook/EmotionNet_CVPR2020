# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# TODO: compared to 2Branch, this version maps image and text into the same space
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)

import argparse
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import CNNs.models as models
import CNNs.utils.util as CNN_utils
from torch.optim import lr_scheduler
from CNNs.dataloaders.utils import none_collate
from PyUtils.file_utils import get_date_str, get_dir, get_stem
from PyUtils import log_utils
import CNNs.datasets as custom_datasets
from CNNs.utils.config import parse_config
from TextClassification.model_DAN import CNN_2Branch as Text_Embed
from TextClassification.model_DAN_2constraints import Text_Transformation
from CNNs.models.resnet import load_state_dict
import torch.nn.functional as F
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args):
    return getattr(module, name)(args)


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN training from Configure Files")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)
    if args.name is None:
        args.name = get_stem(config_file)

    torch.set_default_tensor_type('torch.FloatTensor')
    best_prec1 = 0

    args.script_name = get_stem(__file__)
    current_time_str = get_date_str()
    # if args.resume is None:
    if args.save_directory is None:
        save_directory = get_dir(os.path.join(project_root, 'ckpts', '{:s}'.format(args.name), '{:s}-{:s}'.format(args.ID, current_time_str)))
    else:
        save_directory = get_dir(os.path.join(project_root, 'ckpts', args.save_directory))
    # else:
    #     save_directory = os.path.dirname(args.resume)
    print("Save to {}".format(save_directory))
    log_file = os.path.join(save_directory, 'log-{0}.txt'.format(current_time_str))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)


    print_func = logger.info
    print_func('ConfigFile: {}'.format(config_file))
    args.log_file = log_file

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.pretrained:
        print_func("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)
    else:
        print_func("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False, num_classes=args.num_classes)

    if args.freeze:
        model = CNN_utils.freeze_all_except_fc(model)



    if os.path.isfile(args.text_ckpt):
        print_func("=> loading checkpoint '{}'".format(args.text_ckpt))
        text_data = torch.load(args.text_ckpt)

        print_func("=> loaded checkpoint '{}' for text classification"
              .format(args.text_ckpt))
    else:
        print_func("=> no checkpoint found at '{}'".format(args.text_ckpt))
        return


    text_params = {
        # "Learning_SCHEDULE": [int(i) for i in args.lr_schedule.split(',')],
        # "LEARNING_RATE": args.learning_rate,
        "MAX_SENT_LEN": args.sent_len,
        "BATCH_SIZE": args.batch_size,
        "WORD_DIM": args.word_dim,
        "FILTER_NUM": args.filter_dim,
        "VOCAB_SIZE": len(text_data['tag2idx']), # FIXME: check here!
        "CLASS_SIZE": 8,
        "tag2idx": text_data['tag2idx'],
        "idx2tag": text_data['idx2tag'],
    }
    args.vocab_size = text_params['VOCAB_SIZE']
    text_model = Text_Embed(**text_params)
    text_model.load_state_dict(text_data['model'].state_dict(), strict=False)
    # Not training Text Model, set to eval to freeze BN
    text_model.eval()

    text_transfrom = Text_Transformation(300, 300, 8)





    if args.gpu is not None:
        text_model = text_model.cuda()
        text_transfrom = text_transfrom.cuda(args.gpu)
    else:
        text_model = text_model.cuda()
        text_transfrom = text_transfrom.cuda()


    if args.gpu is not None:
        model = model.cuda(args.gpu)

    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(args.gpu)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.lr_schedule:
        print_func("Using scheduled learning rate")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, [int(i) for i in args.lr_schedule.split(',')], gamma=0.1)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            import collections
            if isinstance(checkpoint, collections.OrderedDict):
                load_state_dict(model, checkpoint, exclude_layers=['fc.weight', 'fc.bias'])


            else:
                load_state_dict(model, checkpoint['state_dict'])
                print_func("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))

            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print_func("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))



    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))

    # Data loading code
    val_dataset = get_instance(custom_datasets, '{0}'.format(args.valloader), args)
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, collate_fn=none_collate)

    if args.evaluate:
        print_func('Validation Only')
        validate(val_loader, model, criterion, args, print_func)
        return
    else:

        train_dataset = get_instance(custom_datasets, '{0}'.format(args.trainloader), args)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=none_collate)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.lr_schedule:
            # CNN_utils.adjust_learning_rate(optimizer, epoch, args.lr)
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print_func("Epoch: [{}], learning rate: {}".format(epoch, current_lr))

        # train for one epoch
        train(train_loader, model, text_model, text_transfrom, criterion, optimizer, epoch, args, print_func)

        # evaluate on validation set
        if val_loader:
            prec1, val_loss = validate(val_loader, model, criterion, args, print_func)
        else:
            prec1 = 0
            val_loss = 0
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        CNN_utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, file_directory=save_directory, epoch=epoch)

        if not args.lr_schedule:
            scheduler.step(val_loss)


def train(train_loader, model, text_model, text_transformation, criterion, optimizer, epoch, args, print_func):
    batch_time = CNN_utils.AverageMeter()
    data_time = CNN_utils.AverageMeter()
    losses_cls = CNN_utils.AverageMeter()
    losses_ebd = CNN_utils.AverageMeter()
    top1 = CNN_utils.AverageMeter()
    top5 = CNN_utils.AverageMeter()

    # switch to train mode
    model.train()
    text_transformation.train()
    text_model.eval()

    end = time.time()

    cos = torch.nn.CosineSimilarity()
    for i, (input, target, text) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        text = text.cuda(args.gpu, non_blocking=True)

        text_class, text_embedding = text_model(text)

        # compute output
        output_cls, output_proj = model(input)
        text_cls, text_proj = text_transformation(text_embedding)

        log_softmax_text_output = F.log_softmax(text_cls, dim=1)
        loss_text_cls = - torch.sum(log_softmax_text_output * target) / text_cls.shape[0]


        log_softmax_output = F.log_softmax(output_cls, dim=1)
        loss_cls = - torch.sum(log_softmax_output * target) / output_cls.shape[0]

        loss_ebd = torch.sum(1 - cos(output_proj, text_proj)) / output_proj.shape[0]

        losses_cls.update(loss_cls.item(), input.size(0))
        losses_ebd.update(loss_ebd.item(), input.size(0))
        loss = loss_cls +  loss_ebd + loss_text_cls

        prec1 = CNN_utils.accuracy_multihots(output_cls, target, topk=(1, 3))

        top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_func('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                  'Loss_ebd {loss_ebd.val:.4f} ({loss_ebd.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_cls=losses_cls, loss_ebd=losses_ebd, top1=top1))

def validate(val_loader, model, criterion, args, print_func):
    if val_loader is None:
        return  0, 0
    batch_time = CNN_utils.AverageMeter()
    losses = CNN_utils.AverageMeter()
    top1 = CNN_utils.AverageMeter()
    # top5 = CNN_utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)


            # compute output
            output, _ = model(input)
            # loss = criterion(output, target)
            log_softmax_output = F.log_softmax(output, dim=1)

            loss = - torch.sum(log_softmax_output * target)/ output.shape[0]
            # measure accuracy and record loss
            prec1 = CNN_utils.accuracy_multihots(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))

            top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_func('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print_func(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()

