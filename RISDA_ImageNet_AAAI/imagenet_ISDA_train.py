#coding=utf-8
import argparse
import os
import random
import shutil
import time
import warnings

from numpy.lib.type_check import imag
from scipy.integrate._ivp.radau import P

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import networks.resnet
import networks.densenet
from ISDA_imagenet import ISDALoss
from networks.globalVar import gloVar
import math
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--stage1', default=80, type=int, metavar='N',
                    help='number of stage-1 epochs to run')
parser.add_argument('--stage2', default=90, type=int, metavar='N',
                    help='number of stage-2 fiexd epochs to run')
parser.add_argument('--head', default=385, type=int, metavar='N',
                    help='number of stage-1 epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=None, type=str,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--alpha_0', default=0, type=float,
                    help='The hyper-parameter \alpha_0 for ISDA, select from {0,0.3, 0.5, 0.8,1.0}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')

parser.add_argument('--beta_0', default=7.5, type=float,
                    help='The hyper-parameter \beta_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')


parser.add_argument('--model', default='resnet50', type=str,
                    help='Model to be trained. '
                         'Select from resnet{18, 34, 50, 101, 152} / resnext{50_32x4d, 101_32x8d} / '
                         'densenet{121, 169, 201, 265}')
parser.add_argument('--pre-train', default='', type=str, metavar='PATH',
                    help='path to the pre-train model (default: none)')
parser.add_argument('--num_classes', type=int, default=1000)

best_acc1 = 0
acc_list = []

img_num_list=[50, 250, 516, 352, 143, 92, 47, 45, 150, 50, 34, 55, 25, 35, 121, 7, 
        232, 13, 392, 157, 87, 181, 177, 171, 114, 56, 134, 56, 279, 248, 287, 83, 65, 56, 
        118, 43, 68, 39, 14, 96, 80, 15, 236, 103, 64, 54, 346, 32, 201, 29, 376, 27, 189, 
        96, 69, 157, 16, 173, 5, 34, 34, 150, 57, 78, 31, 8, 44, 297, 349, 72, 35, 36, 428, 
        10, 137, 58, 80, 44, 30, 15, 126, 93, 34, 135, 226, 281, 1280, 19, 38, 502, 186, 81, 
        61, 222, 15, 28, 55, 819, 74, 24, 10, 93, 14, 111, 61, 32, 116, 133, 128, 66, 35, 50, 
        393, 47, 179, 124, 79, 604, 72, 240, 22, 148, 26, 64, 20, 23, 246, 11, 79, 161, 78, 
        370, 197, 155, 285, 59, 59, 178, 43, 75, 11, 23, 16, 74, 27, 87, 86, 105, 189, 37, 126, 53, 10, 30, 77, 5, 245, 30, 100, 37, 69, 119, 26, 47, 81, 100, 25, 35, 112, 35, 7, 35, 272, 469, 150, 100, 199, 137, 69, 42, 10, 20, 367, 49, 73, 38, 56, 75, 89, 24, 35, 244, 305, 23, 106, 77, 10, 59, 90, 33, 26, 214, 43, 50, 18, 125, 110, 119, 85, 43, 11, 11, 114, 171, 24, 54, 87, 39, 86, 496, 91, 52, 56, 11, 117, 48, 62, 100, 41, 138, 22, 7, 50, 51, 6, 281, 29, 13, 36, 75, 126, 127, 59, 56, 172, 119, 90, 163, 17, 53, 181, 81, 63, 167, 45, 37, 39, 282, 71, 32, 12, 137, 102, 97, 60, 14, 66, 14, 6, 132, 303, 81, 167, 28, 15, 72, 187, 20, 310, 137, 241, 123, 27, 19, 186, 12, 12, 42, 71, 96, 71, 53, 53, 19, 62, 43, 330, 41, 11, 42, 43, 22, 208, 18, 566, 46, 19, 10, 58, 306, 67, 119, 50, 51, 87, 143, 37, 16, 611, 290, 170, 114, 19, 79, 81, 128, 77, 332, 8, 446, 42, 122, 86, 1121, 48, 109, 133, 7, 174, 17, 283, 140, 27, 15, 30, 134, 167, 198, 149, 16, 122, 23, 194, 149, 23, 156, 5, 127, 11, 31, 49, 143, 145, 169, 20, 158, 87, 128, 96, 65, 49, 391, 349, 83, 123, 232, 12, 51, 137, 31, 73, 32, 41, 65, 6, 6, 85, 231, 171, 28, 108, 96, 101, 32, 122, 26, 165, 10, 372, 34, 31, 63, 679, 34, 257, 27, 83, 23, 
        275, 13, 22, 47, 118, 411, 159, 124, 45, 1053, 111, 184, 68, 52, 41, 150, 10, 47, 109, 32, 69, 34, 38, 10, 30, 82, 16, 157, 14, 141, 10, 58, 25, 152, 165, 112, 480, 91, 43, 113, 22, 236, 48, 137, 48, 147, 511, 17, 86, 33, 44, 36, 333, 19, 48, 15, 224, 108, 152, 16, 186, 114, 31, 82, 125, 155, 59, 186, 188, 134, 25, 121, 288, 39, 8, 32, 41, 346, 127, 232, 123, 81, 652, 6, 154, 203, 48, 66, 178, 45, 6, 22, 85, 102, 44, 105, 44, 54, 42, 20, 122, 220, 82, 44, 43, 75, 8, 142, 82, 38, 231, 619, 162, 62, 134, 8, 71, 41, 86, 55, 116, 35, 94, 100, 330, 54, 87, 143, 110, 191, 6, 48, 153, 84, 16, 55, 10, 422, 92, 25, 507, 168, 104, 83, 70, 47, 382, 148, 10, 164, 30, 310, 381, 174, 92, 87, 268, 202, 1173, 110, 8, 125, 51, 340, 37, 209, 71, 14, 14, 46, 252, 14, 239, 58, 22, 112, 22, 101, 130, 278, 18, 220, 21, 253, 39, 292, 317, 179, 36, 305, 32, 65, 169, 683, 98, 376, 125, 39, 8, 28, 298, 132, 43, 69, 80, 413, 19, 25, 354, 120, 72, 86, 50, 76, 72, 102, 310, 21, 26, 119, 140, 75, 63, 57, 10, 10, 31, 106, 54, 100, 90, 64, 88, 34, 90, 36, 85, 82, 6, 64, 86, 29, 6, 30, 204, 27, 209, 115, 24, 72, 253, 55, 10, 33, 104, 38, 46, 296, 26, 128, 170, 15, 66, 130, 138, 28, 1246, 107, 63, 75, 108, 167, 61, 26, 83, 77, 42, 113, 103, 264, 59, 16, 49, 29, 33, 33, 481, 29, 45, 24, 38, 165, 82, 36, 15, 60, 62, 72, 73, 25, 8, 182, 99, 283, 101, 297, 28, 194, 26, 141, 14, 78, 13, 67, 82, 63, 103, 454, 6, 22, 66, 93, 8, 11, 131, 137, 40, 66, 25, 472, 9, 183, 32, 14, 43, 299, 56, 40, 392, 5, 20, 31, 32, 83, 194, 81, 80, 787, 7, 137, 50, 9, 133, 34, 174, 315, 35, 290, 9, 263, 164, 86, 129, 20, 53, 175, 43, 775, 11, 54, 12, 137, 31, 87, 32, 229, 96, 161, 126, 271, 221, 27, 28, 36, 14, 85, 197, 82, 76, 15, 63, 130, 11, 85, 134, 211, 21, 328, 302, 205, 21, 53, 102, 93, 111, 19, 41, 17, 218, 128, 242, 188, 166, 105, 46, 7, 189, 20, 73, 247, 229, 35, 343, 95, 146, 57, 285, 43, 60, 84, 64, 28, 17, 44, 18, 39, 175, 25, 60, 134, 8, 54, 46, 31, 17, 10, 9, 143, 114, 101, 157, 70, 6, 75, 192, 64, 65, 5, 8, 9, 31, 15, 642, 31, 107, 11, 69, 459, 238, 19, 31, 134, 252, 25, 456, 15, 80, 102, 274, 6, 32, 50, 140, 104, 112, 133, 28, 182, 170, 76, 173, 272, 20, 102, 30, 138, 150, 116, 
        110, 216, 42, 39, 116, 16, 13, 215, 85, 155, 19, 7, 36, 66, 
        323, 22, 284, 172, 398, 110, 56, 25, 53, 534, 43, 38, 26, 418, 141, 7, 36, 55, 20, 63, 45, 18, 80, 67, 151, 94, 103, 284, 180, 23, 175, 472, 133, 99, 71, 67, 80, 32, 113, 17, 57, 172, 17, 19, 93, 124, 23, 55, 258, 108, 72, 33, 165, 95, 52, 45, 625, 73, 217, 98, 45, 34, 41, 39, 714, 143, 141, 233, 119, 63, 236, 150, 19, 15, 
        35, 167, 19, 25, 21, 21]


beta = 0.9999
effective_num = 1.0 - np.power(beta, img_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights)
weights = torch.tensor(per_cls_weights).float()

best_acc1_train=0
kg=torch.zeros(len(img_num_list),len(img_num_list))
feature_mean=torch.zeros(len(img_num_list),2048)
out_new=torch.zeros(len(img_num_list),2048)
class_id = [i for i in range(len(img_num_list))]

def main():
    global out_new
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
       

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
   
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc1_train, kg, feature_mean, class_id,weights,out_new
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'res' in args.model:
        model = eval('networks.resnet.' + args.model)()
    elif 'dense' in args.model:
        model = eval('networks.densenet.' + args.model)()
    else:
        print('Please select the model from resnet{18, 34, 50, 101, 152} / '
              'resnext{50_32x4d, 101_32x8d} / densenet{121, 169, 201, 265}')

    if args.pre_train:
        pre_train_model = torch.load(args.pre_train)
        model.load_state_dict(pre_train_model)

    feature_num = model.feature_num
    
    print('Number of final features: {}'.format(
        int(model.feature_num))
    )
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
   
    criterion_isda = ISDALoss(feature_num, len(img_num_list)).cuda(args.gpu)
  
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            criterion_isda =  checkpoint['criterion_isda']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code  #在这里改变data_loading就可以了
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(args.evaluate)
   
    if args.evaluate:
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        test_acc,cf=validate(test_loader, model,True, args)
        print(cf)
        print(test_acc)

        return
    

    weights=weights.cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        alpha=args.alpha_0 * (epoch / args.epochs)
        beta=args.beta_0 * (epoch / args.epochs)

        if epoch<args.stage1:  #80
            train(train_loader, model, optimizer, epoch,args)
        else:
            if epoch>=args.stage2:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in model.named_parameters():
                    if 'fc' not in param_name:
                        param.requires_grad = False
                        
            if epoch==args.stage1:
                acc1_train,cf_normalized = validate(train_loader, model, True,args)
                kg=cf_normalized
                # torch.save(kg,'kg_80.pkl')
                # kg=torch.load('kg_80.pkl')
                kg=torch.tensor(kg).cuda(args.gpu)  
                kg=kg.to(torch.float32)
                print('kg',kg)
                
                sth = gloVar()
                sth.__init__()
                # feature_mean
                feature_mean=get_feature_mean(train_loader, model,args)
                print(feature_mean)
                # torch.save(feature_mean,'feature_mean_log_all_nocolor.pkl')
                # feature_mean=torch.load('feature_mean_log_all_nocolor.pkl')
                feature_mean=feature_mean.to(torch.float32).cuda(args.gpu)
                print(feature_mean)
                
                out_new=torch.matmul(kg,feature_mean) #updated reasoning mu
                out_new=out_new-feature_mean  #displacement of mu
                zero=torch.zeros(len(img_num_list),2048).cuda()
                c=out_new-zero
                out_new=torch.where(c > 0, out_new,zero) #displacement of mu
                sth.set_value(out_new)
                sth.set_feature_mean(feature_mean)
                #select the tail class 
                k=args.head  #num of head class
                img_num_list_arry=np.array(img_num_list)
                index = np.argpartition(img_num_list_arry,-k,axis=0)[-k:]
                index_tail=list(set(class_id).difference(set(index)))#index of tail class
                del img_num_list_arry,index,out_new,feature_mean,zero

            train_ISDA(train_loader, model, criterion_isda, optimizer, epoch,weights, kg, index_tail, alpha, beta,sth,args)
           
        # evaluate on validation set
        acc1,cf_normalized = validate(val_loader, model, False, args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        acc_list.append(acc1)
        print(acc1)
        print(acc_list)
        if epoch>args.stage1:
            np.savetxt('./log/beta7.5_accuracy.txt', np.array(acc_list))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'criterion_isda': criterion_isda,
        }, is_best)
        
        
        # if epoch==79:
        #     save_checkpoint_80({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        #     'criterion_isda': criterion_isda,
        #      }, False)

        print(args.lr)
        print('val_best_acc1:')
        print(best_acc1)

     

def train(train_loader, model, optimizer,epoch,args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output  y, features 
        output,feature =  model(images)
        del feature
        loss = F.cross_entropy(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_ISDA(train_loader, model, criterion, optimizer, epoch, weights, kg_cv, index_tail, alpha, beta,sth, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    weights=weights.cuda(args.gpu)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
     
        #criterion_isda  
        loss, output = criterion(model, images, target, alpha, weights, kg_cv, index_tail, beta,sth,args)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model,flag,args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    true_labels = []
    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
         
            # compute output
            output,feature =  model(images)
            del feature
            loss = F.cross_entropy(output, target)
            
            true_labels += list(target.data.cpu().numpy())
            output_numpy = output.data.cpu().numpy()
            preds_output = list(output_numpy.argmax(axis=1))
            preds+=preds_output

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # confusion_matrix
        if flag:
            cf = confusion_matrix(true_labels, preds).astype(float) #69,69 
            cf_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
            cf_normalized= np.round(cf_normalized,2)
            print(cf_normalized)
        else:
            cf_normalized=''

    return top1.avg,cf_normalized

def get_feature_mean(train_loader, model,args):
    model.eval()
    feature_mean_end=torch.zeros(1000,2048)
    with torch.no_grad():
       
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output,features =  model(images)
            del output
            features = features.cpu().data.numpy()
            
            for out, label in zip(features, target):
                feature_mean_end[label]= feature_mean_end[label]+out

        img_num_list_tensor=torch.tensor(img_num_list).unsqueeze(1)
        feature_mean_end=torch.div(feature_mean_end,img_num_list_tensor) 

    return feature_mean_end


def save_checkpoint(state, is_best, filename='checkpoint/checkpoint.pth.tar'):
    torch.save(state, 'checkpoint/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename, 'checkpoint/best.pth.tar')

def save_checkpoint_80(state, is_best, filename='checkpoint80.pth.tar'):
    torch.save(state, filename)
   

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5:
        lr = args.lr  * float(epoch+1) / 5

    else:
        lr = args.lr * ((0.1 ** int((epoch+1) >= 60)) * (0.1 ** int((epoch+1) >= 80)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Pytorch 1.7
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()