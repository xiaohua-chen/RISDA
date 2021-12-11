import os
import time
import argparse
import random
import copy
from scipy.sparse import data
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import shutil
from resnet import *
from loss import *
from sklearn.metrics import confusion_matrix
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 or cifar100[default])')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--alpha', default=1, type=float, help='[0.25, 0.5, 0.75, 1.0,1.5]')
parser.add_argument('--beta', default=0.5, type=float, help='[0.25, 0.5, 0.75, 1.0,1.5]')
parser.add_argument('--head', default=20, type=int, help='[10, 20, 30, 40]')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--save_name', default='name', type=str)
parser.add_argument('--idx', default='0', type=str)
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--rand_number', default=42, type=int, help='fix random number for data sampling')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
kwargs = {'num_workers': 1, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)              ##
#   cudnn.benchmark = False             ##
#   torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")
best_prec1 = 0
best_prec1_train=0
if args.dataset == 'cifar10':
    kg=torch.zeros(10,10)
    feature_mean=torch.zeros(10,64)
else:
    kg=torch.zeros(100,100)
    feature_mean=torch.zeros(100,64)


 
# Data loading code
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                     (4, 4, 4, 4), mode='reflect').squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
elif args.dataset == 'cifar100':
    train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
else:
    warnings.warn('Dataset is not listed')
    exit()
img_num_list = train_dataset.get_cls_num_list()

print('img_num_list:')
print(img_num_list)
# args.img_num_list = img_num_list
train_sampler = None

imbalanced_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

rho= 0.9999
effective_num = 1.0 - np.power(rho, img_num_list)
per_cls_weights = (1.0 - rho) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
weights = torch.tensor(per_cls_weights).float()
print('weights')
print(weights)

def main():
    global args, best_prec1,best_prec1_train,kg,feature_mean
    args = parser.parse_args()

    model = build_model()

    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume)
                # loc = 'cuda:{}'.format(args.gpu)
                # checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print(args.start_epoch)
           
            best_prec1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                print('ok')
                # best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_a.load_state_dict(checkpoint['optimizer'])
            criterion =  checkpoint['criterion']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.deterministic = True
    if not cudnn.deterministic:
        exit()
    cudnn.benchmark = False 

    criterion = RISDA_CE(64, args.dataset == "cifar10" and 10 or 100, cls_num_list=img_num_list,
                                    max_m=0.5, s=30)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        alpha = args.alpha * float(epoch) / float(args.epochs)
        beta = args.beta * float(epoch) / float(args.epochs)
       
        if epoch < 160:
            train(imbalanced_train_loader, model, optimizer_a, epoch)
            prec1_train, preds_train,labels_train,cf_normalized= validate(imbalanced_train_loader, model, nn.CrossEntropyLoss().cuda(), epoch)
            
            is_best_train = prec1_train > best_prec1_train
            if is_best_train:
                print(cf_normalized)
                kg=cf_normalized
                # torch.save(cf_normalized,'cifar100_im100_kg.pkl')  
            best_prec1_train = max(prec1_train, best_prec1_train)
            
        else:
            if epoch==160:
                #obtain kg and prototype
                kg=torch.tensor(kg).cuda()
                kg=kg.to(torch.float32).cuda()
                feature_mean=get_feature_mean(imbalanced_train_loader, model,len(img_num_list))
                feature_mean=feature_mean.to(torch.float32).cuda()

                #use kg to get reasoning prototype
                out_new=torch.matmul(kg,feature_mean)
                out_new=out_new-feature_mean

            if True:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in model.named_parameters():
                    if 'linear' not in param_name:
                        param.requires_grad = False
                    print('  | ', param_name, param.requires_grad)
            
            train_RISDA(imbalanced_train_loader, model, optimizer_a, epoch, criterion, alpha,kg,beta,out_new,feature_mean,args)
        prec1, preds, labels,cf_normalized= validate(test_loader, model, nn.CrossEntropyLoss().cuda(), epoch)
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        # save_checkpoint(args, {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_prec1,
        #     'optimizer': optimizer_a.state_dict(),
        #     'criterion': criterion,
        # }, is_best)
        
        print('Best accuracy: ', best_prec1)

    print('Best accuracy: ', best_prec1)


def train(train_loader, model, optimizer_a, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        _, y_f = model(input_var,target_var,0,0,0)
        del _
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))

def train_RISDA(train_loader, model,optimizer_a, epoch, criterion, alpha,kg,beta,out_new,feature_mean,args):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    kg=kg.cuda()

    for i, (input, target) in enumerate(train_loader):
       
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        cv = criterion.get_cv()
        cv_var = to_var(cv)
        #reasoning prototype and CoVariance
        features, predicts = model(input_var,target_var,out_new,True,alpha)
        cls_loss = criterion(model.linear, features, predicts, target_var, alpha, weights, cv_var, "update",kg,out_new,feature_mean,beta,args.head)
        
        prec_train = accuracy(predicts.data, target_var.data, topk=(1,))[0]

        losses.update(cls_loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        cls_loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input_var,target_var,0,0,0)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    cf = confusion_matrix(true_labels, preds).astype(float) #69,69 
    cf_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    cf_normalized= np.round(cf_normalized,2)
   
    # torch.save(cf_normalized,'kg_cifar100_im200.pkl')
    return top1.avg, preds, true_labels,cf_normalized


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
        print(1)
    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):

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


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    path = 'checkpoint/' + args.idx + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)

def get_feature_mean(imbalanced_train_loader, model,class_num):
    model.eval()
    feature_mean_end=torch.zeros(class_num,64)
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(target, requires_grad=False)

            features, output = model(input_var,target_var,0,0,0)
            features = features.cpu().data.numpy()
            
            for out, label in zip(features, target):
                feature_mean_end[label]= feature_mean_end[label]+out

        img_num_list_tensor=torch.tensor(img_num_list).unsqueeze(1)
       
        feature_mean_end=torch.div(feature_mean_end,img_num_list_tensor) 
       
        return feature_mean_end

if __name__ == '__main__':
    main()