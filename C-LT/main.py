import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
import shutil

# parse arguments
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.1)
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
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
args = parser.parse_args()
print(args)

kwargs = {'num_workers': 1, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_data_meta,train_data,test_dataset = build_dataset(args.dataset,args.num_meta)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

np.random.seed(42)
random.seed(42)
# make imbalanced data
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}


for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.train_labels) if label == j]


img_num_list = get_img_num_per_cls(args.dataset,args.imb_factor,args.num_meta*args.num_classes)
print(img_num_list)
print(sum(img_num_list))

im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

print(len(idx_to_del))

imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.train_labels = np.delete(train_loader.dataset.train_labels, idx_to_del, axis=0)
imbalanced_train_dataset.train_data = np.delete(train_loader.dataset.train_data, idx_to_del, axis=0)
imbalanced_train_loader = torch.utils.data.DataLoader(
    imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


############################################ imbalanced test loader ###########################
'''
data_list_test = {}
for j in range(args.num_classes):
    data_list_test[j] = [i for i, label in enumerate(test_dataset.test_labels) if label == j]


img_num_list_test = get_img_num_per_cls_test(args.dataset,0.01,num_meta=0)
random.shuffle(img_num_list_test)
print(img_num_list_test)
print(sum(img_num_list_test))
im_tsdata = {}
idx_to_del_ts = []
for cls_idx_ts, img_id_list_ts in data_list_test.items():
    random.shuffle(img_id_list_ts)
    img_num = img_num_list_test[int(cls_idx_ts)]
    im_tsdata[cls_idx] = img_id_list_ts[img_num:]
    idx_to_del_ts.extend(img_id_list_ts[img_num:])
    # import pdb; pdb.set_trace()

print(len(idx_to_del_ts))

imbalanced_test_dataset = copy.deepcopy(test_dataset)
imbalanced_test_dataset.test_labels = np.delete(test_dataset.test_labels, idx_to_del_ts, axis=0)
imbalanced_test_dataset.test_data = np.delete(test_dataset.test_data, idx_to_del_ts, axis=0)
#imbalanced_train_loader = torch.utils.data.DataLoader(
 #   imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    imbalanced_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
'''

#######################################################################################


validation_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0

beta = 0.9999
effective_num = 1.0 - np.power(beta, img_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # print("=> loading checkpoint")
    # checkpoint = torch.load('/home/muhammad/Reweighting_samples/Meta-weight-net_class-imbalance-master/checkpoint/ours/ckpt.best.pth.tar', map_location='cuda:0')
    # args.start_epoch = checkpoint['epoch']
    # best_acc1 = checkpoint['best_acc1']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer_a.load_state_dict(checkpoint['optimizer'])
    # print("=> loaded checkpoint")

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        if epoch < 160:

          train(imbalanced_train_loader, validation_loader,model, optimizer_a,epoch)

        else:
          train_meta(imbalanced_train_loader, validation_loader,model, optimizer_a,epoch)

       
        #tr_prec1, tr_preds, tr_gt_labels = validate(imbalanced_train_loader, model, criterion, epoch)
        # evaluate on validation set
        prec1, preds, gt_labels = validate(test_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        #save_checkpoint(args, {
         #   'epoch': epoch + 1,
         #   'state_dict': model.state_dict(),
         #   'best_acc1': best_prec1,
         #   'optimizer' : optimizer_a.state_dict(),
        #}, is_best)

    print('Best accuracy: ', best_prec1)


def train(train_loader, validation_loader,model,optimizer_a,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    weight_eps_class = [0 for _ in range(int(args.num_classes))]
    total_seen_class = [0 for _ in range(int(args.num_classes))]
    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        #meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        #meta_model.load_state_dict(model.state_dict())

        #meta_model.cuda()

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w) # * w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        #meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        #meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def train_meta(train_loader, validation_loader,model,optimizer_a,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    weight_eps_class = [0 for _ in range(int(args.num_classes))]
    total_seen_class = [0 for _ in range(int(args.num_classes))]
    batch_w_eps = []
    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        target_var = target_var.cpu()

        #import pdb; pdb.set_trace()
        y = torch.eye(args.num_classes)

        labels_one_hot = y[target_var].float().cuda()

        weights = torch.tensor(per_cls_weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        #weights = weights.unsqueeze(1)
        #weights = weights.repeat(1,args.num_classes)

        meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        meta_model.load_state_dict(model.state_dict())

        meta_model.cuda()

        # compute output
        # Lines 4 - 5 initial forward pass to compute the initial weighted loss

        y_f_hat = meta_model(input_var)


        target_var = target_var.cuda()
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)

        weights = to_var(weights)
        eps = to_var(torch.zeros(cost.size()))

        w_pre = weights + eps
        l_f_meta = torch.sum(cost * w_pre)
        meta_model.zero_grad()



        # Line 6-7 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)
        del grads

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        input_validation, target_validation = next(iter(validation_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)

        #import pdb; pdb.set_trace()
        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_metada = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

  
       # import pdb; pdb.set_trace()
        new_eps = eps - 0.01 * grad_eps
        w = weights + new_eps

        del grad_eps

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)

        l_f = torch.mean(cost_w * w)

        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        # meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        #import pdb; pdb.set_trace()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()#async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
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
    # log to TensorBoard
    # import pdb; pdb.set_trace()

    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    #lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_v1(oargs, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
    
    filename = '%s/%s/ckpt.pth.tar' % ('checkpoint', 'ours')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

if __name__ == '__main__':
    main()
