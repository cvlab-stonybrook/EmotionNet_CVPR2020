from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import shutil
from PyUtils.file_utils import get_dir
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#TODO: directly from the original code:


def save_checkpoint(state, is_best, file_directory=None, epoch=0):
    filename = 'checkpoint_{:04d}.pth.tar'
    if file_directory is None:
        file_directory = ''
    else:
        file_directory = get_dir(file_directory)
    file_path = os.path.join(file_directory, filename.format(epoch))
    torch.save(state, file_path)

    if is_best:
        model_best_path = os.path.join(file_directory, 'model_best.pth.tar')
        shutil.copyfile(file_path, model_best_path)


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


def adjust_learning_rate(optimizer, epoch, default_lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = default_lr * (0.1 ** (epoch // 30))
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multihots(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        target_value = torch.gather(target, 1, pred)

        correct_k = (target_value > 0).float().sum(0, keepdim=False)
        res = (correct_k.mul_(100.0 / batch_size))
        return res


def freeze_all_except_fc(model):
    def dfs_freeze(model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)
    dfs_freeze(model)
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    return model


def freeze_all(model):
    def dfs_freeze(model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)
    dfs_freeze(model)
    return model



def unfreeze_all(model):
    def dfs_freeze(model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True
            dfs_freeze(child)
    dfs_freeze(model)
    # model.fc.weight.requires_grad = True
    # model.fc.bias.requires_grad = True
    return model


def report_params(model, print_func=None):
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_func is None:
        print("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))
    else:
        print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))
