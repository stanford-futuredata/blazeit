import argparse
import itertools
import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image

from . import resnet_simple
from .pytorch_utils import *

import torch.utils.data as data
import torchvision
# THIS CLASS DEPENDS ON THE INTERNAL IMPLEMENTATION OF IMAGEFOLDER
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class ImageList(torchvision.datasets.ImageFolder):
    # Images take the form (path, class)
    def __init__(self, classes, imgs, transform=None, target_transform=None,
                 loader=pil_loader):
        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

class RandomRotate(object):
    def __init__(self, rot_range):
        self.rot_range = rot_range
    def __call__(self, img):
        angle = np.random.uniform(-self.rot_range, self.rot_range)
        return img.rotate(angle)

def get_datasets(train_fnames, val_fnames,
                 CLASS_NAMES=None,
                 normalize=None, RESOL=224,
                 batch_size=32, num_workers=16,
                 use_rotate=False):
    NB_CLASSES = len(train_fnames)
    assert NB_CLASSES == len(val_fnames)
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if CLASS_NAMES is None:
        class_names = list(map(str, range(NB_CLASSES)))

    def flatten_fnames(fnames):
        imgs = []
        for ind, imlist in enumerate(fnames):
            tmp = list(zip(imlist, itertools.repeat(ind)))
            imgs += tmp
        return imgs

    train_imgs = flatten_fnames(train_fnames)
    val_imgs = flatten_fnames(val_fnames)

    if use_rotate:
        rotation = [RandomRotate(20)]
    else:
        rotation = []

    train_dataset = ImageList(
            CLASS_NAMES, train_imgs,
            transforms.Compose(rotation + [
                    transforms.RandomSizedCrop(RESOL),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]))
    val_dataset = ImageList(
            CLASS_NAMES, val_imgs,
            transforms.Compose([
                    transforms.Scale(int(256.0 / 224.0 * RESOL)),
                    transforms.CenterCrop(RESOL),
                    transforms.ToTensor(),
                    normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=True, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=False)
    return train_loader, val_loader

RESOL = (65, 65)

TRAIN_DIR = '/lfs/1/ddkang/specializer/imagenet/train'
VAL_DIR = '/lfs/1/ddkang/specializer/imagenet/val'

CLASS_NAMES = os.listdir(TRAIN_DIR)
CLASS_NAMES.sort()
print(CLASS_NAMES)

NB_CLASSES = len(CLASS_NAMES)
FILE_BASE = 'imagenet'

def process_dir(d):
    fnames = []
    for c in os.listdir(d):
        wd = os.path.join(d, c)
        tmp = list(map(lambda x: os.path.join(wd, x), os.listdir(wd)))
        fnames.append(tmp)
    return fnames

train_fnames = process_dir(TRAIN_DIR)
val_fnames = process_dir(VAL_DIR)

train_loader, val_loader = get_datasets(
        train_fnames, val_fnames,
        CLASS_NAMES=CLASS_NAMES,
        RESOL=65,
        num_workers=8)


model_params = [
        # ('trn2', []),
        # ('trn4', [1]),
        # ('trn6', [1, 1]),
        # ('trn8', [1, 1, 1]),
        ('trn10', [1, 1, 1, 1]),
        ('trn18', [2, 2, 2, 2]),
        ('trn34', [3, 4, 6, 3])]
name_to_params = dict(model_params)
BASE_DIR = FILE_BASE
try:
    os.mkdir(BASE_DIR)
except:
    pass
for MODEL_NAME, _ in model_params:
    m1 = '%s/%s-%s-flickr-epoch{epoch:02d}-sgd-cc.t7' % (BASE_DIR, FILE_BASE, MODEL_NAME)
    m2 = '%s/%s-%s-flickr-best-sgd-cc.t7' % (BASE_DIR, FILE_BASE, MODEL_NAME)
    w1 = '%s/%s-%s-weights-flickr-epoch{epoch:02d}-sgd-cc.t7' % (BASE_DIR, FILE_BASE, MODEL_NAME)
    w2 = '%s/%s-%s-weights-flickr-best-sgd-cc.t7' % (BASE_DIR, FILE_BASE, MODEL_NAME)

    model = resnet_simple.rn_builder(
            name_to_params[MODEL_NAME],
            num_classes=NB_CLASSES,
            conv1_size=3, conv1_pad=1, nbf=16,
            downsample_start=False)
    model.cuda()
    print(model)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    trainer(model, criterion, optimizer, scheduler,
            (train_loader, val_loader),
            weight_ckpt_name=w1, weight_best_name=w2,
            model_ckpt_name=m1, model_best_name=m2,
            scheduler_arg='loss')

# Touch file at end
open('%s.txt' % FILE_BASE, 'a').close()
