import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'D':
            layers += [nn.Dropout()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, cfg, num_classes=1000, init_weights=True,
                 dense_sizes=[512, 512]):
        super().__init__()

        nb_last_filt = [x for x in cfg if isinstance(x, int)][-1]
        last_size = 65 // 2 ** sum([x == 'M' for x in cfg])

        if len(dense_sizes) == 2:
            cls_layers = [nn.Linear(nb_last_filt * last_size * last_size, dense_sizes[0]),
                          nn.ReLU(True), nn.Dropout(),
                          nn.Linear(dense_sizes[0], dense_sizes[1]),
                          nn.ReLU(True), nn.Dropout(),
                          nn.Linear(dense_sizes[1], num_classes)]
        elif len(dense_sizes) == 1:
            cls_layers = [nn.Linear(nb_last_filt * last_size * last_size, dense_sizes[0]),
                          nn.ReLU(True), nn.Dropout(),
                          nn.Linear(dense_sizes[0], num_classes)]

        self.features = make_layers(cfg)
        self.classifier = nn.Sequential(*cls_layers)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_tvgg(tvgg_name, nb_classes, weights='imagenet'):
    DATA_PATH = '/lfs/1/ddkang/blazeit/data'
    tvgg_name_to_cfg = \
        [('tvgg8', [16, 'M', 32, 'M', 64, 'M', 128, 'M']),
         ('tvgg10', [16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M']),
         ('tvgg12', [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M'])]
    tvgg_name_to_cfg = dict(tvgg_name_to_cfg)

    base_model = VGG(tvgg_name_to_cfg[tvgg_name])

    def reset_classifier(model):
        # model.classifier[-1] = nn.Linear(512, nb_classes)
        model.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, nb_classes),
        )
        model.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 64),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(64, nb_classes),
        )

    if weights == 'imagenet':
        model_load_fname = os.path.join(DATA_PATH, 'models/tvgg/%s.t7' % tvgg_name)
        base_model.load_state_dict(torch.load(model_load_fname)['state_dict'])
        reset_classifier(base_model)
    elif weights == 'random':
        reset_classifier(base_model)
    else:
        reset_classifier(base_model)
        base_model.load_state_dict(torch.load(weights)['state_dict'])

    return base_model

def create_noscope(noscope_name, nb_classes):
    cfgs = \
        [('ns_32_32_1', ([32, 32, 'M', 'D', 64, 64, 'M', 'D'], [32]))]
    cfgs = dict(cfgs)

    cfg, dense_sizes = cfgs[noscope_name]
    base_model = VGG(cfg, dense_sizes=dense_sizes, num_classes=nb_classes)
    return base_model
