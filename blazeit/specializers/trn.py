import os

import torch
import torch.nn as nn

from . import resnet_simple

# weights: imagenet | random | path
def create_trn(trn_name, nb_classes, weights='imagenet'):
    DATA_PATH = '/lfs/1/ddkang/blazeit/data'
    trn_name_to_layers = \
        [('trn10', [1, 1, 1, 1]),
         ('trn18', [2, 2, 2, 2]),
         ('trn34', [3, 4, 6, 3])]
    trn_name_to_layers = dict(trn_name_to_layers)

    base_model = resnet_simple.PytorchResNet(
            trn_name_to_layers[trn_name], num_classes=1000,
            conv1_size=3, conv1_pad=1, nbf=16, downsample_start=False)

    if weights == 'imagenet':
        model_load_fname = os.path.join(DATA_PATH, 'models/trn/%s.t7' % trn_name)
        base_model.load_state_dict(torch.load(model_load_fname)['state_dict'])
        base_model.fc = nn.Linear(128, nb_classes)
    elif weights == 'random':
        base_model.fc = nn.Linear(128, nb_classes)
    else:
        base_model.fc = nn.Linear(128, nb_classes)
        base_model.load_state_dict(torch.load(weights)['state_dict'])

    return base_model
