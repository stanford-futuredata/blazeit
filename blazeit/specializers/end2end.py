import argparse
import logging
import os
import pprint
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from blazeit.data.generate_fnames import get_csv_fname, get_video_fname
from . import trn
from .specializers import *

def run_binary(trainer, Y_pred, Y, csv_fname, objects):
    # Get metrics
    metrics = trainer.metrics(Y_pred, Y)
    print('metrics:')
    pprint.pprint(metrics)
    print('poison_metrics:')
    pprint.pprint(trainer.poison_metrics(Y_pred, Y))

    # Get thresholds
    threshold, nb_passed = trainer.find_threshold(Y_pred, Y)
    print(threshold, nb_passed, np.sum(Y), len(Y))

    # ind_metrics
    object_name = list(objects)[0]
    trainer.ind_metrics(Y_pred)

    # Get incorrect indices
    tmp = (Y_pred > 0.5) == Y
    inds = np.nonzero(1 - tmp)[0]
    df = pd.DataFrame(data={'ind': inds})
    df.to_csv('bad.csv', index=False, header=False)

def run_count(trainer, Y_pred, Y, csv_fname, objects):
    metrics = trainer.metrics(Y_pred, Y)
    print('metrics:')
    pprint.pprint(metrics)

    trainer.ind_metrics(Y_pred, csv_fname)

def run_multiclassbinary(trainer, Y_pred, Y, csv_fname, objects):
    metrics = trainer.metrics(Y_pred, Y)
    print('metrics:')
    pprint.pprint(metrics)

    trainer.ind_metrics(Y_pred, csv_fname)

def run_multiclasscount(trainer, Y_pred, Y, csv_fname, objects):
    metrics = trainer.metrics(Y_pred, Y)
    print('metrics:')
    pprint.pprint(metrics)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/lfs/1/ddkang/blazeit/data/')
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--model_date', required=True)
    parser.add_argument('--objects', default='', required=True)
    train_parser = parser.add_mutually_exclusive_group(required=True)
    train_parser.add_argument('--train', dest='train', action='store_true')
    train_parser.add_argument('--no-train', dest='train', action='store_false')
    vid_parser = parser.add_mutually_exclusive_group(required=True)
    vid_parser.add_argument('--load_video', dest='load_video', action='store_true')
    vid_parser.add_argument('--no-load_video', dest='load_video', action='store_false')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    DATE = args.date
    MODEL_DATE = args.model_date
    trn_name = 'trn18'
    base_name = args.base_name
    objects = set(args.objects.split(','))
    train = args.train
    normalize = True

    task_type = 'count'
    if task_type == 'binary':
        specializer = BinarySpecializer
        nb_classes = 1
        eval_runner = run_binary
        spec_kwargs = {}
    elif task_type == 'count':
        specializer = CountSpecializer
        nb_classes = 5 # FIXME
        eval_runner = run_count
        spec_kwargs = {'max_count': nb_classes - 1}
    elif task_type == 'multiclass_binary':
        specializer = MulticlassBinarySpecializer
        nb_classes = len(objects)
        eval_runner = run_multiclassbinary
        spec_kwargs = {}
    elif task_type == 'multiclass_count':
        specializer = MulticlassCountSpecializer
        max_counts = [1, 4]
        nb_classes = sum(max_counts) + len(max_counts)
        eval_runner = run_multiclasscount
        spec_kwargs = {'max_counts': max_counts}
    else:
        raise NotImplementedError

    csv_fname = get_csv_fname(DATA_PATH, base_name, DATE)
    video_fname = get_video_fname(DATA_PATH, base_name, DATE,
                                  load_video=args.load_video, normalize=normalize)
    if train:
        weights = 'imagenet' if normalize else 'random'
        base_model = trn.create_trn(trn_name, nb_classes, weights=weights)
        model_dump_fname = os.path.join(
                DATA_PATH, 'models', base_name, '%s-%s-%s.t7' % (base_name, MODEL_DATE, trn_name))
    else:
        model_load_fname = os.path.join(
                DATA_PATH, 'models', base_name, '%s-%s-%s.t7' % (base_name, MODEL_DATE, trn_name))
        model_dump_fname = None
        base_model = trn.create_trn(trn_name, nb_classes, weights=model_load_fname)

    spec = specializer(base_name, video_fname, csv_fname, objects, normalize,
                       base_model, model_dump_fname, **spec_kwargs)
    if train:
        spec.load_data()
        spec.train()
        X = spec.X_val
        Y = spec.Y_val
    else:
        X = spec.getX()
        Y = spec.getY()

    # Get predictions
    begin = time.time()
    Y_pred = spec.eval(X)
    Y_pred = Y_pred[:len(Y)]
    Y = Y[:len(Y_pred)]
    end = time.time()
    print('Pred time:', end - begin)

    eval_runner(spec, Y_pred, Y, csv_fname, objects)


if __name__ == '__main__':
    main()
