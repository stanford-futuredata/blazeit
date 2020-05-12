import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from blazeit.data.generate_fnames import get_csv_fname, get_video_fname
from blazeit.specializers.specializers import CountSpecializer
from blazeit.specializers import tiny_models

class PytorchCount(CountSpecializer):
    def train(self, write_out=False, **kwargs):
        super().train(**kwargs, epochs=[1, 0], lrs=[0.001, 0.001])
        self.write_out = write_out

    def eval(self, X):
        Y_prob = super().eval(X)
        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        return Y_prob

    def get_max_count(self):
        max_count = 0
        Y = self.getY()
        counts = defaultdict(int)
        for y in Y:
            counts[int(y)] += 1
        for num in counts:
            frac = float(counts[num]) / len(Y)
            if frac > 0.01:
                max_count = max(max_count, num)
        return max_count


    def get_pred(self, X):
        Y_prob = self.eval(X)
        Y_max = np.max(Y_prob, axis=1)
        Y_pred = np.argmax(Y_prob, axis=1)

        if self.write_out:
            df = pd.DataFrame(Y_prob)
            df.to_csv('probs.csv', index=False, header=False)

        return Y_pred

def get_nyquist(csv_fname):
    df = pd.read_csv(csv_fname)
    ind_counts = df['ind'].value_counts()
    ind_counts = ind_counts[ind_counts > 3]
    sample_freq = ind_counts.values[int(len(ind_counts) * 0.99)] - 1
    print('Selected', sample_freq, 'as sample frequency')
    return sample_freq

def bootstrap(Y, conf=0.95, nb_bootstrap=10000, nb_samples=50000):
    samples = [np.mean(np.random.choice(Y, nb_samples)) for i in range(nb_bootstrap)]
    low_ind = int(len(samples) * (1 - conf) / 2)
    hi_ind = int(len(samples) - len(samples) * (1 - conf) / 2)
    samples.sort()
    return samples[low_ind], samples[hi_ind]

def get_thresh(Y_prob, Y_true, err=0.05):
    Y_max = np.max(Y_prob, axis=1)
    Y_pred = np.argmax(Y_prob, axis=1)
    tmp = [(ymax, ind) for ind, ymax in enumerate(Y_max)]
    Y_ord = sorted(tmp)

    pred_count = np.sum(Y_pred)
    true_count = float(np.sum(Y_true))
    pred_err = (pred_count - true_count) / true_count * 100.
    print('Thresh pred, true, err: {} {} {}'.
          format(pred_count, true_count, pred_err))
    if len(Y_pred) < len(Y_true):
        Y_pred = np.pad(Y_pred, (0, len(Y_true) - len(Y_pred)), 'constant')
    if len(Y_true) < len(Y_pred):
        Y_true = np.pad(Y_true, (0, len(Y_pred) - len(Y_true)), 'constant')
    print('Thresh bootstrap: {}'.
          format(bootstrap(Y_pred - Y_true)))
    threshold = 0
    ind = 0
    return 0. # FIXME FIXME
    while abs(pred_count - true_count) / true_count > err:
        threshold = Y_ord[ind][0]
        i = Y_ord[ind][1]
        if i >= len(Y_pred) or i >= len(Y_true):
            ind += 1
            continue
        pred_count -= Y_pred[i]
        pred_count += Y_true[i]
        ind += 1
    print(threshold, ind)
    return 0. # FIXME FIXME FIXME
    return threshold

def train_and_test(DATA_PATH, TRAIN_DATE, THRESH_DATE, TEST_DATE,
                   base_name, objects,
                   tiny_name='trn10', normalize=True, nb_classes=-1,
                   load_video=False):
    if nb_classes < 0:
        DATE = TRAIN_DATE
        csv_fname = os.path.join(DATA_PATH, 'filtered', base_name, '%s-%s.csv' % (base_name, DATE))
        spec_kwargs = {'max_count': 10000}
        spec = PytorchCount(base_name, None, csv_fname, objects, normalize,
                            None, None, **spec_kwargs)
        nb_classes = spec.get_max_count() + 1
        print('Selected %d as nb_classes, %d as max_count' % (nb_classes, nb_classes - 1))
    spec_kwargs = {'max_count': nb_classes - 1}

   # setup
    base_model = tiny_models.create_tiny_model(tiny_name, nb_classes, weights='imagenet')
    model_dump_fname = os.path.join(
            DATA_PATH, 'models', base_name, '%s-%s-%s.t7' % (base_name, TRAIN_DATE, tiny_name))


    load_times = []
    total_time = time.time()
    # train
    csv_fname = get_csv_fname(DATA_PATH, base_name, TRAIN_DATE)
    sample_freq = get_nyquist(csv_fname)
    video_fname = get_video_fname(DATA_PATH, base_name, TRAIN_DATE, load_video=load_video)
    spec = PytorchCount(base_name, video_fname, csv_fname, objects, normalize,
                        base_model, model_dump_fname, **spec_kwargs)

    start = time.time()
    spec.load_data(selection='balanced', nb_train=150000)
    end = time.time()
    load_times.append(end - start)
    train_time = time.time()
    spec.train(silent=True)
    train_time = time.time() - train_time

    # thresh
    csv_fname = get_csv_fname(DATA_PATH, base_name, THRESH_DATE)
    video_fname = get_video_fname(DATA_PATH, base_name, THRESH_DATE, load_video=load_video)
    spec_kwargs = {'max_count': 10000}
    spec = PytorchCount(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname, **spec_kwargs)
    start = time.time()
    # Thresholding just requires an estimate of the answer
    X = spec.getX()[::4]
    Y_true = spec.getY()[::4]
    end = time.time()
    load_times.append(end - start)
    thresh_time = time.time()
    Y_prob = spec.eval(X)
    thresh = get_thresh(Y_prob, Y_true)
    thresh_time = time.time() - thresh_time
    del X, Y_prob, Y_true

    # test
    csv_fname = get_csv_fname(DATA_PATH, base_name, TEST_DATE)
    video_fname = get_video_fname(DATA_PATH, base_name, TEST_DATE, load_video=load_video)
    spec = PytorchCount(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname,
                        write_out=True,
                        **spec_kwargs)
    start = time.time()
    X = spec.getX()[::sample_freq]
    Y_true = spec.getY()
    end = time.time()
    load_times.append(end - start)
    # Y_prob = spec.eval(X)
    eval_time = time.time()
    # Y_pred = spec.get_pred(X, Y_true, thresh)
    Y_pred = spec.get_pred(X)

    eval_time = time.time() - eval_time
    pred_count = float(np.sum(Y_pred)) * sample_freq
    true_count = float(np.sum(Y_true))
    print(pred_count, true_count, 100 * (pred_count - true_count) / true_count)

    total_time = time.time() - total_time
    print('Train, thresh, eval time: %.2f, %.2f, %.2f' % (train_time, thresh_time, eval_time))
    print('Times:', total_time - sum(load_times), load_times, total_time)

    return pred_count, sample_freq, Y_pred
