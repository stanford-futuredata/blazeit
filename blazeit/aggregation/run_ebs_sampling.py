import argparse

import numpy as np
import pandas as pd
import scipy.stats

# from blazeit.sampling.samplers import *
from samplers import *

def get_data(base_name, date, true_only=False, repeat=2):
    true_fname = './csvs-conf/%s-%s-true.csv' % (base_name, date)
    pred_fname = './csvs-conf/%s-%s-prob.csv' % (base_name, date)
    print('Loading', true_fname, pred_fname)

    Y_true = pd.read_csv(true_fname, header=None, names=['val'])['val'].values
    if true_only:
        return None, Y_true
    Y_prob = pd.read_csv(pred_fname, header=None).values

    Y_pred = np.argmax(Y_prob, axis=1)
    def interleave(a):
        c = np.empty((a.size * repeat,), dtype=a.dtype)
        for i in range(repeat):
            c[i::repeat] = a
        return c
    Y_pred = interleave(Y_pred)

    Y_true = Y_true[0:len(Y_pred)]
    Y_pred = Y_pred[0:len(Y_true)]
    return Y_pred, Y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--err_tol', type=float, required=True)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--train_date', required=True)
    args = parser.parse_args()

    base_name = args.base_name
    test_date = args.test_date
    train_date = args.train_date

    Y_pred, Y_true = get_data(base_name, test_date)
    _, Y_train = get_data(base_name, train_date, True)
    true_val = np.sum(Y_true)

    R = max(Y_train) + 1

    print('True number {}, true length {}'.format(true_val, len(Y_true)))

    err_tol = args.err_tol
    conf = 0.05
    nb_trials = 100
    samplers = [TrueSampler, ControlCovariateSampler]
    for sampler in samplers:
        within_error = 0
        sampler = sampler(err_tol, conf, Y_pred, Y_true, R)
        total_samples = 0
        for i in range(nb_trials):
            estimate, nb_samples = sampler.sample()
            err = (estimate - true_val) / len(Y_true) # true_val
            total_samples += nb_samples
            if abs(err) < err_tol:
                within_error += 1
        print(within_error, total_samples / nb_trials, estimate)

if __name__ == '__main__':
    main()
