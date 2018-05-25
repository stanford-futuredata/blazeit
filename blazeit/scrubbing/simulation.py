import argparse
import os

import numpy as np
import pandas as pd
import torch

from blazeit.specializers.specializers import MulticlassCountSpecializer

def get_yprobs(max_counts):
    df = pd.read_csv('scrubbing.csv')
    Y_prob_all = df.values
    Y_probs = []
    start = 0
    for i, count in enumerate(max_counts):
        count += 1
        Y_prob = Y_prob_all[:, start:start + count]
        start += count
        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        Y_probs.append(Y_prob)
    return Y_probs

def limit_query(Y_probs, Y_true, limit, correct_counts, distance=300):
    def is_correct(frame):
        if frame >= len(Y_true):
            return False
        for i, count in enumerate(correct_counts):
            if Y_true[frame][i] < count:
                return False
        return True

    Y_prob = []
    for i in range(len(Y_probs[0])):
        tmp = [Y_probs[j][i] for j in range(len(correct_counts))]
        Y_prob.append(tmp)
    Y_sort = sorted(enumerate(Y_prob),
                    key=lambda x: sum([x[1][j][-1] for j in range(len(correct_counts))]),
                    reverse=True)

    blacklist = set()
    correct = []
    nb_calls = 0
    for frame, probs in Y_sort:
        if frame in blacklist:
            continue
        nb_calls += 1
        if is_correct(frame):
            correct.append(frame)
            for i in range(-distance, distance + 1):
                blacklist.add(i + frame)
        if len(correct) >= limit:
            break
    print('Called %d times' % nb_calls)
    print('len(correct): %d' % len(correct))
    return correct, nb_calls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/lfs/1/ddkang/blazeit/data')
    parser.add_argument('--limit', required=True, default=10, type=int)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--objects', default='', required=True)
    parser.add_argument('--counts', default='', required=True)
    args = parser.parse_args()

    DATA_PATH = args.data_path
    base_name = args.base_name
    TEST_DATE = args.test_date
    DATE = TEST_DATE
    limit = args.limit
    objects = args.objects.split(',')
    max_counts = list(map(int, args.counts.split(',')))
    if len(objects) != len(max_counts):
        raise RuntimeError('len(objects) must equal len(counts)')
    if sorted(objects) != objects:
        raise RuntimeError('objects must be in sorted order')

    spec_kwargs = {'max_counts': max_counts}
    csv_fname = os.path.join(DATA_PATH, 'filtered', base_name, '%s-%s.csv' % (base_name, DATE))
    spec = MulticlassCountSpecializer(
            base_name, None, csv_fname, objects,
            True, None, None, **spec_kwargs)

    Y_true = spec.getY()
    Y_probs = get_yprobs(max_counts)
    limit_query(Y_probs, Y_true, limit, max_counts)

if __name__ == '__main__':
    main()
