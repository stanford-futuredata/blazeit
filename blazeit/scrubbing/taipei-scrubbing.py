import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from blazeit.data.generate_fnames import get_csv_fname, get_video_fname
from blazeit.specializers.specializers import MulticlassCountSpecializer
from blazeit.specializers import trn
from blazeit.labelers.base_labeler import MockLabeler
from blazeit.labelers.detectron_labeler import DetectronLabeler
from blazeit.labelers.fgfa_labeler import FGFALabeler

# kwargs: max_counts
class PytorchMultiCount(MulticlassCountSpecializer):
    def eval(self, X):
        Y_prob_all = super().eval(X)
        Y_probs = []
        start = 0
        for i, max_count in enumerate(self.max_counts):
            class_num = max_count + 1
            Y_prob = Y_prob_all[:, start:start + class_num]
            start += class_num
            Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
            Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
            Y_probs.append(Y_prob)

        # df = pd.DataFrame(Y_prob_all)
        # df.to_csv('scrubbing.csv', index=False, header=False)

        return Y_probs

class MultiCountLabeler(object):
    def __init__(self, objects, counts, labeler):
        self.objects = objects
        self.counts = counts
        self.labeler = labeler
    def label_frame(self, frameid):
        def count(obj_name, rows):
            return sum(map(lambda x: x.object_name == obj_name, rows))
        rows = self.labeler.label_frame(frameid)
        for i, obj in enumerate(self.objects):
            if count(obj, rows) < self.counts[i]:
                return False
        return True

def limit_query(Y_probs, labeler, limit, correct_counts, distance=300):
    def is_correct(frame):
        return labeler.label_frame(frame)

    Y_prob = []
    for i in range(len(Y_probs[0])):
        tmp = [Y_probs[j][i] for j in range(len(correct_counts))]
        Y_prob.append(tmp)
    Y_sort = sorted(enumerate(Y_prob),
                    key=lambda x: sum([x[1][j][-1] for j in range(len(correct_counts))]),
                    # key=lambda x: x[1][0][-1] + x[1][1][-1],
                    reverse=True)

    blacklist = set()
    correct = []
    nb_calls = 0
    for frame, probs in Y_sort:
        if frame in blacklist:
            continue
        nb_calls += 1
        if nb_calls % 100 == 0:
            print('Call #%d, %d correct' % (nb_calls, len(correct)))
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
    parser.add_argument('--labeler', required=True)
    parser.add_argument('--limit', required=True, default=10, type=int)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--train_date', required=True)
    parser.add_argument('--thresh_date', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--objects', default='', required=True)
    parser.add_argument('--counts', default='', required=True)
    vid_parser = parser.add_mutually_exclusive_group(required=True)
    vid_parser.add_argument('--load_video', dest='load_video', action='store_true')
    vid_parser.add_argument('--no-load_video', dest='load_video', action='store_false')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    TRAIN_DATE = args.train_date
    THRESH_DATE = args.thresh_date
    TEST_DATE = args.test_date
    LIMIT = args.limit
    trn_name = 'trn10'
    base_name = args.base_name
    max_counts = list(map(int, args.counts.split(',')))
    objects = args.objects.split(',')
    if len(objects) != len(max_counts):
        raise RuntimeError('len(objects) must equal len(counts)')
    if sorted(objects) != objects:
        raise RuntimeError('objects must be in sorted order')
    normalize = True
    LABELER = args.labeler
    assert LABELER in ['mock-detectron', 'mock-fgfa', 'detectron', 'fgfa']

    # max_counts = [1, 5]
    spec_kwargs = {'max_counts': max_counts}
    nb_classes = sum(max_counts) + len(max_counts)

    # setup
    base_model = trn.create_trn(trn_name, nb_classes, weights='imagenet')
    model_dump_fname = os.path.join(
            DATA_PATH, 'models', base_name, '%s-%s-%s.t7' % (base_name, TRAIN_DATE, trn_name))

    load_times = []
    total_time = time.time()
    # train
    csv_fname = get_csv_fname(DATA_PATH, base_name, TRAIN_DATE)
    video_fname = get_video_fname(DATA_PATH, base_name, TRAIN_DATE, load_video=args.load_video)
    spec = PytorchMultiCount(base_name, video_fname, csv_fname, objects, normalize,
                             base_model, model_dump_fname, **spec_kwargs)

    start = time.time()
    spec.load_data()
    load_times.append(time.time() - start)
    train_time = time.time()
    spec.train()
    train_time = time.time() - train_time
    # spec.model.load_state_dict(torch.load(model_dump_fname)['state_dict'])

    # test
    csv_fname = get_csv_fname(DATA_PATH, base_name, TEST_DATE)
    video_fname = get_video_fname(DATA_PATH, base_name, TEST_DATE, load_video=args.load_video)
    spec = PytorchMultiCount(base_name, video_fname, csv_fname, objects, normalize,
                             spec.model, model_dump_fname, **spec_kwargs)

    start = time.time()
    X = spec.getX()
    load_times.append(time.time() - start)
    eval_time = time.time()
    Y_prob = spec.eval(X)
    eval_time = time.time() - eval_time

    video_fname = get_video_fname(DATA_PATH, base_name, TEST_DATE, load_video=True)
    print(video_fname)
    print(csv_fname)
    start = time.time()
    if LABELER == 'mock-detectron':
        full_labeler = MockLabeler(csv_fname, 0.05, base_name, video_fname, 0.8)
        det_time = 1/3.
    elif LABELER == 'mock-fgfa':
        full_labeler = MockLabeler(csv_fname, 0.05, base_name, video_fname, 0.2)
        det_time = 1/0.24
    elif LABELER == 'detectron':
        full_labeler = DetectronLabeler(0.05, base_name, video_fname, 0.8)
        det_time = 0.
    elif LABELER == 'fgfa':
        full_labeler = FGFALabeler(0.05, base_name, video_fname, 0.2)
        det_time = 0.
    else:
        raise NotImplementedError
    labeler = MultiCountLabeler(objects, max_counts, full_labeler)
    load_times.append(time.time() - start)

    frames, nb_calls = limit_query(Y_prob, labeler, limit=LIMIT, correct_counts=max_counts)
    total_time = time.time() - total_time
    load_times.append(labeler.labeler.decode_time)
    print('Load times:', sum(load_times), load_times)
    total_time -= sum(load_times)
    print('Times:', total_time, total_time + nb_calls * det_time)
    print('Train time:', train_time)
    print('Eval time:', eval_time)
    print(frames)


if __name__ == '__main__':
    main()
