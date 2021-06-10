import functools
import logging

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from collections import defaultdict

from .base_specializer import BaseSpecializer

logger = logging.getLogger(__name__)

class BCEWeightLL(torch.nn.Module):
    def __init__(self, weights):
        super(BCEWeightLL, self).__init__()
        self.bcell = torch.nn.BCEWithLogitsLoss()
        self.weights = weights
        assert len(self.weights) == 2
    def forward(self, inp, target):
        cur_weights = np.where(target.data.cpu().numpy(), self.weights[0], self.weights[1])
        self.bcell.weights = cur_weights
        return self.bcell(inp, target)

class BinarySpecializer(BaseSpecializer):
    def getY(self, nb_frames=None):
        assert len(self.objects) == 1
        frame_to_rows = self.get_frame_to_rows()
        if not nb_frames:
            nb_frames = max(frame_to_rows) + 1

        Y = np.zeros(nb_frames)
        for frame in frame_to_rows:
            Y[frame] = 1
        return Y

    def train(self, **kwargs):
        # criterion = nn.BCEWithLogitsLoss().cuda()
        from sklearn.utils import class_weight
        weights = class_weight.compute_class_weight('balanced', [0,1], self.Y_train)
        criterion = BCEWeightLL(weights)
        self._train(criterion, metric='bce', **kwargs)

    def eval(self, X, **kwargs):
        return self._eval(X, **kwargs).ravel()

    def find_threshold(self, Y_prob, Y_true, fnr=0.02, fpr=0.02):
        total_num_pos = np.sum(Y_true)
        total_num_neg = len(Y_true) - total_num_pos
        cutoff_num_pos = total_num_pos * (1 - fnr)
        cutoff_num_neg = total_num_neg * (1 - fpr)

        tmp = list(zip(Y_prob, Y_true))
        tmp.sort(reverse=True)
        npos = 0
        for i, (prob, label) in enumerate(tmp):
            npos += label
            if npos > cutoff_num_pos:
                return prob, i
        return tmp[-1][0], len(tmp)

    def find_two_sided_thresh(self, Y_prob, Y_true, fnr=0.02, fpr=0.02):
        total_num_pos = np.sum(Y_true)
        total_num_neg = len(Y_true) - total_num_pos
        cutoff_fn = total_num_pos * fnr
        cutoff_fp = total_num_neg * fpr

        tmp = list(zip(Y_prob, Y_true))
        tmp.sort()
        npos = 0
        for i, (prob, label) in enumerate(tmp):
            npos += label
            if npos > cutoff_fn:
                lo_thresh = prob
                nb_lo = i
                break

        tmp.reverse()
        nneg = 0
        for i, (prob, label) in enumerate(tmp):
            nneg += not label
            if nneg > cutoff_fp:
                hi_thresh = prob
                nb_hi = i
                break

        return (lo_thresh, nb_lo), (hi_thresh, nb_hi)

    def poison_metrics(self, Y_prob, Y_true, threshold=0.0, window=10):
        to_check = set()
        for i in range(len(Y_prob)):
            if Y_prob[i] > threshold:
                for j in range(-window, window+1):
                    to_check.add(i + j)
        Y_pred = list(map(lambda x: x in to_check, range(len(Y_prob))))
        Y_pred = np.array(Y_pred, dtype=np.float32)
        return self.metrics(Y_pred, Y_true, threshold=0.5)

    # threshold=logit(0.5)
    def metrics(self, Y_prob, Y_true, threshold=0):
        Y_pred = Y_prob > threshold
        if len(Y_true) != len(Y_pred):
            logger.warning('Y_true and Y_pred are mismatched')
        Y_true = Y_true[:len(Y_pred)]
        Y_pred = Y_pred[:len(Y_true)]
        confusion = sklearn.metrics.confusion_matrix(Y_true, Y_pred)
        # Minor smoothing to prevent division by 0 errors
        TN = float(confusion[0][0]) + 1
        FN = float(confusion[1][0]) + 1
        TP = float(confusion[1][1]) + 1
        FP = float(confusion[0][1]) + 1
        metrics = {'recall': TP / (TP + FN),
                   'specificity': TN / (FP + TN),
                   'precision': TP / (TP + FP),
                   'npv':  TN / (TN + FN),
                   'fpr': FP / (FP + TN),
                   'fdr': FP / (FP + TP),
                   'fnr': FN / (FN + TP),
                   'accuracy': (TP + TN) / (TP + FP + TN + FN),
                   'f1': (2 * TP) / (2 * TP + FP + FN),
                   'tpr': float(np.sum(Y_true)) / len(Y_true),
                   'cpr': float(np.sum(Y_pred)) / len(Y_pred)}
        return metrics

    def ind_metrics(self, Y_prob, threshold=0):
        # FIXME: hack
        # generate frames_to_ind
        frame_to_rows = self.get_frame_to_rows()
        frame_to_ind = defaultdict(set)
        all_inds = set()
        for frame in frame_to_rows:
            for row in frame_to_rows[frame]:
                frame_to_ind[frame].add(row.ind)
                all_inds.add(row.ind)
        nb_inds = len(all_inds)

        for i in range(len(Y_prob)):
            if Y_prob[i] < threshold:
                continue
            if i not in frame_to_ind:
                continue
            all_inds -= frame_to_ind[i]

        print('frac of inds covered:', 1. - float(len(all_inds)) / nb_inds)


class MulticlassBinarySpecializer(BinarySpecializer):
    def getY(self, nb_frames=None):
        frame_to_rows = self.get_frame_to_rows()
        if not nb_frames:
            nb_frames = max(frame_to_rows) + 1

        objs = sorted(list(self.objects))
        objs_to_ind = dict(zip(objs, range(len(objs))))

        Y = np.zeros((nb_frames, len(objs)))
        for frame in frame_to_rows:
            for row in frame_to_rows[frame]:
                Y[frame, objs_to_ind[row.object_name]] = 1
        return Y

    def train(self):
        criterion = nn.BCEWithLogitsLoss().cuda()
        self._train(criterion, metric='bce')

    def eval(self, X, **kwargs):
        return self._eval(X, **kwargs)

    def metrics(self, Y_prob, Y_true):
        metrics = []
        for i in range(Y_prob.shape[1]):
            m = super().metrics(Y_prob[:, i], Y_true[:, i])
            metrics.append(m)
        return metrics

    def ind_metrics(self, Y_prob):
        for i, obj in enumerate(sorted(self.objects)):
            super().ind_metrics(Y_prob[:, i], obj)


# kwargs: max_count
class CountSpecializer(BaseSpecializer):
    def getY(self, nb_frames=-1):
        assert len(self.objects) == 1
        frame_to_rows = self.get_frame_to_rows()
        nb_frames = max(max(frame_to_rows) + 1, nb_frames)

        Y = np.zeros(nb_frames, dtype=np.int64)
        for frame in frame_to_rows:
            Y[frame] = min(self.max_count, len(frame_to_rows[frame]))
        return Y

    def train(self, **kwargs):
        # This method will throw if Y_train does not contain samples of every class
        from sklearn.utils import class_weight
        weights = class_weight.compute_class_weight('balanced', range(self.max_count + 1), self.Y_train)
        weights = weights.astype(np.float32)
        weights = torch.autograd.Variable(torch.from_numpy(weights).cuda())
        criterion = nn.CrossEntropyLoss(weights)
        # criterion = nn.CrossEntropyLoss()
        self._train(criterion, metric='topk', **kwargs)

    def eval(self, X, **kwargs):
        return self._eval(X, **kwargs)

    def metrics(self, Y_prob, Y_true):
        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        Y_pred = np.argmax(Y_prob, axis=1)

        if len(Y_true) != len(Y_pred):
            logger.warning('Y_true and Y_pred are mismatched')
        Y_true = Y_true[:len(Y_pred)]
        Y_pred = Y_pred[:len(Y_true)]

        acc = sklearn.metrics.accuracy_score(Y_true, Y_pred)
        confusion = sklearn.metrics.confusion_matrix(Y_true, Y_pred)
        metrics = {
                'accuracy': acc,
                'sum_abs_error': np.sum(np.abs(Y_pred - Y_true)),
                'error': np.sum(Y_true - Y_pred),
                'confusion': confusion,
                'nb_instances': sum(Y_true),
                'nb_frames': len(Y_true)}
        return metrics

    def find_threshold(self, Y_prob, Y_true):
        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        Y_pred = np.argmax(Y_prob, axis=1)

        m1 = map(lambda x: sorted(x, reverse=True), Y_prob)
        diffs = map(lambda x: x[0] - x[1], m2)
        tmp = list(zip(diffs, Y_prob, Y_true))
        tmp.sort() # lowest diffs first

        nb_correct = sklearn.metrics.accuracy_score(Y_true, Y_pred) * len(Y_pred)
        nb_flipped = 0
        for i in range(tmp):
            diff, yp, yt = tmp[i]
            if np.argmax(yp) != yt:
                nb_flipped += 1
                if nb_flipped + nb_correct > 0.95 * len(Y_pred):
                    thresh = diff
                    break
        return diff, i

    def ind_metrics(self, Y_prob, csv_fname):
        # FIXME: hack
        # generate frames_to_ind
        object_name = list(self.objects)[0]
        df = pd.read_csv(csv_fname)
        df = df[df['object_name'] == object_name]
        frames_to_ind = defaultdict(set)
        all_inds = set()
        for row in df.itertuples():
            frames_to_ind[row.frame].add(row.ind)
            all_inds.add(row.ind)
        nb_inds = len(all_inds)

        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        Y_pred = np.argmax(Y_prob, axis=1)

        max_class = max(Y_pred)
        for i in range(len(Y_pred)):
            if Y_pred[i] != max_class:
                continue
            all_inds -= frames_to_ind[i]
        print('frac of inds covered:', 1. - float(len(all_inds)) / nb_inds)


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, class_nums, weights=None):
        super().__init__()
        self.class_nums = class_nums
        # self.weights = weights if weights else [1] * sum(class_nums)
        self.weights = None # FIXME
        self.size_average = True
        self.ignore_index = -100
        self.reduce_obs = True
    def forward(self, inp, target):
        assert not target.requires_grad
        ret = torch.from_numpy(np.zeros(1, dtype=np.float32))
        ret = torch.autograd.Variable(ret.cuda())
        start = 0
        for ind, cn in enumerate(self.class_nums):
            it = inp[:, start:start + cn]
            ret += nn.functional.cross_entropy(
                    it, target[:, ind], None, self.size_average,
                    self.ignore_index, self.reduce_obs)
            start += cn
        return ret

def multi_count_metric(output, target, class_nums):
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    accs = np.zeros(len(target[0]))

    start = 0
    for ind, cn in enumerate(class_nums):
        Y_tmp = torch.autograd.Variable(torch.from_numpy(output[:, start:start + cn]))
        Y_tmp = torch.nn.functional.softmax(Y_tmp, dim=1).data.numpy()
        Y_pred = np.argmax(Y_tmp, axis=1)
        start += cn
        accs[ind] += np.sum(Y_pred == target[:, ind])
    return [accs / len(target) * 100]


# kwargs: max_counts
class MulticlassCountSpecializer(CountSpecializer):
    def getY(self, nb_frames=None):
        frame_to_rows = self.get_frame_to_rows()
        if not nb_frames:
            nb_frames = max(frame_to_rows) + 1

        obj_to_ind = {obj: ind for ind, obj in enumerate(sorted(self.objects))}
        objs = sorted(self.objects)
        Y = np.zeros((nb_frames, len(self.max_counts)), dtype=np.int64)
        for frame in frame_to_rows:
            for ind, obj in enumerate(objs):
                Y[frame, ind] = len([x for x in frame_to_rows[frame] if x.object_name == obj])
                Y[frame, ind] = min(self.max_counts[ind], Y[frame, ind])

        return Y

    def train(self):
        self.class_nums = [x + 1 for x in self.max_counts]
        criterion = MultiCrossEntropyLoss(self.class_nums)
        metric = functools.partial(multi_count_metric, class_nums=self.class_nums)
        self._train(criterion, metric=metric)

    def eval(self, X, **kwargs):
        return self._eval(X, **kwargs)

    def metrics(self, Y_prob, Y_true):
        self.class_nums = [x + 1 for x in self.max_counts]
        start = 0
        metrics = []
        for ind, cn in enumerate(self.class_nums):
            m = super().metrics(Y_prob[:, start:start + cn], Y_true[:, ind])
            metrics.append(m)
            start += cn
        return metrics
