import logging
import os
import tempfile

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.utils.data
from collections import defaultdict
from scipy.special import expit

from blazeit.data.video_data import get_video_data

from .pytorch_utils import *
from . import resnet_simple

logger = logging.getLogger(__name__)

class BaseSpecializer(object):
    def __init__(self, base_name, video_fname, csv_fname, objects, normalize,
                 model, model_dump_fname, **kwargs):
        self.base_name = base_name
        self.video_fname = video_fname
        self.csv_fname = csv_fname
        self.objects = objects
        self.normalize = normalize

        self.model = model
        self.model_dump_fname = model_dump_fname

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.vid_data = get_video_data(base_name)

    def getX(self, dump=True, data_path='/lfs/1/ddkang/blazeit/data/'):
        if self.video_fname[-4:] == '.npy':
            X = np.load(self.video_fname, mmap_mode='r')
        elif self.video_fname[-4:] == '.mp4' or self.video_fname[-5:] == '.webm':
            # TODO: make swag parallel decode
            raise NotImplementedError
            _, json_fname = tempfile.mkstemp()
            self.vid_data.serialize(json_fname)
            bd = SingleVideoDecoder(json_fname, self.video_fname)
            X = bd.read()
            logger.info('Finished decoding video')
            try:
                os.remove(json_fname)
            except:
                pass
            logger.info('Removed temp json, starting normalizing')
            if self.normalize:
                X[...,:] -= [0.485, 0.456, 0.406]
                X[...,:] /= [0.229, 0.224, 0.225]
                dir_name = 'resol-65'
            else:
                dir_name = 'resol-65-avg'
            logger.info('Finished normalizing')
            npy_dir = os.path.join(data_path, dir_name, self.base_name)
            out_fname = os.path.splitext(os.path.basename(self.video_fname))[0]
            out_fname = os.path.join(npy_dir, out_fname + '.npy')
            logger.info('Writing video to %s', out_fname)
            np.save(out_fname, X)
            logger.info('Wrote out file')
        else:
            logger.critical('File format of %s not supported', self.video_fname)
            raise NotImplementedError
        return X

    def get_frame_to_rows(self):
        df = pd.read_csv(self.csv_fname)
        df = df[df['object_name'].isin(self.objects)]

        groups = defaultdict(list)
        for row in df.itertuples():
            groups[row.frame].append(row)
        return groups

    def getXY(self):
        X = self.getX()
        Y = self.getY(nb_frames=len(X))
        return X, Y

    # TODO: other methods of splitting data?
    def get_train_val(self, nb_train=100000, nb_val=20000, selection='balanced', XY=None):
        def random_inds(X, Y):
            nb_total = nb_train + nb_val
            inds = np.random.permutation(len(X))[0:nb_total]
            return inds[0:nb_train], inds[nb_train:]

        def proportional_inds(X, Y):
            labels = np.unique(Y, axis=0)
            ind_map = defaultdict(list)
            for i, label in enumerate(Y):
                try:
                    ind_map[tuple(label)].append(i)
                except:
                    ind_map[label].append(i)
            fractions = defaultdict(float)
            for label in ind_map:
                fractions[label] = len(ind_map[label]) / float(len(Y))

            train_inds = []
            val_inds = []
            for label in ind_map:
                total_samples = int(fractions[label] * (nb_train + nb_val))
                train_samples = int(fractions[label] * nb_train)
                inds = np.random.permutation(ind_map[label])[0:total_samples]
                train_inds += list(inds[0:train_samples])
                val_inds += list(inds[train_samples:])
            return train_inds, val_inds

        def balanced_inds(X, Y):
            labels = np.unique(Y, axis=0)
            ind_map = defaultdict(list)
            for i, label in enumerate(Y):
                try:
                    ind_map[tuple(label)].append(i)
                except:
                    ind_map[label].append(i)
            train_samples_per_class = nb_train // len(ind_map)
            val_samples_per_class = nb_val // len(ind_map)
            total_samples_per_class = train_samples_per_class + val_samples_per_class

            train_inds = []
            val_inds = []
            for label in ind_map:
                inds = np.random.permutation(ind_map[label])[0:total_samples_per_class]
                train_inds += list(inds[0:train_samples_per_class])
                val_inds += list(inds[train_samples_per_class:])
            return train_inds, val_inds

        def split(Z, train_inds, val_inds):
            return Z[train_inds], Z[val_inds]

        X, Y = self.getXY() if XY is None else XY
        X = X[0:len(Y)]
        Y = Y[0:len(X)]
        logger.info('Running %s' % selection)
        if selection == 'balanced':
            train_inds, val_inds = balanced_inds(X, Y)
        elif selection == 'random':
            train_inds, val_inds = random_inds(X, Y)
        elif selection == 'proportional':
            train_inds, val_inds = proportional_inds(X, Y)
        else:
            raise NotImplementedError
        X_train, X_val = split(X, train_inds, val_inds)
        Y_train, Y_val = split(Y, train_inds, val_inds)
        return (X_train, Y_train), (X_val, Y_val)

    def load_data(self, batch_size=16, num_workers=4, data=None, **kwargs):
        if data is None:
            t1, t2 = self.get_train_val(**kwargs)
            X_train, Y_train = t1
            X_val, Y_val = t2
        else:
            X_train, Y_train, X_val, Y_val = data
        X_train = X_train.transpose((0, 3, 1, 2))
        X_val = X_val.transpose((0, 3, 1, 2))

        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val

        self.train_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(Y_train))
        self.val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(Y_val))
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True, pin_memory=False)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True, pin_memory=False)

    # train_last: all | last_section | last_block
    def _train(self, criterion, metric='topk', train_last='all',
               epochs=None, lrs=None, silent=False):
        if epochs is None:
            epochs = [1, 1] if self.normalize else [0, 2]
        if lrs is None:
            lrs = [0.1, 0.01] if self.normalize else [0., 0.1]
        def run_epoch(lr=0.01, nb_epochs=1):
            if nb_epochs == 0:
                return
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            optimizer = torch.optim.SGD(params, lr,
                                        momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            acc, loss = trainer(self.model, criterion, optimizer, scheduler,
                                (self.train_loader, self.val_loader),
                                weight_best_name=self.model_dump_fname,
                                nb_epochs=nb_epochs,
                                scheduler_arg='loss', metric=metric,
                                silent=silent)
            logger.debug('Acc, loss: %f, %f', acc, loss)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc.parameters():
            p.requires_grad = True
        if self.normalize:
            run_epoch(lrs[0], epochs[0])

        if train_last == 'all':
            params = self.model.parameters()
        elif train_last == 'last_section':
            params = self.model.sections[-1].parameters()
        elif train_last == 'last_block':
            params = self.model.sections[-1][-1].parameters()
        else:
            raise NotImplementedError
        for p in params:
            p.requires_grad = True
        if self.normalize:
            run_epoch(lrs[1], epochs[1])
        else:
            run_epoch(lrs[1], epochs[1])

    def _eval(self, X, eval_batch_size=2048):
        if X.shape[1] != 3:
            X = X.transpose((0, 3, 1, 2))
        self.model.eval()
        self.model.cuda()
        Y_tmp = []
        count = 0
        for batch in np.array_split(X, len(X) // eval_batch_size):
            # Transpose makes the array non-continuous??
            inp = torch.from_numpy(batch.copy()).cuda(non_blocking=True)
            inp_var = torch.autograd.Variable(inp, requires_grad=False)
            with torch.no_grad():
                output = self.model(inp_var)
            tmp = output.cpu().data.numpy()
            Y_tmp.append(tmp)
            count += 1
        Y_pred = np.vstack(Y_tmp)
        return Y_pred

    def getY(self, nb_frames=None):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def eval(self):
        raise NotImplementedError
    def metrics(self, Y_prob, Y_true):
        raise NotImplementedError
