import time
from collections import defaultdict

import numpy as np
import pandas as pd

import swag

from blazeit.data.video_data import get_video_data

class BaseLabeler(object):
    def __init__(self, record_thresh, base_name, vid_fname, return_thresh=None):
        self.vid_fname = vid_fname
        self.base_name = base_name
        self.record_thresh = record_thresh
        self.return_thresh = return_thresh
        self.vin = swag.VideoCapture(self.vid_fname)
        self.vid_data = get_video_data(self.base_name)
        self.decode_time = 0.

    # Not sure if this default implementation is the right thing
    def _filter_rows(self, rows):
        if self.return_thresh is not None:
            rows = filter(lambda x: x.confidence > self.return_thresh, rows)
        rows = filter(lambda x: self.vid_data.process_row(x) != None, rows)
        return list(rows)

    def _label_frame(self, frame, frameid):
        raise NotImplementedError

    def label_frame(self, frameid):
        start = time.time()
        self.vin.set(1, frameid)
        ret, frame = self.vin.read()
        if not ret:
            raise RuntimeError('frameid %d invalid' % frameid)
        self.decode_time += time.time() - start
        rows = self._label_frame(frame, frameid)
        return self._filter_rows(rows)

class MockLabeler(BaseLabeler):
    def __init__(self, csv_fname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        df = pd.read_csv(csv_fname)
        self.frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            self.frame_to_rows[row.frame].append(row)

    def label_frame(self, frameid):
        return self.frame_to_rows[frameid]
