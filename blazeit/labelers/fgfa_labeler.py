import os
import time

import numpy as np

from .base_labeler import BaseLabeler

from fgfa.fgfa_rfcn.standalone import Labeler

class FGFALabeler(BaseLabeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _DATA_PATH = '/lfs/1/ddkang/blazeit/data/models/fgfa/'
        _WTS_FNAME = 'rfcn_fgfa_flownet_vid'
        _WTS_FNAME = os.path.join(_DATA_PATH, _WTS_FNAME)
        _CFG_FNAME = 'fgfa_rfcn_vid_demo.yaml'
        _CFG_FNAME = os.path.join(_DATA_PATH, _CFG_FNAME)

        # We need the first frame for fgfa
        _, sample_frame = self.vin.read()
        self._labeler = Labeler(_CFG_FNAME, _WTS_FNAME, sample_frame)

    def label_frame(self, frameid):
        start = time.time()
        self.vin.set(1, frameid - 1)
        frames = []
        for i in range(3):
            ret, frame = self.vin.read()
            if not ret:
                raise RuntimeError('frameid %d invalid' % frameid)
            frames.append(frame)
        self.decode_time += time.time() - start
        rows = self._label_frame(frames, frameid)
        return self._filter_rows(rows)

    # FIXME: needs work
    def _label_frame(self, frames, frameid):
        return self._labeler.label_frame(frames, frameid)
