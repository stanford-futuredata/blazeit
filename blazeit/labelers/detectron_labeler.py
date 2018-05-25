import os

import numpy as np

from .base_labeler import BaseLabeler
from detectron.tools.standalone import Labeler


class DetectronLabeler(BaseLabeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _DATA_PATH = '/lfs/1/ddkang/blazeit/data/models/detectron/'
        _WTS_FNAME = 'mask_rcnn-X-152-32x8d-FPN-IN5k.pkl'
        _WTS_FNAME = os.path.join(_DATA_PATH, _WTS_FNAME)
        _CFG_FNAME = 'e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml'
        _CFG_FNAME = os.path.join(_DATA_PATH, _CFG_FNAME)

        self._labeler = Labeler(_CFG_FNAME, _WTS_FNAME)

    def _label_frame(self, frame, frameid):
        return self._labeler.label_frame(frame, frameid)
