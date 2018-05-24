import cv2
import pyclipper
import numpy as np
import ujson as json
from collections import defaultdict, namedtuple

CropArea = namedtuple('CropArea', 'xmin ymin xmax ymax')

class VideoData(object):
    # crop -> (xmin, ymin, xmax, ymax)
    # exclude -> list of poly (list of points)
    def __init__(self, crop=None, exclude=[], classes=[]):
        self.crop = crop
        self.exclude = exclude
        self.classes = classes

    def draw_frame(self, frame, exclude_color=(255, 255, 255), crop_color=(0, 0, 0)):
        polys = np.array(list(map(np.array, self.exclude)))
        cv2.polylines(frame, polys, True,
                      color=exclude_color, thickness=2)
        if self.crop:
            tl = (int(self.crop.xmin), int(self.crop.ymin))
            br = (int(self.crop.xmax), int(self.crop.ymax))
            cv2.rectangle(frame, tl, br, crop_color, 2)
        return frame

    def process_frame(self, frame):
        for poly in self.exclude:
            poly = np.array(poly)
            frame = cv2.fillConvexPoly(frame, poly, color=(0, 0, 0))
        if self.crop:
            xmin, ymin, xmax, ymax = map(int, self.crop)
            frame = frame[ymin:ymax, xmin:xmax, :]
        return frame

    def process_row(self, row):
        if row.object_name not in self.classes:
            return None
        # Deal with excluded areas
        row_poly = [(row.xmin, row.ymin), (row.xmin, row.ymax),
                    (row.xmax, row.ymax), (row.xmax, row.ymin)]
        row_area = (row.xmax - row.xmin) * (row.ymax - row.ymin)
        for poly in self.exclude:
            pc = pyclipper.Pyclipper()
            pc.AddPath(poly, pyclipper.PT_CLIP, True)
            pc.AddPath(row_poly, pyclipper.PT_SUBJECT, True)
            inter = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD,
                               pyclipper.PFT_EVENODD)
            if len(inter) == 0:
                continue
            inter_area = pyclipper.Area(inter[0])
            if inter_area / row_area > 0.6:
                return None

        # Deal w/ crop
        crop = self.crop # lazy
        if not crop:
            return row
        if row.xmax < crop.xmin or crop.xmax < row.xmin or \
                row.ymax < crop.ymin or crop.ymax < row.ymin:
            return None
        xA = min(row.xmax, crop.xmax)
        yA = min(row.ymax, crop.ymax)
        xB = max(row.xmin, crop.xmin)
        yB = max(row.ymin, crop.ymin)
        inter_area = (xA - xB) * (yA - yB)
        row_area = (row.xmax - row.xmin) * (row.ymax - row.ymin)
        if inter_area / row_area < 0.25:
            return None
        else:
            return row

    def serialize(self, fname, indent=0):
        with open(fname, 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=indent)

def get_video_data(base_name):
    data = {}

    # jackson
    crop = CropArea(0, 540, 1750, 1080)
    exclude = [[(0, 0), (0, 780), (490, 780), (490, 0)],
               [(1115, 0), (1115, 600), (1365, 600), (1715, 500), (1715, 0)]]
    classes = ['car', 'truck']
    data['jackson-town-square'] = VideoData(crop, exclude, classes)

    # taipei-hires
    crop = CropArea(0, 150, 1050, 720)
    exclude = [[(875, 720), (1170, 455), (1170, 720)],
               [(0, 0), (1280, 0), (1280, 75), (0, 500)]]
    classes = ['bus', 'car', 'truck', 'person']
    data['taipei-hires'] = VideoData(crop, exclude, classes)

    # venice-rialto
    crop = CropArea(440, 660, 1675, 1050)
    exclude = [[(0, 1025), (480, 1025), (1175, 660), (0, 660)],
               [(1350, 1080), (1540, 750), (1920, 750), (1920, 1080)],
               [(1058, 738), (1133, 688), (1133, 648), (993, 606), (993, 738)]]
    classes = ['watercraft', 'boat']
    data['venice-rialto'] = VideoData(crop, exclude, classes)

    # venice-grand-canal
    crop = CropArea(0, 490, 1300, 935)
    exclude = [[(885, 1080), (885, 710), (1300, 635), (1920, 635), (1920, 1080)]]
    classes = ['watercraft', 'boat']
    data['venice-grand-canal'] = VideoData(crop, exclude, classes)

    # amsterdam
    crop = CropArea(575, 390, 1250, 720)
    exclude = [[(730, 610), (820, 610), (820, 500), (730, 500)]]
    classes = ['bicycle', 'car', 'person']
    data['amsterdam'] = VideoData(crop, exclude, classes)

    # archie day
    crop = CropArea(2170, 800, 3840, 2160)
    exclude = [[(3030, 1750), (3230, 1750), (3230, 2160), (3030, 2160)]]
    classes = ['bicycle', 'car', 'person']
    data['archie-day'] = VideoData(crop, exclude, classes)

    data['default'] = VideoData()

    return data[base_name]
