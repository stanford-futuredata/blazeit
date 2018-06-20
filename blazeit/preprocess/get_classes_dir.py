import argparse
import functools
import multiprocessing as mp
import os
import subprocess

import feather
import pandas as pd

from blazeit.data.video_data import get_video_data
from get_classes import feather_to_csv

def f(fname, classes=None, feather_dir=None, csv_dir=None, OBJECTS=None, feather_to_csv=None):
    print('Processing:', fname)
    feather_in_fname = os.path.join(feather_dir, fname)
    base_fname = os.path.splitext(fname)[0]
    csv_out_fname = os.path.join(csv_dir, base_fname + '.csv')
    feather_to_csv(classes, feather_in_fname, csv_out_fname, OBJECTS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--base_dir', default='/lfs/1/ddkang/blazeit/data/')
    parser.add_argument('--type', default='ivid')
    args = parser.parse_args()

    base_name = args.base_name
    base_dir = args.base_dir
    vid_data = get_video_data(base_name)

    OBJECTS = vid_data.classes

    feather_dir = os.path.join(base_dir, 'feather', base_name)
    csv_dir = os.path.join(base_dir, 'csv', base_name)

    proc = functools.partial(f, classes=args.type, feather_dir=feather_dir,
                             csv_dir=csv_dir, OBJECTS=OBJECTS,
                             feather_to_csv=feather_to_csv)
    pool = mp.Pool()
    pool.map(proc, sorted(os.listdir(feather_dir)))

    '''for fname in sorted(os.listdir(feather_dir)):
        print('Processing:', fname)
        feather_in_fname = os.path.join(feather_dir, fname)
        base_fname = os.path.splitext(fname)[0]
        csv_out_fname = os.path.join(csv_dir, base_fname + '.csv')
        feather_to_csv(feather_in_fname, csv_out_fname, OBJECTS)'''

if __name__ == '__main__':
    main()
