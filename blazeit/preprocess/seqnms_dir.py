import argparse
import glob
import os
import subprocess

from blazeit.data.video_data import get_video_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--base_dir', default='/lfs/1/ddkang/blazeit/data/')
    prefilter_parser = parser.add_mutually_exclusive_group(required=False)
    prefilter_parser.add_argument('--prefilter', dest='prefilter', action='store_true')
    prefilter_parser.add_argument('--no-prefilter', dest='prefilter', action='store_false')
    parser.set_defaults(prefilter=False)
    args = parser.parse_args()

    base_name = args.base_name
    base_dir = args.base_dir
    vid_data = get_video_data(base_name)
    classes = set(vid_data.classes)

    csv_dir = os.path.join(base_dir, 'prefilter' if args.prefilter else 'csv', base_name)
    seqnms_dir = os.path.join(base_dir, 'seqnms', base_name)

    for fname in sorted(os.listdir(csv_dir)):
        print('Processing:', fname)
        csv_in_fname = os.path.join(csv_dir, fname)
        csv_out_fname = os.path.join(seqnms_dir, fname)
        ret = subprocess.call(['./seqnms', csv_in_fname, csv_out_fname])
        if ret != 0:
            print(ret)
            raise RuntimeError('Something very bad happened')
