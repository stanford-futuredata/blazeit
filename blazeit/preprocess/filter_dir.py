import argparse
import functools
import glob
import multiprocessing as mp
import os
import subprocess
import tempfile

from blazeit.data.video_data import get_video_data

def f(fname, seqnms_dir=None, filtered_dir=None):
    print('Processing:', fname)
    csv_in_fname = os.path.join(seqnms_dir, fname)
    csv_out_fname = os.path.join(filtered_dir, fname)
    process_csv(csv_in_fname, csv_out_fname, base_name)
    print('Finished:', fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--base_dir', default='/lfs/1/ddkang/blazeit/data/')
    parser.add_argument('--type', default='fgfa')
    args = parser.parse_args()

    if args.type == 'fgfa':
        min_conf = 0.2
    elif args.type == 'mask':
        min_conf = 0.8
    else:
        raise NotImplementedError

    base_name = args.base_name
    base_dir = args.base_dir

    seqnms_dir = os.path.join(base_dir, 'seqnms', base_name)
    filtered_dir = os.path.join(base_dir, 'filtered', base_name)

    '''proc = functools.partial(f, seqnms_dir=seqnms_dir, filtered_dir=filtered_dir)
    pool = mp.Pool()
    pool.map(proc, sorted(os.listdir(seqnms_dir)))'''

    video_data = get_video_data(base_name)
    _, json_fname = tempfile.mkstemp()
    video_data.serialize(json_fname)

    for fname in sorted(os.listdir(seqnms_dir)):
        print('Processing:', fname)
        csv_in_fname = os.path.join(seqnms_dir, fname)
        csv_out_fname = os.path.join(filtered_dir, fname)
        ret = subprocess.call(['./filter_cc', str(min_conf), json_fname, csv_in_fname, csv_out_fname])
        if ret != 0:
            print(ret)
            raise RuntimeError('Something very bad happened')
