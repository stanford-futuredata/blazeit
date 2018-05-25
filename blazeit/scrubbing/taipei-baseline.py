import argparse
import os
from collections import defaultdict

import pandas as pd

def filt(rows, objects, counts):
    def count(object_name):
        return sum(map(lambda x: x.object_name == object_name, rows))
    for i, obj in enumerate(objects):
        c = count(obj)
        if c < counts[i]:
            return False
    return True

def run_baseline(baseline_name, correct_frames, look_frames, limit, distance=300):
    blacklist = set()
    nb_calls = 0
    nb_correct = 0
    correct = []
    for frame in look_frames:
        if frame in blacklist:
            continue
        nb_calls += 1
        if frame in correct_frames:
            correct.append(frame)
            nb_correct += 1
            for i in range(-distance, distance + 1):
                blacklist.add(i + frame)
        if nb_correct >= limit:
            break
    print('Run baseline %s' % baseline_name)
    print('Called %d times, found %d instances' % (nb_calls, nb_correct))
    det_time = 0.22
    print('Total time:', nb_calls * det_time)
    # print('%d,%d' % (limit, nb_calls))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/lfs/1/ddkang/blazeit/data')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--objects', default='', required=True)
    parser.add_argument('--counts', default='', required=True)
    args = parser.parse_args()

    DATA_PATH = args.data_path
    TEST_DATE = args.test_date
    base_name = args.base_name
    LIMIT = args.limit
    counts = list(map(int, args.counts.split(',')))
    objects = args.objects.split(',')
    if len(objects) != len(counts):
        raise RuntimeError('len(objects) must equal len(counts)')
    if sorted(objects) != objects:
        raise RuntimeError('objects must be in sorted order')

    DATE = TEST_DATE
    csv_fname = os.path.join(DATA_PATH, 'filtered', base_name, '%s-%s.csv' % (base_name, DATE))
    df = pd.read_csv(csv_fname)

    frame_to_rows = defaultdict(list)
    for row in df.itertuples():
        frame_to_rows[row.frame].append(row)

    correct_frames = filter(lambda x: filt(frame_to_rows[x], objects, counts),
                            frame_to_rows.keys())
    correct_frames = sorted(list(correct_frames))
    correct_frames = set(correct_frames)

    nb_frames = max(df['frame'])
    run_baseline('naive', correct_frames, list(range(nb_frames)), limit=LIMIT)

    noscope_frames = []
    for frame in frame_to_rows:
        tmp = set(map(lambda x: x.object_name, frame_to_rows[frame]))
        append = True
        for obj in objects:
            if obj not in tmp:
                append = False
                break
        if append:
            noscope_frames.append(frame)
    run_baseline('noscope', correct_frames, noscope_frames, limit=LIMIT)

if __name__ == '__main__':
    main()
