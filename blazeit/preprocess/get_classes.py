import argparse
import os
import pandas as pd
import feather

def filter_df_objects(df, classes, OBJECTS):
    cname_to_obj = dict(zip(range(len(classes)), classes))

    df['cname'] = df['cname'].apply(lambda x: cname_to_obj[x])
    df.rename(columns={'cname': 'object_name', 'conf': 'confidence'}, inplace=True)

    df = df[df['object_name'].isin(OBJECTS)]

    return df

def feather_to_csv(classes, feather_in_fname, csv_out_fname, OBJECTS):
    if isinstance(classes, str):
        if classes == 'ivid':
            classes = ['__background__','airplane', 'antelope', 'bear', 'bicycle',
                       'bird', 'bus', 'car', 'cattle',
                       'dog', 'domestic_cat', 'elephant', 'fox',
                       'giant_panda', 'hamster', 'horse', 'lion',
                       'lizard', 'monkey', 'motorcycle', 'rabbit',
                       'red_panda', 'sheep', 'snake', 'squirrel',
                       'tiger', 'train', 'turtle', 'watercraft',
                       'whale', 'zebra']
        elif classes == 'coco':
            classes = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        else:
            raise NotImplementedError
    elif isinstance(classes, list):
        pass
    else:
        raise NotImplementedError

    df = feather.read_dataframe(feather_in_fname)
    df = filter_df_objects(df, classes, OBJECTS)
    df.to_csv(csv_out_fname, index=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_type', required=True) # ivid or coco
    parser.add_argument('--classes', required=True)
    parser.add_argument('--feather_in_fname', required=True)
    parser.add_argument('--csv_out_fname', required=True)
    args = parser.parse_args()

    OBJECTS = set(args.classes.split(','))
    feather_in_fname = args.feather_in_fname
    csv_out_fname = args.csv_out_fname

    feather_to_csv(args.det_type, feather_in_fname, csv_out_fname, OBJECTS)

if __name__ == '__main__':
    main()
