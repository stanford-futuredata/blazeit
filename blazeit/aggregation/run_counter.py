import argparse
import logging
from blazeit.aggregation.counter import train_and_test

def main():
    logging.basicConfig(level=logging.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/lfs/1/ddkang/blazeit/data/')
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--train_date', required=True)
    parser.add_argument('--thresh_date', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--objects', required=True)
    parser.add_argument('--tiny_model', default='trn10')
    parser.add_argument('--nb_classes', default=-1, )
    vid_parser = parser.add_mutually_exclusive_group(required=True)
    vid_parser.add_argument('--load_video', dest='load_video', action='store_true')
    vid_parser.add_argument('--no-load_video', dest='load_video', action='store_false')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    TRAIN_DATE = args.train_date
    THRESH_DATE = args.thresh_date
    TEST_DATE = args.test_date
    tiny_name = args.tiny_model
    base_name = args.base_name
    objects = set(args.objects.split(','))

    nb_classes = args.nb_classes

    train_and_test(DATA_PATH, TRAIN_DATE, THRESH_DATE, TEST_DATE,
                   base_name, objects,
                   tiny_name=tiny_name,
                   load_video=args.load_video)

if __name__ == '__main__':
    main()
