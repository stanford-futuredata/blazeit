import os

def get_csv_fname(data_path, base_name, date):
    file_base = '%s-%s' % (base_name, date)
    fname = os.path.join(
            data_path, 'filtered', base_name, '%s.csv' % file_base)
    return fname

def get_video_fname(data_path, base_name, date,
                    load_video=False, normalize=False):
    if normalize:
        raise NotImplementedError
    if load_video:
        vbase = 'svideo'
        fbase = date
    else:
        vbase = 'resol-65'
        fbase = '%s-%s.npy' % (base_name, date)
    fname = os.path.join(
            data_path, vbase, base_name, fbase)
    return fname
