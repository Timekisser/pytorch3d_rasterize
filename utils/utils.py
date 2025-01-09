import os

def get_filenames(filelist, filelist_folder):
    with open(os.path.join(filelist_folder, filelist), 'r') as fid:
        lines = fid.readlines()
    filenames = [line.strip() for line in lines]
    return filenames