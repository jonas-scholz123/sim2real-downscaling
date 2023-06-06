import os


def ensure_dir_exists(fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)


def ensure_exists(dirpath):
    os.makedirs(dirpath, exist_ok=True)
