import os


def ensure_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
