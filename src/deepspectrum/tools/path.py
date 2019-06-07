import pathlib
from os.path import basename


def get_relative_path(path, prefix):
    filepath = pathlib.PurePath(path)
    if prefix is None:
        return basename(filepath)
    else:
        filepath = filepath.relative_to(prefix)
        return str(filepath)