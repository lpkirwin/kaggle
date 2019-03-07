"""Common functions."""

import os
from os.path import join, abspath, pardir

project_dir = join(abspath(__file__), pardir, pardir)
data_dir = join(project_dir, "data")

CURRENT_COMPETITION = None

def comp_path(filename):
    get_context()
    return join(data_dir, CURRENT_COMPETITION, filename)


def raw_path(filename):
    get_context()
    return join(data_dir, CURRENT_COMPETITION, "raw", filename)


def set_context(comp_name):
    data_subdirs = os.listdir(data_dir)
    assert (
        comp_name in data_subdirs
    ), f"Competition not recognized. Current competitions are: {data_subdirs}"
    global CURRENT_COMPETITION
    CURRENT_COMPETITION = comp_name
    print("Files in data directory:")
    list_files(join(data_dir, CURRENT_COMPETITION))


def get_context():
    assert CURRENT_COMPETITION is not None, "CURRENT_COMPETITION not set!"
    return CURRENT_COMPETITION


# https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
def list_files(path):
    print("______\n")
    for root, _, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            if not f.startswith("."):
                print("{}{}".format(subindent, f))
    print("______\n")

