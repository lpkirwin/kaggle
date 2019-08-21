"""Common functions."""

import os
import time
import numpy as np
from hashlib import md5
from tqdm import tqdm
from os.path import join, abspath, pardir, dirname

project_dir = join(dirname(abspath(__file__)), pardir)
data_dir = join(project_dir, "data")

CURRENT_COMPETITION = None



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


def get_dpath(relative_path):
    """Path relative to competition data folder"""
    get_context()
    parts = relative_path.split("/")
    return join(*[data_dir, CURRENT_COMPETITION] + parts)

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


def reduce_mem_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def get_obj_ref(obj, hash_len=8):
    obj_type = repr(type(obj)).split(".")[-1][:-2]
    obj_hash = md5(repr(obj).encode("utf-8")).hexdigest()[-hash_len:]
    return obj_type + "_" + obj_hash


def save_model(est, oof_pred, te_pred):

    ref = get_obj_ref(est)

    pd.Series(oof_pred).to_pickle(join(get_dpath))

    te_pred = pd.Series(te_pred)