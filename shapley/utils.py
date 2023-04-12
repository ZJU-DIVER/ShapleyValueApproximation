import math
import os
import time
from pathlib import Path
from itertools import chain, combinations
import numpy as np
import pandas as pd
from typing import Iterator
from matplotlib import pyplot as plt


def power_set(iterable) -> Iterator:
    """Generate the power set of the all elements of an iterable obj.
    """
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(1,
                                          len(s) + 1))


def time_function(f, *args) -> float:
    """Call a function f with args and return the time (in seconds)
    that it took to execute.
    """

    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def get_ele_idxs(ele, ele_list) -> list:
    """Return all index of a specific element in the element list
    """
    idx = -1
    if not isinstance(ele_list, list):
        ele_list = list(ele_list)
    n = ele_list.count(ele)
    idxs = [0] * n
    for i in range(n):
        idx = ele_list.index(ele, idx + 1, len(ele_list))
        idxs[i] = idx
    return idxs


def split_num(m_list, num) -> np.ndarray:
    """Split num
    """
    perm_arr_list = None
    local_state = np.random.RandomState(None)
    for m in m_list:
        assert m >= 0
        if m != 0:
            m = int(m)
            quotient = int(m / num)
            remainder = m % num
            if remainder > 0:
                perm_arr = [[quotient]] * (num - remainder) + [[quotient + 1]
                                                               ] * remainder
                local_state.shuffle(perm_arr)
            else:
                perm_arr = [[quotient]] * num
        else:
            perm_arr = [[0]] * num

        if perm_arr_list is None:
            perm_arr_list = perm_arr
        else:
            perm_arr_list = np.concatenate((perm_arr_list, perm_arr), axis=-1)

    return np.asarray(perm_arr_list)


def split_permutation(m, num) -> np.ndarray:
    """Split permutation
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num

    perm_arr = []
    r = []
    for i in range(m):
        r.append(i)
        if (remainder > 0
                and len(r) == quotient + 1) or (remainder <= 0
                                                and len(r) == quotient):
            remainder -= 1
            perm_arr.append(r)
            r = []
    return perm_arr


def split_permutation_num(m, num) -> np.ndarray:
    """Split permutation num
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num
    if remainder > 0:
        perm_arr = [quotient] * (num - remainder) + [quotient + 1] * remainder
    else:
        perm_arr = [quotient] * num
    return np.asarray(perm_arr)


def split_permutations_t_list(permutations, t_list, num) -> list:
    """Split permutation num
    """
    m = len(permutations)
    m_list = split_permutation_num(m, num)
    res = list()
    for local_m in m_list:
        res.append([permutations[:local_m], t_list[:local_m]])
        permutations = permutations[local_m:]
        t_list = t_list[local_m:]
    return res


def save_npy(file_name, arr):
    check_folder('res')
    if (isinstance(arr, np.ndarray)):
        np.save(Path.cwd().joinpath('res').joinpath(file_name), arr)


def load_npy(file_name):
    check_folder('res')
    arr = np.load(Path.cwd().joinpath('res').joinpath(file_name))
    if (isinstance(arr, np.ndarray)):
        return arr


def check_folder(floder_name):
    """Check whether the folder exists
    """
    if Path(floder_name).exists() == False:
        prefix = Path.cwd()
        Path.mkdir(prefix.joinpath(floder_name))


def normalize(list1, list2):
    """Normalize list1 to list2
    """
    coef = np.sum(list1) / np.sum(list2)
    return coef * list2


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(1,
                                          len(s) + 1))


def preprocess_data_sub(train_file_name, valid_file_name):
    """Process the training dataset and the valid dataset
    """
    train_df = pd.read_csv(
        os.path.abspath('.') + '/data_files/' + train_file_name)
    valid_df = pd.read_csv(
        os.path.abspath('.') + '/data_files/' + valid_file_name)

    # train_df = pd.read_csv(r'../data_files/' + train_file_name)
    # valid_df = pd.read_csv(r'../data_files/' + valid_file_name)

    columns_name = train_df.columns

    x_train = train_df.drop(columns=['Y']).values
    x_valid = valid_df.drop(columns=['Y']).values
    y_train = train_df.Y.values
    y_valid = valid_df.Y.values

    return x_train, y_train, x_valid, y_valid, columns_name


def show(baseline: list, others: list, labels: list, title: str):
    xData = [i for i in range(baseline.__len__())]
    lines = [list(baseline)]
    for l in others:
        lines.append(list(l))
    for i in range(len(lines)):
        plt.scatter(xData, lines[i], label=labels[i])

    plt.title(title)
    plt.legend()
    plt.show()
