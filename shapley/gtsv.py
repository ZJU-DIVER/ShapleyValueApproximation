import datetime
import math
import random
from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy.special import comb
from tqdm import trange
import os
import shutil
import collections
import gc
from .utils import (split_permutation_num, split_permutation, split_num,
                    power_set)

max_time = 100000


def exact_shap_n(game, log_step):
    """Compute the exact Shapley value
    """
    n = game.n

    def add(l):
        for i in range(n - 1, -1, -1):
            if l[i] == 0:
                l[i] = 1
                break
            else:
                l[i] = 0
        return l

    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = np.math.factorial
    coalition = np.arange(n)
    for s in range(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    l = np.zeros(n)

    j = 0
    add(l)
    while np.sum(l) != 0:
        if j % log_step == 0:
            print(j)
        j += 1
        idx = []
        for i in range(n):
            if l[i] == 1:
                idx.append(i)
        u = game.eval_utility(idx)
        for i in idx:
            ext_sv[i] += coef[len(idx) - 1] * u
        for i in set(coalition) - set(idx):
            ext_sv[i] -= coef[len(idx)] * u
        add(l)

    return ext_sv


def exact_shap(game):
    """Compute the exact Shapley value
    """
    n = game.n
    # np.ones(n) * (-1 * game.null()[0] / n)  # np.zeros(n)
    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = np.math.factorial
    coalition = np.arange(n)
    for s in trange(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    sets = list(power_set(coalition))
    for idx in trange(len(sets)):
        # S = np.zeros((1, n), dtype=bool)
        # S[0][list(sets[idx])] = 1
        u = game.eval_utility(list(sets[idx]))
        for i in sets[idx]:
            ext_sv[i] += coef[len(sets[idx]) - 1] * u
        for i in set(coalition) - set(sets[idx]):
            ext_sv[i] -= coef[len(sets[idx])] * u

    return ext_sv


def mc_shap(game, m, proc_num=1) -> np.ndarray:
    """Compute the Monte Carlo Shapley value by sampling m permutations
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_mc_shap_task, game)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _mc_shap_task(game, local_m) -> np.ndarray:
    """Compute Shapley value by sampling local_m permutations
    """
    n = game.n
    local_state = np.random.RandomState(None)
    sv = np.zeros(n)
    idxs = np.arange(n)

    for _ in trange(local_m):
        local_state.shuffle(idxs)
        old_u = game.null
        for j in range(1, n + 1):
            temp_u = game.eval_utility(idxs[:j])
            contribution = temp_u - old_u
            sv[idxs[j - 1]] += contribution
            old_u = temp_u

    return sv


def mch(game, m, proc_num=1) -> np.ndarray:
    """Compute Shapley value by sampling m*n marginal contributions based on hoeffding
    """
    n = game.n
    sv = np.zeros(n)

    dir = os.path.join('res', 'temp_' + game.gt + '_' + str(game.n) +
                       'n_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    os.mkdir(dir)
    deldir = str(os.getcwd()) + '/' + dir

    mk = np.zeros(n)
    mk_list = []
    for i in range(n):
        mk_list.append(np.power(i + 1, 2 / 3))
    s = sum(mk_list)
    for i in range(n):
        mk[i] = math.ceil(m * mk_list[i] / s)

    args = split_num(mk, proc_num)
    pool = Pool()
    func = partial(_mch_task, game, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    utility = np.zeros((n, n))
    print("load utility")
    for i in trange(len(ret)):
        r = ret[i]
        utility += np.load(dir + 'utility_' + str(r) + '.npy')
    shutil.rmtree(deldir)

    for i in range(n):
        for j in range(n):
            sv[i] += utility[i][j] / mk[i]
        sv[i] /= n

    return sv


def _mch_task(game, dir, mk):
    """Compute Shapley value by sampling mk(list) marginal contributions
    """
    n = game.n
    utility = np.zeros((n, n))

    local_state = np.random.RandomState(None)
    for i in trange(n):
        idxs = []
        for _ in range(n):
            if _ is not i:
                idxs.append(_)
        for j in range(n):
            for _ in range(int(mk[j])):
                local_state.shuffle(idxs)
                if game.gt == 'voting':
                    u_1, hsw = game.eval_utility(idxs[:j], 'mcx')
                    u_2, hsw = game.eval_utility(idxs[:j] + [i], 'mcx', hsw)
                elif game.gt == 'airport':
                    u_1 = game.eval_utility(idxs[:j], 'mcx')
                    u_2 = game.eval_utility(idxs[:j] + [i], 'mcx', u_1)
                elif game.gt == 'tree':
                    u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'mcx')
                else:
                    u_1 = game.eval_utility(idxs[:j])
                    u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j] += (u_2 - u_1)

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'args_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)

    np.save(dir + 'utility_' + str(pid) + '.npy', utility)
    return pid


def mcn(game, m, proc_num=1) -> np.ndarray:
    """Compute Shapley value by sampling m marginal contributions by optimum allocation
    """
    n = game.n
    var = np.zeros((n, n))
    utility = np.zeros((n, n))

    dir = os.path.join('res', 'temp_' + game.gt + '_' + str(game.n) +
                       'n_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    os.mkdir(dir)
    deldir = str(os.getcwd()) + '/' + dir

    initial_m = max(2, int(m / (2 * n * n)))
    print("initial_m", initial_m)

    args = split_permutation(n, proc_num)
    pool = Pool()
    func = partial(_mcn_initial, game, initial_m, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    var = np.zeros((n, n))
    print("load1")
    for i in trange(len(ret)):
        r = ret[i]
        utility += np.load(dir + 'utility_sum_' + str(r) + '.npy')
        var += np.load(dir + 'var_' + str(r) + '.npy')
    shutil.rmtree(deldir)
    os.mkdir(dir)

    # allocate remaining samples
    # m -= n*n*initial_m
    # mst = np.zeros((n, n))
    # np.maximum(0,  var / np.sum(var) * m - initial_m, out=mst)

    # or allocate samples by complementing
    mst = var / np.sum(var) * m - initial_m
    for i in trange(n):
        for j in range(n):
            if mst[i][j] < 0:
                mst[i][j] = 0
    mst = mst.astype(int)

    args = [[] for i in range(proc_num)]
    for _ in trange(len(mst)):
        temp_args = split_num(mst[_], proc_num).tolist()
        for i in range(len(temp_args)):
            args[i].append(temp_args[i])

    # compute allocation with para
    '''
    args = split_permutation(n, 10)
    pool = Pool()
    func = partial(_split_num_par, mst, proc_num, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    args = np.zeros((proc_num, n, n))
    print("load split")
    for i in trange(len(ret)):
        r = ret[i]
        args += np.load(dir + 'args_' + str(r) + '.npy', allow_pickle=True)
    shutil.rmtree(deldir)
    os.mkdir(dir)
    args = args.astype(int)
    '''

    # later
    pool = Pool()
    func = partial(_mcn_task, game, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    print("load2")
    for i in trange(len(ret)):
        r = ret[i]
        utility += np.load(dir + 'utility_' + str(r) + '.npy')
    shutil.rmtree(deldir)

    sv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sv[i] += utility[i][j] / (mst[i][j] + initial_m)
        sv[i] /= n
    return sv


def _split_num_par(m_list, proc_num, dir, i_list):
    n = len(m_list)
    args = np.zeros((proc_num, n, n))

    for i in trange(len(i_list)):
        _ = i_list[i]
        temp_args = split_num(m_list[_], proc_num).tolist()
        temp_args = np.array(temp_args, ndmin=2)
        for i in range(proc_num):
            args[i][_] += temp_args[i]

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'args_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)
    np.save(dir + 'args_' + str(pid) + '.npy', args)
    return pid


def _mcn_initial(game, initial_m, dir, i_list):
    """Initial for mcn
    """
    n = game.n
    local_state = np.random.RandomState(None)
    utility_sum = np.zeros((n, n))
    var = np.zeros((n, n))

    for h in trange(len(i_list)):
        i = i_list[h]
        idxs = []
        for _ in range(n):
            if _ is not i:
                idxs.append(_)
        for j in range(n):
            temp = []
            for _ in range(initial_m):
                local_state.shuffle(idxs)
                if game.gt == 'voting':
                    u_1, hsw = game.eval_utility(idxs[:j], 'mcx')
                    u_2, hsw = game.eval_utility(idxs[:j] + [i], 'mcx', hsw)
                elif game.gt == 'airport':
                    u_1 = game.eval_utility(idxs[:j], 'mcx')
                    u_2 = game.eval_utility(idxs[:j] + [i], 'mcx', u_1)
                elif game.gt == 'tree':
                    u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'mcx')
                else:
                    u_1 = game.eval_utility(idxs[:j])
                    u_2 = game.eval_utility(idxs[:j] + [i])

                temp.append(u_2 - u_1)
                utility_sum[i][j] += u_2 - u_1
            var[i][j] = np.var(temp, ddof=1)

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'utility_sum_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)
    np.save(dir + 'utility_sum_' + str(pid) + '.npy', utility_sum)
    np.save(dir + 'var_' + str(pid) + '.npy', var)
    return pid


def _mcn_task(game, dir, mst):
    """Compute Shapley value by sampling mst(list) marginal contributions
    """
    n = game.n
    local_state = np.random.RandomState(None)
    utility = np.zeros((n, n))
    for i in trange(n):
        idxs = []
        for _ in range(n):
            if _ is not i:
                idxs.append(_)
        for j in range(n):
            for _ in range(mst[i][j]):
                local_state.shuffle(idxs)
                if game.gt == 'voting':
                    u_1, hsw = game.eval_utility(idxs[:j], 'mcx')
                    u_2, hsw = game.eval_utility(idxs[:j] + [i], 'mcx', hsw)
                elif game.gt == 'airport':
                    u_1 = game.eval_utility(idxs[:j], 'mcx')
                    u_2 = game.eval_utility(idxs[:j] + [i], 'mcx', u_1)
                elif game.gt == 'tree':
                    u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'mcx')
                else:
                    u_1 = game.eval_utility(idxs[:j])
                    u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j] += (u_2 - u_1)

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'utility_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)

    np.save(dir + 'utility_' + str(pid) + '.npy', utility)
    return pid


def cc_shap(game, m, proc_num=1) -> np.ndarray:
    """Compute Shapley value by sampling m complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = game.n

    dir = os.path.join('res', 'temp_' + game.gt + '_' + str(game.n) +
                       'n_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    deldir = str(os.getcwd()) + '/' + dir
    os.mkdir(dir)

    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_cc_shap_task, game, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    sv = np.zeros(n)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    for r in ret:
        utility += np.load(dir + 'utility_' + str(r) + '.npy')
        count += np.load(dir + 'count_' + str(r) + '.npy')

    shutil.rmtree(deldir)

    for i in range(n + 1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv


def _cc_shap_task(game, dir, local_m):
    """Compute Shapley value by sampling local_m complementary contributions
    """
    n = game.n
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(local_m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        if game.gt == 'voting':
            u_1 = game.eval_utility(idxs[:j])
            u_2 = 1 - u_1
        elif game.gt == 'tree':
            u_1, u_2 = game.eval_utility(idxs[:j], 'ccx')
        else:
            u_1 = game.eval_utility(idxs[:j])
            u_2 = game.eval_utility(idxs[j:])

        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'utility_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)
    np.save(dir + 'utility_' + str(pid) + '.npy', utility)
    np.save(dir + 'count_' + str(pid) + '.npy', count)
    return pid


def ccshap_neyman(game, initial_m, m, proc_num=1, initial_proc=True, type=1) -> np.ndarray:
    """Compute Shapley value by sampling m complementary contributions based on Neyman
    """
    n = game.n
    sv = np.zeros(n)
    var = np.zeros((n, n))
    local_state = np.random.RandomState(None)
    utility_sum = np.zeros((n, n))
    count_sum = np.zeros((n, n))

    count = 0

    dir = os.path.join('res', 'temp_' + game.gt + '_' + str(game.n) +
                       'n_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    os.mkdir(dir)
    deldir = str(os.getcwd()) + '/' + dir

    if not initial_proc:
        utility = [[[] for _ in range(n)] for _ in range(n)]
        coef = [comb(n - 1, s) for s in range(n)]
        while True:
            temp_count = count
            for i in trange(n):
                idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
                for j in range(n):
                    if len(utility[i][j]) >= initial_m or len(
                            utility[i][j]) >= coef[j]:
                        continue
                    local_state.shuffle(idxs)
                    count += 1
                    if game.gt == 'voting':
                        u_1 = game.eval_utility(idxs[:j] + [i])
                        u_2 = 1 - u_1
                    elif game.gt == 'tree':
                        u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'ccx')
                    else:
                        u_1 = game.eval_utility(idxs[:j] + [i])
                        u_2 = game.eval_utility(idxs[j:])
                    utility[i][j].append(u_1 - u_2)
                    for l in range(n - 1):
                        if l < j:
                            utility[idxs[l]][j].append(u_1 - u_2)
                        else:
                            utility[idxs[l]][n - j - 2].append(u_2 - u_1)

            if count == temp_count:
                break

        for i in range(n):
            for j in range(n):
                if j != 0 and j != n - 1:
                    var[i][j] = np.var(utility[i][j], ddof=1)
                utility_sum[i][j] = np.sum(utility[i][j])
                count_sum[i][j] = len(utility[i][j])

        del utility
        gc.collect()

    else:
        args = split_permutation(math.ceil((n + 1) / 2), proc_num)
        for i in range(len(args)):
            for j in range(len(args[i])):
                args[i][j] += math.ceil(n / 2) - 1

        pool = Pool()
        if type == 1:
            func = partial(_ccshap_neyman_initial, game, initial_m, dir)
        else:
            func = partial(_ccshap_neyman_initial_a, game, initial_m, dir)
        ret = pool.map(func, args)
        pool.close()
        pool.join()

        print("initial load")
        for r in ret:
            count += r[0]
            var += np.load(dir + 'var_' + str(r[1]) + '.npy')
            utility_sum += np.load(dir + 'utility_sum_' + str(r[1]) + '.npy')
            count_sum += np.load(dir + 'count_sum_' + str(r[1]) + '.npy')

        shutil.rmtree(deldir)
        os.mkdir(dir)

    # compute variance
    var_sum = 0
    sigma_j = np.zeros(n)
    sigma_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            sigma_j[j] += var[i][j] / (j + 1)
            if n - j - 2 < 0:
                sigma_n_j[j] += 0
            else:
                sigma_n_j[j] += var[i][n - j - 2] / (n - j - 1)
        var_sum += np.sqrt(sigma_j[j] + sigma_n_j[j])

    # compute allocation
    m -= count
    print("m", m)
    local_m = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        local_m[j] = max(
            0,
            math.ceil(m * np.sqrt(sigma_j[j] + sigma_n_j[j]) / var_sum))

    # second stage
    args = split_num(local_m, proc_num)
    pool = Pool()
    func = partial(_ccshap_neyman_task, game, dir)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    print("second stage load")
    for i in trange(len(ret)):
        r = ret[i]
        utility_sum += np.load(dir + 'utility_' + str(r) + '.npy')
        count_sum += np.load(dir + 'count_' + str(r) + '.npy')
    shutil.rmtree(deldir)

    for i in range(n):
        for j in range(n):
            sv[i] += 0 if count_sum[i][j] == 0 else utility_sum[i][j] / \
                count_sum[i][j]
        sv[i] /= n

    return sv


def _ccshap_neyman_initial_a(game, initial_m, dir, j_list):
    """ Compute variance by Welford
    """
    n = game.n
    var = np.zeros((n, n))
    utility_sum = np.zeros((n, n))
    count_sum = np.zeros((n, n))
    mean_sum = np.zeros((n, n))

    coef = [comb(n - 1, s) for s in range(n)]

    nj_list = collections.deque()
    for j in j_list:
        nj_list.append(j)
        h = n - j - 2
        if h == j or h < 0:
            continue
        nj_list.appendleft(h)

    local_state = np.random.RandomState(None)
    count = 0
    temp_count = -1
    for ttt in trange(max_time):
        if count == temp_count:
            break
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in nj_list:
                if count_sum[i][j] >= initial_m or count_sum[i][j] >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                if game.gt == 'voting':
                    u_1 = game.eval_utility(idxs[:j] + [i])
                    u_2 = 1 - u_1
                elif game.gt == 'tree':
                    u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'ccx')
                else:
                    u_1 = game.eval_utility(idxs[:j] + [i])
                    u_2 = game.eval_utility(idxs[j:])

                before_mean = mean_sum[i][j]
                utility_sum[i][j] += (u_1 - u_2)
                count_sum[i][j] += 1
                mean_sum[i][j] = utility_sum[i][j] / count_sum[i][j]
                var[i][j] += (u_1 - u_2 - before_mean) * \
                    (u_1 - u_2 - mean_sum[i][j])

                for l in range(n - 1):
                    if l < j:
                        before_mean = mean_sum[idxs[l]][j]
                        utility_sum[idxs[l]][j] += (u_1 - u_2)
                        count_sum[idxs[l]][j] += 1
                        mean_sum[idxs[l]][j] = utility_sum[idxs[l]
                                                           ][j] / count_sum[idxs[l]][j]
                        var[idxs[l]][j] += (u_1 - u_2 -
                                            before_mean) * (u_1 - u_2 - mean_sum[idxs[l]][j])
                    else:
                        before_mean = mean_sum[idxs[l]][n - j - 2]
                        utility_sum[idxs[l]][n - j - 2] += (u_2 - u_1)
                        count_sum[idxs[l]][n - j - 2] += 1
                        mean_sum[idxs[l]][n - j - 2] = utility_sum[idxs[l]
                                                                   ][n - j - 2] / count_sum[idxs[l]][n - j - 2]
                        var[idxs[l]][n - j - 2] += (u_2 - u_1 - before_mean) * (
                            u_2 - u_1 - mean_sum[idxs[l]][n - j - 2])

    print("compute variance")
    for i in trange(n):
        for j in nj_list:
            var[i][j] = 0 if count_sum[i][j] == 1 else var[i][j] / \
                (count_sum[i][j] - 1)

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'var_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)

    np.save(dir + 'var_' + str(pid) + '.npy', var)
    np.save(dir + 'utility_sum_' + str(pid) + '.npy', utility_sum)
    np.save(dir + 'count_sum_' + str(pid) + '.npy', count_sum)

    # print("finish", nj_list)
    return count, pid


def _ccshap_neyman_initial(game, initial_m, dir, j_list):
    """Initial ccn
    """
    n = game.n
    var = np.zeros((n, n))
    utility_sum = np.zeros((n, n))
    count_sum = np.zeros((n, n))
    utility = [[[] for _ in range(n)] for _ in range(n)]

    coef = [comb(n - 1, s) for s in range(n)]

    nj_list = collections.deque()
    for j in j_list:
        nj_list.append(j)
        h = n - j - 2
        if h == j or h < 0:
            continue
        nj_list.appendleft(h)
    nj_list = list(nj_list)

    local_state = np.random.RandomState(None)

    count = 0
    temp_count = -1
    for ttt in trange(max_time):
        if count == temp_count:
            break
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in nj_list:
                if len(utility[i][j]) >= initial_m or len(utility[i][j]) >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                if game.gt == 'voting':
                    u_1 = game.eval_utility(idxs[:j] + [i])
                    u_2 = 1 - u_1
                elif game.gt == 'tree':
                    u_1, u_2 = game.eval_utility(idxs[:j] + [i], 'ccx')
                else:
                    u_1 = game.eval_utility(idxs[:j] + [i])
                    u_2 = game.eval_utility(idxs[j:])

                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)

    print("compute sum and variance")
    for i in trange(n):
        for j in nj_list:
            utility_sum[i][j] = np.sum(utility[i][j])
            count_sum[i][j] = len(utility[i][j])
            var[i][j] = 0 if count_sum[i][j] == 1 else np.var(
                utility[i][j], ddof=1)

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'var_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)

    np.save(dir + 'var_' + str(pid) + '.npy', var)
    np.save(dir + 'utility_sum_' + str(pid) + '.npy', utility_sum)
    np.save(dir + 'count_sum_' + str(pid) + '.npy', count_sum)

    print("finish", nj_list)
    return count, pid


def _ccshap_neyman_task(game, dir, local_m):
    """Compute remaining complementary contributions in ccn
    """
    n = game.n
    utility = np.zeros((n, n))
    count = np.zeros((n, n))
    idxs = np.arange(n)

    local_state = np.random.RandomState(None)
    for j in trange(n):
        for _ in range(local_m[j]):
            local_state.shuffle(idxs)

            if game.gt == 'voting':
                u_1 = game.eval_utility(idxs[:j + 1])
                u_2 = 1 - u_1
            elif game.gt == 'tree':
                u_1, u_2 = game.eval_utility(idxs[:j + 1], 'ccx')
            else:
                u_1 = game.eval_utility(idxs[:j + 1])
                u_2 = game.eval_utility(idxs[j + 1:])

            temp = np.zeros(n)
            temp[idxs[:j + 1]] = 1
            utility[:, j] += temp * (u_1 - u_2)
            count[:, j] += temp

            temp = np.zeros(n)
            temp[idxs[j + 1:]] = 1
            utility[:, n - j - 2] += temp * (u_2 - u_1)
            count[:, n - j - 2] += temp

    random.seed()
    state = random.getstate()
    random.setstate(state)
    pid = random.getrandbits(15)
    while os.path.exists(dir + 'count_' + str(pid) + '.npy'):
        pid = random.getrandbits(15)

    np.save(dir + 'utility_' + str(pid) + '.npy', utility)
    np.save(dir + 'count_' + str(pid) + '.npy', count)

    return pid


def ccshap_bernstein(game,
                     m,
                     initial_m,
                     r,
                     thelta,
                     flag=False) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Bernstein
    """
    n = game.n

    def select_coalitions_on(rankings):
        n = len(rankings)
        idxs_set = set([i for i in range(n)])
        res_idx = []
        res_s = math.inf
        max_score = 0
        for s in range(math.ceil(n / 2), n - 1):
            perm = [
                rankings[i][n - s - 1] - rankings[i][s - 1] for i in range(n)
            ]
            tidxs = np.argsort(perm)
            rank = perm[tidxs[s - 1]]
            local_state.shuffle(tidxs)
            idxs = []
            for i in tidxs:
                if perm[i] > rank:
                    idxs.append(tidxs[i])
            i = 0
            while len(idxs) < n - s and i < n:
                if perm[tidxs[i]] == rank:
                    idxs.append(tidxs[i])
                i += 1
            idxs += list(idxs_set - set(idxs))
            score = 0
            for i in range(n - s):
                score += rankings[idxs[i]][n - s - 1]
            for i in range(n - s, n):
                score += rankings[idxs[i]][s - 1]
            if score > max_score:
                max_score = score
                res_idx, res_s = idxs, s
        return max_score, res_idx, res_s

    # use original algorithm and a little random
    def select_coalitions_rd(rankings):
        local_state = np.random.RandomState(None)
        n = len(rankings)

        rankings_matrix = rankings

        idx = np.arange(n)
        max_score = 0
        for s in range(math.ceil(n / 2), n):
            local_state.shuffle(idx)

            l = []
            fl = []
            for i in range(s):
                temp = rankings_matrix[idx[i]][s - 1] - rankings_matrix[
                    idx[i]][n - s - 1]
                l.append(temp)
                fl.append(-temp)

            r = []
            fr = []
            for i in range(s, n):
                temp = rankings_matrix[idx[i]][n - s -
                                               1] - rankings_matrix[idx[i]][s -
                                                                            1]
                r.append(temp)
                fr.append(-temp)

            sli = np.argsort(fl)
            slr = np.argsort(fr)

            score = 0
            p = 0
            while p < s and p < n - s and score + l[sli[p]] + r[slr[p]] < score:
                score += l[sli[p]] + r[slr[p]]
                idx[sli[p]], idx[s + slr[p]] = idx[s + slr[p]], idx[sli[p]]
                p += 1

            score = 0
            for i in range(s):
                score += rankings_matrix[idx[i]][s - 1]
            for i in range(s, n):
                score += rankings_matrix[idx[i]][n - s - 1]

            if score > max_score:
                max_score = score
                res_idx, res_s = idx, s

        return max_score, res_idx, res_s

    local_state = np.random.RandomState(None)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    coef = [comb(n - 1, s) for s in range(n)]
    # thelta = np.log(10 / (1 + thelta))

    if flag:
        print("prob: ", thelta)
    else:
        print("prob: ", 1 - 3 * np.power(math.e, -thelta))

    count = 0
    while True:
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(utility[i][j]) >= initial_m or len(
                        utility[i][j]) >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                u_1 = game.eval_utility(idxs[:j] + [i])
                u_2 = game.eval_utility(idxs[j:])
                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)
        if count == temp_count:
            break

    for i in range(n):
        for j in range(n):
            if flag:
                var[i][j] = np.var(utility[i][j], ddof=1)
                var[i][j] = (np.sqrt(2 * var[i][j] * math.log(2 / thelta) /
                                     len(utility[i][j])) +
                             7 * math.log(2 / thelta) /
                             (3 * (len(utility[i][j]) - 1)))
            else:
                var[i][j] = np.var(utility[i][j])
                var[i][j] = (
                    np.sqrt(2 * var[i][j] * thelta / len(utility[i][j])) +
                    3 * r * thelta / len(utility[i][j]))

    count = sum([len(utility[0][j]) for j in range(n)])
    m -= count
    print(m)
    for _ in trange(m):
        min_score, idxs, j = select_coalitions_rd(var)

        u_1 = game.eval_utility(idxs[:j])
        u_2 = game.eval_utility(idxs[j:])
        for l in range(n):
            if l < j:
                utility[idxs[l]][j - 1].append(u_1 - u_2)
                if flag:
                    var[idxs[l]][j -
                                 1] = np.var(utility[idxs[l]][j - 1], ddof=1)
                    var[idxs[l]][
                        j -
                        1] = 1 * (np.sqrt(2 * var[idxs[l]][j - 1] * math.log(
                            2 / thelta) / len(utility[idxs[l]][j - 1])) +
                        7 * math.log(2 / thelta) /
                        (3 * (len(utility[idxs[l]][j - 1]) - 1)))
                else:
                    var[idxs[l]][j - 1] = np.var(utility[idxs[l]][j - 1])
                    var[idxs[l]][j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][j - 1] * thelta /
                                len(utility[idxs[l]][j - 1])) +
                        3 * 1 * thelta / len(utility[idxs[l]][j - 1]))
            else:
                utility[idxs[l]][n - j - 1].append(u_2 - u_1)
                if flag:
                    var[idxs[l]][n - j - 1] = np.var(
                        utility[idxs[l]][n - j - 1], ddof=1)
                    var[idxs[l]][n - j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][n - j - 1] * math.log(
                            2 / thelta) / len(utility[idxs[l]][n - j - 1])) +
                        7 * math.log(2 / thelta) /
                        (3 * (len(utility[idxs[l]][n - j - 1]) - 1)))
                else:
                    var[idxs[l]][n - j - 1] = np.var(utility[idxs[l]][n - j -
                                                                      1])
                    var[idxs[l]][n - j - 1] = 1 * (
                        np.sqrt(2 * var[idxs[l]][n - j - 1] * thelta /
                                len(utility[idxs[l]][n - j - 1])) +
                        3 * 1 * thelta / len(utility[idxs[l]][n - j - 1]))

    sv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sv[i] += np.mean(utility[i][j])
        sv[i] /= n
    return sv


def mc_shap_notpara(game, m) -> np.ndarray:
    """Compute Shapley value by sampling m permutations; not parallel
    """
    local_state = np.random.RandomState(None)
    n = game.n
    sv = np.zeros(n)
    idxs = np.arange(n)

    for _ in trange(m):
        local_state.shuffle(idxs)
        old_u = game.null  # game.null()[0]
        for j in range(1, n + 1):
            temp_u = game.eval_utility(idxs[:j])
            contribution = temp_u - old_u
            sv[idxs[j - 1]] += contribution
            old_u = temp_u

    return sv / m


def mch_notpara(game, m) -> np.ndarray:
    """Compute the Shapley value by sampling marginal contributions based on hoeffding; not parallel
    """
    n = game.n
    sv = np.zeros(n)
    mk = np.zeros(n)
    mk_list = []
    for i in range(n):
        mk_list.append(np.power(i + 1, 2 / 3))
    s = sum(mk_list)
    for i in range(n):
        mk[i] = math.ceil(m * mk_list[i] / s)

    print("mk:", sum(mk))

    local_state = np.random.RandomState(None)
    utility = np.zeros((n, n))
    for i in trange(n):
        idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
        for j in range(n):
            for _ in range(int(mk[j])):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j] += (u_2 - u_1)

    for i in range(n):
        for j in range(n):
            sv[i] += utility[i][j] / mk[j]
        sv[i] /= n

    return sv


def mcn_notpara(game, m) -> np.ndarray:
    """Compute the Shapley value by sampling marginal contributions by optimum allocation; not parallel
    """
    n = game.n
    var = np.zeros((n, n))
    utility = [[[] for i in range(n)] for i in range(n)]
    count = np.zeros((n, n))
    initial_m = max(2, int(m / (2 * n * n)))

    local_state = np.random.RandomState(None)
    for i in trange(n):
        idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
        for j in range(n):
            for _ in range(initial_m):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j].append(u_2 - u_1)
                count[i][j] += 1

    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j]) * count[i][j] / (count[i][j] - 1)

    mst = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mst[i][j] = max(0, m * var[i][j] / np.sum(var) - count[i][j])

    for i in trange(n):
        idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
        for j in range(n):
            for _ in range(int(mst[i][j])):
                local_state.shuffle(idxs)
                u_1 = game.eval_utility(idxs[:j])
                u_2 = game.eval_utility(idxs[:j] + [i])
                utility[i][j].append(u_2 - u_1)

    sv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sv[i] += np.mean(utility[i][j])
        sv[i] /= n

    return sv


def cc_shap_nopara(game, m) -> np.ndarray:
    """Compute the Shapley value by sampling m complementary contributions; not parallel
    """
    n = game.n
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        u_1 = game.eval_utility(idxs[:j])
        u_2 = game.eval_utility(idxs[j:])

        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    sv = np.zeros(n)
    for i in range(n + 1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv


def ccshap_neyman_notpara(game, initial_m, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Neyman; not parallel
    """
    n = game.n
    sv = np.zeros(n)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    local_state = np.random.RandomState(None)
    coef = [comb(n - 1, s) for s in range(n)]

    # initialize
    count = 0
    while True:
        temp_count = count
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(utility[i][j]) >= initial_m or len(
                        utility[i][j]) >= coef[j]:
                    continue
                local_state.shuffle(idxs)
                count += 1
                u_1 = game.eval_utility(idxs[:j] + [i])
                u_2 = game.eval_utility(idxs[j:])
                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)

        if count == temp_count:
            break

    # compute variance
    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j], ddof=1)

    # compute allocation
    var_sum = 0
    sigma_j = np.zeros(n)
    sigma_n_j = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        for i in range(n):
            sigma_j[j] += var[i][j] / (j + 1)
            if n - j - 2 < 0:
                sigma_n_j[j] += 0
            else:
                sigma_n_j[j] += var[i][n - j - 2] / (n - j - 1)
        var_sum += np.sqrt(sigma_j[j] + sigma_n_j[j])

    m -= count
    m = np.zeros(n)
    for j in range(math.ceil(n / 2) - 1, n):
        m[j] = max(
            0,
            math.ceil(m * np.sqrt(sigma_j[j] + sigma_n_j[j]) / var_sum))

    # second stage
    new_utility = np.zeros((n, n))
    count = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            new_utility[i][j] = np.sum(utility[i][j])
            count[i][j] = len(utility[i][j])

    idxs = np.arange(n)
    for j in trange(n):
        for _ in range(int(m[j])):
            local_state.shuffle(idxs)
            u_1 = game.eval_utility(idxs[:j + 1])
            u_2 = game.eval_utility(idxs[j + 1:])

            temp = np.zeros(n)
            temp[idxs[:j + 1]] = 1
            new_utility[:, j] += temp * (u_1 - u_2)
            count[:, j] += temp

            temp = np.zeros(n)
            temp[idxs[j + 1:]] = 1
            new_utility[:, n - j - 2] += temp * (u_2 - u_1)
            count[:, n - j - 2] += temp

    for i in range(n):
        for j in range(n):
            sv[i] += 0 if count[i][j] == 0 else new_utility[i][j] / count[i][j]
        sv[i] /= n

    return sv
