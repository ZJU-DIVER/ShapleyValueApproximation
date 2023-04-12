import sys  # NOQA: E402
sys.path.append("..")  # NOQA: E402
import pandas as pd
import numpy as np
from shapley.utils import *


def compute_std(gt, n, samples, algs, times, bbdir='', bdir='', norm=False):

    for _gt in gt:
        res = []
        for _algs in algs:
            max_res = []
            avg_res = []
            for _n in n:
                dir = bbdir + _gt + '_' + str(_n) + 'n' + '/' + _gt + '_'
                dir = bdir + dir

                sv = []
                for j in range(times):
                    ndir = dir + _algs + '_n_' + \
                        str(_n) + '_m_' + str(samples) + '_' + str(j) + '.npy'
                    a = load_npy(ndir)
                    if norm:
                        a /= np.sum(a)
                    sv.append(a)

                mean = np.mean(sv, axis=0)
                r = np.std(sv, axis=0)

                c = 0
                cv = []
                for j in range(len(mean)):
                    temp = True
                    for _sv in sv:
                        if _sv[j] <= 0:
                            temp = False
                    if temp:
                        c += 1
                        cv.append(abs(r[j] / mean[j]))

                mr = np.mean(cv)
                ar = np.max(cv)

                print(_gt, _n, _algs, (ar, mr))
                max_res.append(ar)
                avg_res.append(mr)
            res.append(tuple(avg_res))
        a = pd.DataFrame(res)
        a.to_csv(_gt + '.csv')


def compute_error(gt, n, samples, algs, times, exact, bbdir='', bdir='', norm=False):
    num = len(exact)

    for _gt in gt:
        res_avg_err = []
        res_max_err = []
        for _algs in algs:
            avg_err = []
            max_err = []
            for _n in n:
                temp_err = np.zeros(num)

                dir = bbdir + _gt + '_' + str(_n) + 'n' + '/' + _gt + '_'
                dir = bdir + dir

                for j in range(times):
                    ndir = dir + _algs + '_n_' + \
                        str(_n) + '_m_' + str(samples) + '_' + str(j) + '.npy'
                    sv = load_npy(ndir)
                    if norm:
                        sv /= np.sum(sv)

                    for i in range(len(sv)):
                        if exact[i] == 0:
                            continue
                        temp_err[i] += abs((sv[i] - exact[i]) / exact[i])

                temp_err /= times
                ar = np.mean(temp_err)
                mr = np.max(temp_err)
                print(_gt, _n, _algs, (ar, mr))
                max_err.append(mr)
                avg_err.append(ar)
            res_max_err.append(tuple(max_err))
            res_avg_err.append(tuple(avg_err))

        a = pd.DataFrame(res_avg_err)
        a.to_csv(_gt + '_avg.csv')
        a = pd.DataFrame(res_avg_err)
        a.to_csv(_gt + '_max.csv')


if __name__ == '__main__':
    voting_exact = [0.08831, 0.07973, 0.05096, 0.04898, 0.04898, 0.04700, 0.03917, 0.03147, 0.03147, 0.02577, 0.02388,
                    0.02388, 0.02200, 0.02200, 0.02200, 0.02013, 0.01827, 0.01827, 0.01827, 0.01827, 0.01641, 0.01641,
                    0.01641, 0.01641, 0.01456, 0.01456, 0.01272, 0.01272, 0.01272, 0.01272, 0.01088, 0.01088, 0.01088, 0.01088,
                    0.009053, 0.007230, 0.007230, 0.007230, 0.007230, 0.007230, 0.007230, 0.007230, 0.007230, 0.007230,
                    0.005412, 0.005412, 0.005412, 0.005412, 0.005412, 0.005412, 0.005412]

    compute_error(['voting'], [51], 1000, ['mc', 'mcn', 'mch', 'cc', 'ccn', 'ccb'], 1,
                  voting_exact, bbdir='error_', norm=False)
