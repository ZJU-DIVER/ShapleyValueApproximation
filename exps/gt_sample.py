import sys
import os  # NOQA: E402
sys.path.append("..")  # NOQA: E402
from shapley.utils import *
from shapley.game import *
from shapley.gtsv import *
import os
from sklearn import svm


def run(gt, n, alg, times, samples, w=True, bdir='', proc_num=40):
    if gt != 'model':
        if w:
            game = Game(gt, n, w='w_' + str(n) + '.npy')
        else:
            game = Game(gt, n)
    else:
        x_train, y_train, x_test, y_test, columns_name = preprocess_data_sub(
            'b_train.csv', 'b_test.csv')
        # model = svm.LinearSVC(dual=False)
        model = svm.SVC(decision_function_shape='ovo')
        game = Game(gt, n, None, x_train, y_train, x_test, y_test, model)
        print("total acc:", game.eval_utility([i for i in range(n)]))

    dir = os.path.join(
        'res', bdir + game.gt + '_' + str(game.n) + 'n')
    if not os.path.exists(dir):
        os.mkdir(dir)
    dir += '/'

    if 'mc' in alg:
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_mc_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample
                mc_sv = mc_shap(game, m, proc_num)
                np.save(filename, mc_sv)
        print('finish mc', gt, n)

    print()
    if 'mch' in alg:
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_mch_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample
                mc_sv = mch(game, m, proc_num)
                np.save(filename, mc_sv)
        print('finish mch', gt, n)

    print()
    if 'mcn' in alg:
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_mcn_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample * n
                mc_sv = mcn(game, m, proc_num)
                np.save(filename, mc_sv)
        print('finish mcn', gt, n)

    print()
    if 'cc' in alg:
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_cc_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample * n
                mc_sv = cc_shap(game, m, proc_num)
                np.save(filename, mc_sv)
        print('finish cc', gt, n)

    print()
    if 'ccn' in alg:
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_ccn_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample * n
                initial_m = max(2, int(m / (n * n * 2)))
                print("initial_m:", initial_m)
                mc_sv = ccshap_neyman(game, initial_m, m, proc_num)
                np.save(filename, mc_sv)
        print('finish ccn', gt, n)

    print()
    if 'ccb' in alg:
        initial_m_ocb = 1
        r = 10
        thelta = 5
        flag = False
        for _ in range(times):
            for sample in samples:
                filename = dir + gt + '_ccb_n_' + \
                    str(n) + '_m_' + str(sample) + '_' + str(_) + '.npy'
                m = sample * n
                mc_sv = ccshap_bernstein(
                    game, m, initial_m_ocb, r, thelta, flag)
                np.save(filename, mc_sv)
        print('finish ccb')


if __name__ == '__main__':
    times = 1
    proc_num = 40
    ns = [51]
    gts = ['voting']
    alg = ['mc', 'mcn', 'mch', 'cc', 'ccn', 'ccb']
    samples = [1000]

    for i in range(len(ns)):
        run(gts[i], ns[i], alg, times, samples, w=False,
            bdir='error_', proc_num=proc_num)
