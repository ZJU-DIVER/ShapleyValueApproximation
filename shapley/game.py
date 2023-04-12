import numpy as np
from sklearn import metrics
from shapley.utils import load_npy


class Game:
    def __init__(self,
                 gt,
                 n,
                 w=None,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 model=None):
        self.gt = gt
        self.n = n
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

        if w is None:
            if gt == 'voting':
                self.w = [
                    45, 41, 27, 26, 26, 25, 21, 17, 17, 14, 13, 13, 12, 12, 12,
                    11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 6,
                    6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3
                ]
                self.hw = np.sum(self.w) / 2
            elif gt == 'airport':
                self.w = [
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
                ]
        else:
            self.w = load_npy(w)
            if gt == 'voting':
                self.hw = np.sum(self.w) / 2

        if gt != 'tree' and gt != 'model' and gt != 'feature':
            self.w = np.array(self.w)

        if gt != 'feature':
            self.null = 0
        else:
            S = np.zeros((1, self.n), dtype=bool)
            self.null = self.model(S)[0][0]

    def eval_utility(self, x, m=None, hsw=None):
        """Evaluate the coalition utility.
        """
        if len(x) == 0:
            if self.gt == 'voting' and m is not None:
                return self.null, self.null
            else:
                return self.null
        if self.gt == 'voting':
            if m == 'mcx':
                if hsw is None:
                    r = np.sum(self.w[x])
                else:
                    r = hsw + self.w[x[-1]]
                return 1 if r > self.hw else 0, r
            else:
                r = np.sum(self.w[x])
                return 1 if r > self.hw else 0
        elif self.gt == 'airport':
            if m == 'mcx':
                if hsw is None:
                    r = np.max(self.w[x])
                else:
                    r = max(hsw, self.w[x[-1]])
            else:
                r = np.max(self.w[x])
            return r
        elif self.gt == 'tree':
            if m is None:
                r = self.n
                x = np.sort(x)
                m = r - (x[-1] - x[0])
                for i in range(1, len(x)):
                    t = x[i] - x[i - 1]
                    if t > m:
                        m = t
                return r - m
            elif m == 'mcx':
                r = self.n
                add_x = x[-1]
                x = np.sort(x)
                m = r - (x[-1] - x[0])
                for i in range(1, len(x)):
                    t = x[i] - x[i - 1]
                    if t > m:
                        m = t
                u_2 = r - m

                x = np.delete(x, np.where(x == add_x))
                if len(x) == 0:
                    u_1 = self.null
                else:
                    m = r - (x[-1] - x[0])
                    for i in range(1, len(x)):
                        t = x[i] - x[i - 1]
                        if t > m:
                            m = t
                    u_1 = r - m
                return u_1, u_2
            elif m == 'ccx':
                r = self.n
                x = np.sort(x)
                m = r - (x[-1] - x[0])

                left_m = 0
                i = 0
                while i < len(x) and x[i] == i:
                    left_m += 1
                    i += 1

                if i != len(x):
                    i = len(x) - 1
                    temp_i = 0
                    while i > -1 and x[i] == r - 1 - temp_i:
                        left_m += 1
                        i -= 1
                        temp_i += 1

                temp_t = 1
                for i in range(1, len(x)):
                    t = x[i] - x[i - 1]
                    if t > m:
                        m = t

                    if x[i] == x[i - 1] + 1:
                        temp_t += 1
                    else:
                        if temp_t > left_m:
                            left_m = temp_t
                        temp_t = 1
                left_m = max(left_m, temp_t)

                u_1 = r - m
                u_2 = r - left_m - 1
                if u_2 < 0:
                    u_2 = 0
                return u_1, u_2
        elif self.gt == 'model':
            temp_x, temp_y = self.x_train[x], self.y_train[x]
            single_pred_label = (True
                                 if len(np.unique(temp_y)) == 1 else False)

            if single_pred_label:
                y_pred = [temp_y[0]] * len(self.y_test)
            else:
                try:
                    self.model.fit(temp_x, temp_y)
                    y_pred = self.model.predict(self.x_test)
                except:
                    return None

            return metrics.accuracy_score(self.y_test, y_pred, normalize=True)
        elif self.gt == 'feature':
            S = np.zeros((1, self.n), dtype=bool)
            S[0][x] = 1
            u = self.model(S)[0][0]
            return u
