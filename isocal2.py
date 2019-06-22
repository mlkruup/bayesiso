import math
import pandas as pd
import numpy as np
from scipy import interpolate

class _MyIsotonicCalibration:

    def __init__(self, strict=True):
        self.strict = strict # if the monotonic increasing property has to be there (restored)
        self.all_same = False
        self.area = None

    def fit(self, z, y):

        self.preds = [[z, y]]

        classes = list(y)
        calibrated_sc_cl_pairs = self.fit_isotonic_regression(z, classes)
        self.pairs = calibrated_sc_cl_pairs
        self.cal_model = self.isotonic_regression_model(calibrated_sc_cl_pairs)
        return self

    def fit_isotonic_regression(self, scores, classes, debug=False):
        # http://www.cakesolutions.net/teamblogs/isotonic-regression-implementation-in-apache-spark
        pairs = sorted([CalObject(t[0], t[1]) for t in list(zip(scores, classes))], key=lambda x: (x.sc, x.cl), reverse=False)
        pairs = {i: [pairs[i]] for i in range(len(pairs))}
        # https://www.researchgate.net/profile/Jan_De_Leeuw/publication/24062238_Correctness_of_Kruskal's_algorithms_for_monotone_regression_with_ties/links/54cfbbf50cf29ca811003442.pdf
        # sklearn uses 2nd approach from here

        if debug:
            print(pairs)

        i = 0
        n = len(pairs)

        while i < n:
            #self.check_pair_correctness(pairs)
            j = i
            if debug:
                print("i", i, "j", j)
            if j < n - 1 and pairs[j][0].csc > pairs[j + 1][0].csc:
                j += 1
                if debug:
                    print("increased j to", j)

            assert abs(i - j) <= 1

            if i == j:
                i += 1
                if debug:
                    print("increased i to", i)
            else:
                while i >= 0 and pairs[i] is not None and pairs[j] is not None and pairs[i][0].csc > pairs[j][0].csc:
                    #self.check_pair_correctness(pairs)
                    pairs = self.pool(pairs, i, j)
                    i -= 1
                    while i >= 0 and pairs[i] is None:
                        i -= 1
                i = j

        res = self.format_pairs(pairs)
        return self.get_final_scores(res)


    def pool(self, pairs, i, j):
        new = pairs[i] + pairs[j]
        #if self.strict:
        #    for k in range(len(new) - 1):
        #        try:
        #            assert new[k].sc <= new[k + 1].sc
        #        except AssertionError:
        #            for x in range(i, j + 1):
        #                print(x, pairs[x])
        #            raise AssertionError
        score = sum([x.csc for x in new]) / len(new)
        for x in new:
            x.csc = score
        pairs[j] = new
        pairs[i] = None
        return pairs


    def format_pairs(self, pairs):
        res = []
        for i in range(len(pairs)):
            if pairs[i] is not None:
                temp = pairs[i]
                res += temp
        return res #sorted(res, key=lambda x: x.csc, reverse=False)






    def check_pair_correctness(self, pairs):
        a = 0
        for x in pairs:
            if pairs[x] is None:
                continue
            assert pairs[x][0].csc == pairs[x][-1].csc
            for y in pairs[x]:
                assert y.sc >= a
                a = y.sc

    def get_final_scores(self, pairs):
        for pair in pairs:
            pair.fsc = pair.csc
        return pairs

    def isotonic_regression_model(self, pairs):

        x = [pairs[i].sc for i in range(len(pairs))]
        y = [pairs[i].fsc for i in range(len(pairs))]

        self.area = np.trapz(y, np.linspace(0,1,len(y)))

        
        # in case all predicted scores are same
        if len(x) == 1 and len(y) == 1:
            x.append(x[0])
            y.append(y[0])

        model = interpolate.interp1d(x, y)

        self.xmin, self.ymin = min([xx for xx in x if xx != float("-inf")]), min(y)
        self.xmax, self.ymax = max([xx for xx in x if xx != float("inf")]), max(y)

        return model


    def predict(self, z):

        _0scores = []
        _1scores = []
        for value in z:
            p = self.predict_one(value)
            try:
                assert p is not None
                assert p >= 0 and p <= 1
            except:
                print(value, p, self.predict_one(1), self.predict_one(0.999999998699), self.xmin, self.ymin, self.xmax, self.ymax, self.predict_one(-100), self.predict_one(100))
                # if p < 1.1:
                #    p = 1.0
                # else:
                #    raise AssertionError
                raise AssertionError
            _1scores.append(p)
            _0scores.append(1 - p)

        #self.preds.append([z, x.copy(), _1scores])

        return np.array(_1scores)
        #return np.array(list(zip(_0scores, _1scores)))

    def predict_one(self, value):
        if value <= self.xmin:
            return self.ymin
        elif value >= self.xmax:
            return self.ymax
        return self.cal_model(value)


class CalObject:

    def __init__(self, sc, cl):
        self.sc = sc # score from the inital model
        self.cl = cl # true class
        self.csc = cl # calibrated score

    def __repr__(self):
        return ",".join([str(x) for x in [self.sc, self.cl, self.csc]])