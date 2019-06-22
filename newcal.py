import math
import pandas as pd
import numpy as np
#from sympy import *
from scipy import interpolate
from isocal2 import _MyIsotonicCalibration
import warnings
#warnings.filterwarnings("error")

class _MyIsotonicCalibration_NEW:

    def __init__(self, sampling = 10000, distr = 1, beta = 0.1, kind = 2):
        self.sampling = sampling
        self.distr = distr
        self.beta = beta
        self.area = None
        self.kind = kind
        self.likelihoods = None
        self.diff_scores = None


    def fit(self, z_join, y_join, z_repl, y_repl, nr0=None, nr1=None):

        self.preds = [[z_join, y_join]]
        self.preds2 = [[z_repl, y_repl]]
        self.diff_scores = len(z_join)

        classes = list(y_join)
        calibrated_sc_cl_pairs = self.fit_isotonic_regression(z_join, classes, z_repl, y_repl, nr0, nr1)
        self.pairs = calibrated_sc_cl_pairs
        self.cal_model = self.isotonic_regression_model(calibrated_sc_cl_pairs)
        return self

    def fit_isotonic_regression(self, scores, classes, z_repl, y_repl, nr0, nr1, debug=False):
        if nr0 is not None:
            pairs = sorted([CalObject(t[0], t[1], t[2], t[3]) for t in list(zip(scores, classes, nr0, nr1))], key=lambda x: (x.sc, x.cl), reverse=False)
            scores = [pair.sc for pair in pairs]
            classes = [pair.cl for pair in pairs]
            nr0 = [pair.n0 for pair in pairs]
            nr1 = [pair.n1 for pair in pairs]
        else:
            pairs = sorted([CalObject(t[0], t[1], t[2], t[3]) for t in list(zip(scores, classes, [None for i in range(len(classes))], [None for i in range(len(classes))]))], key=lambda x: (x.sc, x.cl), reverse=False)
            scores = [pair.sc for pair in pairs]
            classes = [pair.cl for pair in pairs]
        if self.sampling:
            if self.kind == 1:
                self.res1 = isotonic_e_sampling(classes, n=self.sampling, distr = self.distr, beta = self.beta,
                                                nr0 = nr0, nr1 = nr1)
            elif self.kind == 2:
                self.res1, self.likelihoods = isotonic_e_sampling2(scores, classes, z_repl, y_repl, n=self.sampling, distr = self.distr, beta = self.beta,
                                                nr0 = nr0, nr1 = nr1)
            elif self.kind == 3:
                self.res1, self.likelihoods = isotonic_e_sampling3(classes, n=self.sampling, distr = self.distr, beta = self.beta,
                                                nr0 = nr0, nr1 = nr1)
        else:
            self.res1 = isotonic_e(classes)
        res = [float(x) for x in self.res1]
        return self.get_final_scores(res, scores, classes)


    def get_final_scores(self, res, scores, classes):
        pairs = []
        for i in range(len(res)):
            x = CalObject(scores[i], classes[i], None, None)
            x.csc = res[i]
            pairs.append(x)
        return pairs

    def isotonic_regression_model(self, pairs):

        x = [pairs[i].sc for i in range(len(pairs))]
        y = [pairs[i].csc for i in range(len(pairs))]
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

    def __init__(self, sc, cl, n0, n1):
        self.sc = sc # score from the inital model
        self.cl = cl # true class
        self.csc = cl # calibrated score
        self.n0 = n0
        self.n1 = n1

    def __repr__(self):
        return ",".join([str(x) for x in [self.sc, self.cl, self.csc]])


def integrate_one_by_one(v, j, p, results=False):
    res = 1
    intermediate_results = [None for i in range(len(v))]
    for i in range(len(v) - 1, 0, -1):
        if results and j < i:
            res = results[i]
            continue
        if i == j:
            f = ((1 - p[i]) * res * (1 - v[i]) + p[i] * res * v[i]) * p[i] * 2
        else:
            f = ((1 - p[i]) * res * (1 - v[i]) + p[i] * res * v[i]) * 2
        res = None #integrate(expand(f), (p[i], p[i - 1], 1))
        intermediate_results[i] = res
    if j == 0:
        f = ((1 - p[0]) * res * (1 - v[0]) + p[0] * res * v[0]) * 2 * len(v) * p[0] * 2
    else:
        f = (((1 - p[0]) * res * (1 - v[0]) + p[0] * res * v[0]) * 2 * len(v)) * 2
    res = None #integrate(expand(f), (p[0], 0, 1))
    # return res, None # remove later
    intermediate_results[0] = res
    if results == False:
        return res, intermediate_results
    else:
        return res


def isotonic_e(classes):
    n = len(classes)
    p = None #symbols(' '.join(["p%d" % i for i in range(1, n + 1)]))

    # prod = [p[i] if classes[i] == 1 else (1 - p[i]) for i in range(len(classes))]
    # density_fun = ((n * 2) * np.prod(np.array(prod)))

    # density = integrate(density_fun, *[(p[i], p[i-1],1) for i in range(n - 1, 0, -1)], (p[0], 0, 1))
    density, intermediate_results = integrate_one_by_one(classes, None, p, results=False)
    # return density
    results = []
    for i in range(len(p)):
        # f = (p[i] * (n * 2) * np.prod(np.array(prod)))
        # res = integrate(f, *[(p[i], p[i-1],1) for i in range(n - 1, 0, -1)], (p[0], 0, 1))
        res = integrate_one_by_one(classes, i, p, results=intermediate_results)
        results.append(res / density)
        if results[-1] > 1 or results[-1] < 0:
            print(res, density, i)
            break
            # print(res, density, i)
    return results


def generate_scores(n, lower, upper):
    if n == 0:
        return []
    index = np.random.randint(1, (n + 1))
    value = np.random.uniform(lower, upper)
    left = generate_scores(index - 1, lower, value)
    right = generate_scores(n - index, value, upper)
    return left + [value] + right


def generate_scores2(n, lower, upper):
    if n == 0:
        return []
    if n == 1:
        index = 1
    else:
        index = n // 2
    value = np.random.uniform(lower, upper)
    left = generate_scores2(index - 1, lower, value)
    right = generate_scores2(n - index, value, upper)
    return left + [value] + right


def isotonic_e_sampling(classes, n=10000, distr=1, beta=0.1, nr0=None, nr1=None):
    all_f = [[] for cl in classes]
    all_dens = []
    nr_classes = len(classes)
    try:
        for j in range(n):
            if distr == 1:
                scores = generate_scores(len(classes), 0, 1)  # sorted(np.random.uniform(0,1,nr_classes))
            elif distr == 2:
                scores = generate_scores2(len(classes), 0, 1)
            elif distr == 3:
                scores = generate_scores3(len(classes), 0, 1, beta=beta)
            elif distr == 4:
                scores = generate_scores4(len(classes), 0, 1)
            elif distr == 5:
                scores = generate_scores5(len(classes), 0, 1)
            elif distr == 6:
                scores = generate_scores6(len(classes))
            if nr0 is not None:
                dens = np.exp(nr_classes / 2 + np.sum(
                    [np.log(1 - scores[i]) * nr0[i] + np.log(scores[i]) * nr1[i] for i in range(len(classes))]))
            else:
                dens = np.exp(nr_classes / 2 + np.sum(
                    [np.log(1 - scores[i]) if classes[i] == 0 else np.log(scores[i]) for i in range(len(classes))]))
            for i in range(len(classes)):
                all_f[i].append(dens * scores[i])
            all_dens.append(dens)

        dens = np.mean(all_dens)
        x = [np.mean(s) / dens for s in all_f]
    except RuntimeWarning:
        print(classes)
        raise Exception
    return x


def isotonic_e_sampling2(scores, classes, z_repl, y_repl, n=10000, distr=1, beta=0.1, nr0=None, nr1=None):
    all_f = [[] for cl in classes]
    all_dens = []
    nr_classes = len(classes)
    if distr == 7:
        iso_scores = get_isotonic_sample(classes)
    if distr == 8:
        #iso_scores = get_isotonic_sample(classes)
        iso_scores, iso_scores_repl = get_isotonic_sample(y_repl, z_repl = z_repl, z_join = scores)
        lower_bounds, upper_bounds = calculate_isotonic_bounds(classes, iso_scores, z_repl = z_repl, y_repl = y_repl, z_join = scores, iso_scores_repl = iso_scores_repl, w = max(1, int(len(y_repl) / 20)))
    if distr == 9:
        iso_scores, iso_scores_repl = get_isotonic_sample(y_repl, z_repl=z_repl, z_join=scores)
        bounds = calculate_minimal_bounds2(y_repl, iso_scores_repl, z_repl, step=1)
        start = ("l5", "u5")
    for j in range(n):
        if distr == 1:
            scores = generate_scores(len(classes), 0, 1)  # sorted(np.random.uniform(0,1,nr_classes))
        elif distr == 2:
            scores = generate_scores2(len(classes), 0, 1)
        elif distr == 3:
            scores = generate_scores3(len(classes), 0, 1, beta=beta)
        elif distr == 4:
            scores = generate_scores4(len(classes), 0, 1)
        elif distr == 5:
            scores = generate_scores5(len(classes), 0, 1)
        elif distr == 6:
            scores = generate_scores6(len(classes))
        elif distr == 7:
            if j == -1:
                scores = iso_scores
            else:
                scores = generate_scores7(len(classes), 0, 1, iso_scores)
        elif distr == 8:
            scores = generate_scores8(len(classes), 0, 1, lower_bounds, upper_bounds)
        elif distr == 9:
            scores = generate_scores8(len(classes), 0, 1, bounds[start[0]], bounds[start[1]])


        if nr0 is not None:
            dens = np.sum(
                [(np.log(1 - scores[i]) if scores[i] < 1 else np.log(1e-323)) * nr0[i] + (np.log(scores[i]) if scores[i] > 0 else np.log(1e-323)) * nr1[i] for i in range(len(classes))])
        else:
            dens = np.sum(
                [np.log(1 - scores[i]) if classes[i] == 0 else np.log(scores[i]) for i in range(len(classes))])
        for i in range(len(classes)):
            all_f[i].append(scores[i])
        all_dens.append(dens)

        if j == 1000 and distr == 9:
            c = np.max(all_dens)
            # print(np.max(all_dens), np.min(all_dens), c)
            all_dens_temp = [np.exp(a - c) if (a - c) > -745 else 0 for a in all_dens]
            dens = np.sum(all_dens_temp)
            if dens < 2:
                start = ("l4", "u4")


    try:
        c = np.max(all_dens)
        #print(np.max(all_dens), np.min(all_dens), c)
        all_dens = [np.exp(a - c) if (a - c) > -745 else 0 for a in all_dens]
        dens = np.sum(all_dens)
        #print(dens, np.max(all_dens), np.min(all_dens))
        x = [np.sum(np.array(all_f[i]) * np.array(all_dens)) / dens for i in range(nr_classes)]
        #x = [np.exp(np.log(np.mean(np.array(all_f[i]) * np.array(all_dens))) - np.log(dens)) for i in range(nr_classes)]

    except RuntimeWarning:
        print(classes)
        raise Exception

    return x, all_dens


def calculate_isotonic_bounds(classes, iso_scores, z_repl = None, y_repl = None, z_join = None, iso_scores_repl = None, w = 100):
    if z_repl is None:
        e = 1 / np.sqrt(len(classes))
        classes_reversed = list(reversed([abs(1 - cl) for cl in classes]))
        iso_scores_reversed = list(reversed([abs(1 - sc) for sc in iso_scores]))
        lower_bounds = []
        upper_bounds = []
        for i in range(len(classes)):
            if i == len(classes) - 1:
                upper = 1
            else:
                upper = np.mean(classes[(i + 1):(i + 1 + w)])
            upper_bounds.append(max(upper, upper_bounds[i-1],iso_scores[i]) if len(upper_bounds) > 0 else max(upper, iso_scores[i]))
            if i == len(classes) - 1:
                lower = 1
            else:
                lower = np.mean(classes_reversed[(i + 1):(i + 1 + w)])
            lower_bounds.append(max(lower, lower_bounds[i-1], iso_scores_reversed[i]) if len(lower_bounds) > 0 else max(lower, iso_scores_reversed[i]))
        upper_bounds = [min(1, upper + e) for upper in upper_bounds]
        lower_bounds = [max(0, lower - e) for lower in list(reversed([1 - lower for lower in lower_bounds]))]
        return lower_bounds, upper_bounds
    else:
        e = 1 / np.sqrt(len(y_repl))
        classes_reversed = list(reversed([abs(1 - cl) for cl in y_repl]))
        iso_scores_reversed = list(reversed([abs(1 - sc) for sc in iso_scores_repl]))
        lower_bounds = []
        upper_bounds = []
        for i in range(len(y_repl)):
            if i == len(y_repl) - 1:
                upper = 1
            else:
                upper = np.mean(y_repl[(i + 1):(i + 1 + w)])
            upper_bounds.append(max(upper, upper_bounds[i-1],iso_scores_repl[i]) if len(upper_bounds) > 0 else max(upper, iso_scores_repl[i]))
            if i == len(y_repl) - 1:
                lower = 1
            else:
                lower = np.mean(classes_reversed[(i + 1):(i + 1 + w)])
            lower_bounds.append(max(lower, lower_bounds[i-1], iso_scores_reversed[i]) if len(lower_bounds) > 0 else max(lower, iso_scores_reversed[i]))
        upper_bounds = [min(1, upper + e) for upper in upper_bounds]
        lower_bounds = [max(0, lower - e) for lower in list(reversed([1 - lower for lower in lower_bounds]))]
        lower_bounds2, upper_bounds2 = join_bounds(lower_bounds, upper_bounds, z_repl)

        return lower_bounds2, upper_bounds2



def join_bounds(lower_bounds, upper_bounds, z_repl):
    upper_bounds2 = []
    lower_bounds2 = []
    l_b = []
    u_b = []
    for i in range(len(upper_bounds) - 1):
        score = z_repl[i]
        score_next = z_repl[i + 1]
        if score == score_next:
            l_b.append(lower_bounds[i])
            u_b.append(upper_bounds[i])
        if score_next > score:
            l_b.append(lower_bounds[i])
            lower_bounds2.append(np.min(l_b))
            l_b = []
            u_b.append(upper_bounds[i])
            upper_bounds2.append(np.max(u_b))
            u_b = []
    if score == score_next:
        lower_bounds2.append(np.min(l_b))
        upper_bounds2.append(np.max(u_b))
    elif score_next > score:
        lower_bounds2.append(lower_bounds[len(upper_bounds) - 1])
        upper_bounds2.append(upper_bounds[len(upper_bounds) - 1])
    return lower_bounds2, upper_bounds2

def calc_sd(points):
    n = len(points)
    nr1 = sum(points)
    nr0 = n - nr1
    p_lc = (nr1 + 1) / (n + 2)
    sd = np.sqrt(n * p_lc * (1 - p_lc)) / n
    return sd, nr1, n


def calculate_isotonic_bounds_sd(classes, iso_scores, w = 100, times_sd = 3):
    lower_bounds = []
    upper_bounds = []
    for i in range(len(classes)):
        if i == 0:
            lower = 0
        elif (i - w) < 0:
            points = classes[0:i]
            sd, nr1, n = calc_sd(points)
            lower = max(0, nr1 / n - sd * times_sd)
        else:
            points = classes[(i - w):i]
            sd, nr1, n = calc_sd(points)
            lower = max(0, nr1 / n - sd * times_sd)
        if i == len(classes) - 1:
            upper = 1
        else:
            points = classes[(i + 1):(i + 1 + w)]
            sd, nr1, n = calc_sd(points)
            upper = min(1, nr1 / n + sd * times_sd)
        upper_bounds.append(upper)
        lower_bounds.append(lower)
    return lower_bounds, upper_bounds


def calculate_minimal_bounds(classes, iso_scores, times_sd = 3, step = 1):
    n = len(classes)
    lower_bounds, upper_bounds = calculate_isotonic_bounds_sd(classes, iso_scores, w = 1, times_sd = times_sd)
    for w in range(2, n, step):
        lower_bounds_w, upper_bounds_w = calculate_isotonic_bounds_sd(classes, iso_scores, w = w, times_sd = times_sd)
        lower_bounds = [max(lower_bounds[i], lower_bounds_w[i]) for i in range(n)]
        upper_bounds = [min(upper_bounds[i], upper_bounds_w[i]) for i in range(n)]
    lower_bounds2 = make_monotonic(lower_bounds, "lower")
    upper_bounds2 = make_monotonic(upper_bounds, "upper")
    return lower_bounds2, upper_bounds2


def calculate_minimal_bounds2(classes, iso_scores, z_repl, step=1):
    n = len(classes)
    p_lc_matrix_lower = np.zeros((n, n))
    p_lc_matrix_upper = np.ones((n, n))
    sd_matrix = np.zeros((n, n))
    for start in range(0, n, step):
        for end in range(start, n, step):
            part = classes[start:(end + 1)]
            len_part = len(part)
            nr1 = np.sum(part)
            nr0 = len_part - nr1
            p_lc = (nr1 + 1) / (len_part + 2)
            sd = np.sqrt(len_part * p_lc * (1 - p_lc)) / len_part
            p_lc_matrix_lower[start][end] = p_lc
            p_lc_matrix_upper[start][end] = p_lc
            sd_matrix[start][end] = sd

    lower3_matrix = p_lc_matrix_lower - sd_matrix * 3
    upper3_matrix = p_lc_matrix_upper + sd_matrix * 3
    lower4_matrix = p_lc_matrix_lower - sd_matrix * 4
    upper4_matrix = p_lc_matrix_upper + sd_matrix * 4
    lower5_matrix = p_lc_matrix_lower - sd_matrix * 5
    upper5_matrix = p_lc_matrix_upper + sd_matrix * 5

    lower3_values = choose_correct_values(np.max(lower3_matrix, axis=0), step)
    lower4_values = choose_correct_values(np.max(lower4_matrix, axis=0), step)
    lower5_values = choose_correct_values(np.max(lower5_matrix, axis=0), step)

    upper3_values = choose_correct_values(np.min(upper3_matrix, axis=1), step)
    upper4_values = choose_correct_values(np.min(upper4_matrix, axis=1), step)
    upper5_values = choose_correct_values(np.min(upper5_matrix, axis=1), step)

    iso_scores_parts = [iso_scores[i] for i in range(0, len(iso_scores), step)]

    lower3_values = check_if_iso_is_in(make_monotonic(lower3_values, "lower"), iso_scores_parts, "lower")
    lower4_values = check_if_iso_is_in(make_monotonic(lower4_values, "lower"), iso_scores_parts, "lower")
    lower5_values = check_if_iso_is_in(make_monotonic(lower5_values, "lower"), iso_scores_parts, "lower")

    upper3_values = check_if_iso_is_in(make_monotonic(upper3_values, "upper"), iso_scores_parts, "upper")
    upper4_values = check_if_iso_is_in(make_monotonic(upper4_values, "upper"), iso_scores_parts, "upper")
    upper5_values = check_if_iso_is_in(make_monotonic(upper5_values, "upper"), iso_scores_parts, "upper")

    lower3_bounds, upper3_bounds = join_bounds(lower3_values, upper3_values, z_repl)
    lower4_bounds, upper4_bounds = join_bounds(lower4_values, upper4_values, z_repl)
    lower5_bounds, upper5_bounds = join_bounds(lower5_values, upper5_values, z_repl)


    return {"l3": lower3_bounds, "l4": lower4_bounds, "l5": lower5_bounds,
            "u3": upper3_bounds, "u4": upper4_bounds, "u5": upper5_bounds}


def choose_correct_values(values, step):
    if step == 1:
        return values
    new_values = []
    for i in range(0, len(values), step):
        new_values.append(values[i])
    return new_values

def make_monotonic(bounds, bound_type):
    if bound_type == "lower":
        new_bounds = bounds[:]
    elif bound_type == "upper":
        new_bounds = list(reversed([1 - bound for bound in bounds]))
    for i in range(1, len(bounds)):
        if new_bounds[i - 1] > new_bounds[i]:
            new_bounds[i] = new_bounds[i - 1]
    if bound_type == "upper":
        new_bounds = list(reversed([1 - bound for bound in new_bounds]))
    return new_bounds

def check_if_iso_is_in(bounds, iso_scores, bound_type):
    new_bounds = bounds[:]
    for i in range(len(bounds)):
        if bound_type == "lower":
            new_bounds[i] = min(bounds[i], iso_scores[i])
        elif bound_type == "upper":
            new_bounds[i] = max(bounds[i], iso_scores[i])
    return new_bounds

def get_isotonic_sample(classes, z_repl = None, z_join = None):
    scores = z_repl if z_repl is not None else np.linspace(0,1,len(classes))
    cal = _MyIsotonicCalibration()
    cal_model = cal.fit(scores, classes)
    if z_join is None:
        res = cal_model.predict(scores)
        return res
    else:
        res1 = cal_model.predict(z_join)
        res2 = cal_model.predict(z_repl)
        return res1, res2


def get_isotonic_samples(classes):
    res = []
    res.append(get_isotonic_sample(classes))
    indeces = np.linspace(0,len(classes) - 1,min(len(classes),100))
    for i in indeces:
        i = int(i)
        classes2 = classes[:]
        classes2[i] = abs(classes2[i] - 1)
        res.append(get_isotonic_sample(classes2))
    return res


def isotonic_e_sampling3(classes, n=10000, distr=1, beta=0.1, nr0=None, nr1=None):
    all_f = [[] for cl in classes]
    all_dens = []
    nr_classes = len(classes)
    for j in range(n):
        if distr == 1:
            scores = generate_scores(len(classes), 0, 1)  # sorted(np.random.uniform(0,1,nr_classes))
        elif distr == 2:
            scores = generate_scores2(len(classes), 0, 1)
        elif distr == 3:
            scores = generate_scores3(len(classes), 0, 1, beta=beta)
        elif distr == 4:
            scores = generate_scores4(len(classes), 0, 1)
        elif distr == 5:
            scores = generate_scores5(len(classes), 0, 1)
        elif distr == 6:
            scores = generate_scores6(len(classes))
        if nr0 is not None:
            dens = np.sum(
                [np.log(1 - scores[i]) * nr0[i] + np.log(scores[i]) * nr1[i] for i in range(len(classes))])
        else:
            dens = np.sum(
                [np.log(1 - scores[i]) if classes[i] == 0 else np.log(scores[i]) for i in range(len(classes))])
        for i in range(len(classes)):
            all_f[i].append(scores[i])
        all_dens.append(dens)

        #all_f_iso = [[] for cl in classes]
        #all_dens_iso = []
    res = get_isotonic_samples(classes)
    for scores in res:
        if nr0 is not None:
            dens = np.sum(
                [(np.log(1 - scores[i]) if scores[i] < 1 else np.log(1e-323)) * nr0[i] + (np.log(scores[i]) if scores[i] > 0 else np.log(1e-323)) * nr1[i] for i in range(len(classes))])
        else:
            dens = np.sum(
                [np.log(1 - scores[i]) if classes[i] == 0 else np.log(scores[i]) for i in range(len(classes))])
        for i in range(len(classes)):
            all_f[i].append(scores[i])
        all_dens.append(dens)







    try:
        c = np.max(all_dens)
        print(np.max(all_dens), np.min(all_dens), c)
        all_dens = [np.exp(a - c) if (a - c) > -745 else 0 for a in all_dens]
        dens = np.sum(all_dens)
        print(dens, np.max(all_dens), np.min(all_dens))
        x = [np.sum(np.array(all_f[i]) * np.array(all_dens)) / dens for i in range(nr_classes)]
        #x = [np.exp(np.log(np.mean(np.array(all_f[i]) * np.array(all_dens))) - np.log(dens)) for i in range(nr_classes)]

    except RuntimeWarning:
        print(classes)
        raise Exception

    return x, all_dens

def generate_scores3(n, lower, upper, beta=0.1):
    if n == 0:
        return []
    index = np.random.randint(1, (n + 1))
    value = np.random.beta(beta, beta, 1)[0]
    value = lower + (upper - lower) * value
    left = generate_scores3(index - 1, lower, value, beta=beta)
    right = generate_scores3(n - index, value, upper, beta=beta)
    return left + [value] + right

def generate_scores7(n, lower, upper, iso_scores):
    if n == 0:
        return []
    index = np.random.randint(1, (n + 1))
    iso_value = iso_scores[index - 1]
    iso_lower = max(iso_value - 0.05, lower)
    iso_upper = min(iso_value + 0.05, upper)
    value = np.random.uniform(iso_lower, iso_upper, 1)[0]
    left = generate_scores7(index - 1, lower, value, iso_scores[:index - 1])
    right = generate_scores7(n - index, value, upper, iso_scores[index:])
    return left + [value] + right


def generate_scores8(n, lower, upper, lower_bounds, upper_bounds):
    if n == 0:
        return []
    index = np.random.randint(1, (n + 1))
    iso_lower = max(lower_bounds[index - 1], lower)
    iso_upper = min(upper_bounds[index - 1], upper)
    value = np.random.uniform(iso_lower, iso_upper, 1)[0]
    left = generate_scores8(index - 1, lower, value, lower_bounds[:index - 1], upper_bounds[:index - 1])
    right = generate_scores8(n - index, value, upper, lower_bounds[index:], upper_bounds[index:])
    return left + [value] + right


def generate_scores4(n, lower, upper):
    if n == 0:
        return []
    elif n == 1:
        index = 1
    elif n == 2:
        index = np.random.randint(1,3,1)
    else:
        index = np.int(np.round(np.random.beta(0.1,0.1,1)[0] * (n - 1))) + 1
    value = np.random.uniform(lower, upper, 1)[0]
    left = generate_scores4(index - 1, lower, value)
    right = generate_scores4(n - index, value, upper)
    return left + [value] + right


def generate_scores5(n, lower, upper):
    if n == 0:
        return []
    elif n == 1:
        index = 1
    elif n == 2:
        index = np.random.randint(1,3,1)
    else:
        index = np.int(np.round(np.random.beta(0.1,0.1,1)[0] * (n - 1))) + 1
    value = np.random.beta(0.1, 0.1, 1)[0]
    value = lower + (upper - lower) * value
    left = generate_scores5(index - 1, lower, value)
    right = generate_scores5(n - index, value, upper)
    return left + [value] + right


def build_line_real_axis(xlow, xhigh, ylow, yhigh):
    if yhigh - ylow < 0.0001:
        return []
    x = np.random.uniform(xlow, xhigh, 1)[0]
    y = np.random.uniform(ylow, yhigh, 1)[0]
    return build_line_real_axis(xlow, x, ylow, y) + [(x, y)] + build_line_real_axis(x, xhigh, y, yhigh)

def generate_scores6(n):
    pairs = [(0,0)] + build_line_real_axis(0,1,0,1) + [(1,1)]
    x = [s[0] for s in pairs]
    y = [s[1] for s in pairs]
    model = interpolate.interp1d(x, y)
    return model(np.linspace(0, 1, n + 2)[1:-1])