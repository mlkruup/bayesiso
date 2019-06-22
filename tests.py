import pandas as pd

from model import *
from old.isocal import *
from scores import *


def measure_tests():
    a = pd.Series([0.9, 0.7, 0.1, 0.2])
    b = pd.Series([1, 0, 0, 1])
    df = pd.DataFrame({"score": a, "cl": b})
    assert accuracy(df) == 0.5
    assert roc_auc(df) == 0.75
    assert bs(df) == 0.2875
    assert round(logloss(df), 4) == 0.7560


def isocal_test2():
    scores = [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1]
    classes = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    ic = IsotonicCalibration(None)
    cal_pairs = ic.fit_isotonic_regression(scores, classes)
    ic.cal_model = ic.isotonic_regression_model(cal_pairs)
    for p in cal_pairs:
        print(p.sc, p.csc, p.fsc, p.cl, ic.predict(p.sc))

def isocal_test():
    scores = [2.7, 0.5, 1.2, 2.2, 0.7, 0.3]
    classes = [1, 1, 0, 1, 0, 0]
    ic = IsotonicCalibration(None)
    cal_pairs = ic.fit_isotonic_regression(scores, classes)
    model = ic.isotonic_regression_model(cal_pairs)
    ic.cal_model = model
    res = ic.predict(2)
    for i in [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]:
        print(ic.predict(i))
    assert round(res, 2) == round(13/15, 2)


def isocal_test3():
    scores = [2.7, 0.4, 0.4, 0.4, 1.2, 2.2, 0.7, 0.3]
    classes = [1, 1, 1, 0, 0, 1, 0, 0]
    ic = IsotonicCalibration(None)
    cal_pairs = ic.fit_isotonic_regression(scores, classes)
    model = ic.isotonic_regression_model(cal_pairs)
    ic.cal_model = model
    for i in [0.3, 0.5, 0.7, 1.2, 2.2, 2.7]:
        print(ic.predict(i))



def tests():
    measure_tests()
    isocal_test()
    isocal_test2()
    isocal_test3()



if __name__ == "__main__":
    tests()