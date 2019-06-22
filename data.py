import pandas as pd
import numpy as np
import random
from utils import Pair, join_pairs_with_same_score

class Data:
    def __init__(self, dsname, calibration_size):
        self.dsname = dsname
        self.calibration_size = [100, 1000, 3000] #calibration_size
        self.read_data()


    def read_data(self):

        max_class_sizes = {"sea": 614342, "breastw": 25820, "sonar": 532538, "heart": 555946, "planes": 20420,
                           "house16": 16040, "cal_housing": 12255, "houses": 11726, "house8": 16040, "fried": 20427,
                           "letter": 19187, "spectf_test": 784810, "austr": 573051, "spectf": 718700, "click1": 142949,
                           "click2": 332393, "click3": 33220, "skin": 194198, "creditcard": 284315, "numerai": 48658}
        if self.dsname == "sea":
            self.data = read_csv_df("data/csvs/SEA(50)_1.csv", class_position="last", pos_class_name="groupA")
        elif self.dsname == "breastw":
            self.data = read_csv_df("data/csvs/BNG(breast-w)_2.csv", class_position="last", pos_class_name="malignant")
        elif self.dsname == "sonar":
            self.data = read_csv_df("data/csvs/BNG(sonar)_3.csv", class_position="last", pos_class_name="Mine")
        elif self.dsname == "heart":
            self.data = read_csv_df("data/csvs/BNG(heart-statlog)_4.csv", class_position="last", pos_class_name="present")
        elif self.dsname == "planes":
            self.data = read_csv_df("data/csvs/2dplanes_7.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "house16":
            self.data = read_csv_df("data/csvs/house_16H_8.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "cal_housing":
            self.data = read_csv_df("data/csvs/cal_housing_9.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "houses":
            self.data = read_csv_df("data/csvs/houses_10.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "house8":
            self.data = read_csv_df("data/csvs/house_8L_11.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "fried":
            self.data = read_csv_df("data/csvs/fried_12.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "letter":
            self.data = read_csv_df("data/csvs/letter_13.csv", class_position="last", pos_class_name="P")
        elif self.dsname == "spectf_test":
            self.data = read_csv_df("data/csvs/BNG(spectf_test)_14.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "austr":
            self.data = read_csv_df("data/csvs/BNG(Australian)_15.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "spectf":
            self.data = read_csv_df("data/csvs/BNG(SPECTF)_16.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "click1":
            self.data = read_csv_df("data/csvs/Click_prediction_small_17.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "click2":
            self.data = read_csv_df("data/csvs/Click_prediction_small_18.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "click3":
            self.data = read_csv_df("data/csvs/Click_prediction_small_19.csv", class_position="first", pos_class_name=1)
        elif self.dsname == "skin":
            self.data = read_csv_df("data/csvs/skin-segmentation_20.csv", class_position="last", pos_class_name=1)
        elif self.dsname == "creditcard":
            self.data = read_csv_df("data/csvs/creditcard_5.csv", class_position="last", pos_class_name=1)
        elif self.dsname == "numerai":
            self.data = read_csv_df("data/csvs/numerai28.6_6.csv", class_position="last", pos_class_name=1)

        if self.dsname in max_class_sizes:
        #    assert self.data[self.data.columns[-1]].value_counts().max() == max_class_sizes[self.dsname]

            self.max_class =  self.data[self.data.columns[-1]].value_counts().max()

        else:
            if self.dsname == "covtype":
                self.data = read_covtype()
            if self.dsname == "adult":
                self.data = read_adult()
            if self.dsname == "letter_old":
                self.data = read_letter()

        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]


    def set_5f_cv_outer(self, nr):
        test_ix = self.get_5f_testindex(self.rows, nr)
        train_ix = self.get_5f_trainindex(self.rows, test_ix)
        self.test_ix = test_ix
        self.train_ix = train_ix
        self.big_test_x = self.data.ix[test_ix, :self.cols - 2].reset_index(drop=True)
        self.big_test_y = self.data.ix[test_ix, self.cols - 1].reset_index(drop=True)


    def set_5f_cv_inner(self, nr):
        cal_ix = self.get_5f_calindex_inner(self.train_ix, nr, self.calibration_size)
        #train_ix, cal_ix = self.get_10f_trainindex(self.train_ix, test_ix)
        train_ix = self.get_5f_trainindex_inner(self.train_ix, cal_ix[max(self.calibration_size)])
        #self.test_x = self.data.ix[test_ix, :self.cols - 2].reset_index(drop=True)
        #self.test_y = self.data.ix[test_ix, self.cols - 1].reset_index(drop=True)
        self.full_train_x = self.data.ix[cal_ix[max(self.calibration_size)] + train_ix, :self.cols - 2].reset_index(drop=True)
        self.full_train_y = self.data.ix[cal_ix[max(self.calibration_size)] + train_ix, self.cols - 1].reset_index(drop=True)
        self.pre_cal_train_x = self.data.ix[train_ix, :self.cols - 2].reset_index(drop=True)
        self.pre_cal_train_y = self.data.ix[train_ix, self.cols - 1].reset_index(drop=True)
        self.cal_train_x = {}
        self.cal_train_y = {}
        for size in self.calibration_size:
            self.cal_train_x[size] = self.data.ix[cal_ix[size], :self.cols - 2].reset_index(drop=True)
            self.cal_train_y[size] = self.data.ix[cal_ix[size], self.cols - 1].reset_index(drop=True)


    def get_full_train_x(self):
        return self.full_train_x.copy()

    def get_full_train_y(self):
        return self.full_train_y.copy()

    def get_pre_cal_train_x(self):
        return self.pre_cal_train_x.copy()

    def get_pre_cal_train_y(self):
        return self.pre_cal_train_y.copy()

    def get_cal_train_x(self, size):
        return self.cal_train_x[size].copy()

    def get_cal_train_y(self, size):
        return self.cal_train_y[size].copy()

    def get_big_test_x(self):
        return self.big_test_x.copy()

    def get_big_test_y(self):
        return self.big_test_y.copy()

    def platts_correction_y(self, y):
        y2 = y.copy()
        y2[y2 == 1] = (y2.sum() + 1) / (y2.sum() + 2)
        y2[y2 == 0] = (1 / (y2.value_counts()[0] + 2))
        return y2


    def group_scores(self, z, y, method="repl"):
        pairs = sorted([Pair(t[0], t[1]) for t in list(zip(z, y))], key=lambda x: (x.sc, x.cl),
                       reverse=False)
        pairs = {i: [pairs[i]] for i in range(len(pairs))}
        z_repl, y_repl, nr0, nr1 = join_pairs_with_same_score(pairs, grouping_method=method)
        return z_repl, y_repl, nr0, nr1

    def sven_correction(self, z_repl, y_repl, prob = 1, nr_of_points = 1):
        if prob == "cr":
            cr = sum(y_repl) / len(y_repl)
            z_sven = [min(z_repl) - 0.1] + z_repl + [max(z_repl) + 0.1]
            y_sven = [cr] * nr_of_points + y_repl + [cr] * nr_of_points
        elif prob == "01":
            z_sven = [min(z_repl) - 0.1, min(z_repl) - 0.1] + z_repl + [max(z_repl) + 0.1, max(z_repl) + 0.1]
            y_sven = [0.5, 0.5] + y_repl + [0.5, 0.5]
        else:
            z_sven = [min(z_repl) - 0.1] + z_repl + [max(z_repl) + 0.1]
            y_sven = [prob] * nr_of_points + y_repl + [1 - prob] * nr_of_points
        return z_sven, y_sven

    def platts_correction(self, z, y):
        y2 = self.platts_correction_y(pd.Series(y))
        return self.group_scores(z, y2)


    @staticmethod
    def get_5f_testindex(rows, cvnr):
        cvnr = cvnr - 1
        test_size = int(np.floor(rows * 0.2))
        start = cvnr * test_size
        end = start + test_size
        return list(range(start, end))

    @staticmethod
    def get_5f_trainindex(rows, testindex):
        train = sorted(list(set(range(rows)).difference(set(testindex))))
        random.seed(testindex[0])
        train = sorted(train)
        return train


    @staticmethod
    def get_5f_calindex_inner(index, cvnr, calibration_size):
        cvnr = cvnr - 1
        test_size = int(np.floor(len(index)  * 0.2))
        start = cvnr * test_size
        end = start + test_size
        cal_ix = {}
        for size in calibration_size:
            i = index[start:end]
            random.seed(size)
            random.shuffle(i)
            i = i[:size]
            i = sorted(i)
            cal_ix[size] = i
        return cal_ix

    @staticmethod
    def get_5f_trainindex_inner(all, calindex, size = 3000):
        train = sorted(list(set(all).difference(set(calindex))))
        random.seed(train[0])
        random.shuffle(train)
        train = train[:size]
        train = sorted(train)
        return train

    # needs updating
    def print_stats(self):
        assert self.full_train_x.shape[0] == self.full_train_y.shape[0]
        assert self.pre_cal_train_x.shape[0] == self.pre_cal_train_y.shape[0]
        assert self.cal_train_x.shape[0] == self.cal_train_y.shape[0]
        #assert self.test_x.shape[0] == self.test_y.shape[0]
        print(self.full_train_y.value_counts())
        print(self.pre_cal_train_y.value_counts())
        print(self.cal_train_y.value_counts())
        #print(self.test_y.value_counts())


def read_csv_df(name, class_position="first", pos_class_name=1):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = pd.read_csv(name, sep=" ")
    df.columns = [i for i in range(len(df.columns))]
    if class_position == "first":
        df = df[list(df.columns[1:]) + [df.columns[0]]]
        df.columns = [i for i in range(len(df.columns))]
    df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 1 if x == pos_class_name else 0)
    df = df.select_dtypes(include=numerics)
    df.columns = [i for i in range(len(df.columns))]
    np.random.seed(1)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)
    return df.head(20000)

