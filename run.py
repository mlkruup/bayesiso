from data import Data
from model import Model
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import tree
import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description='Run analysis pipeline.')

parser.add_argument('--size', dest='size', help='size for the training set', type=int)
parser.add_argument('--dataset', dest='dataset', help='dataset number', type=int)

args = parser.parse_args()
datasets = ["sea", "breastw", "sonar", "heart", "planes", "house16", "cal_housing", "houses", "house8",
            "fried", "letter", "spectf_test", "austr", "spectf", "click1", "click2", "click3", "skin",
            "creditcard", "numerai"]


def run(size, dataset):
    init_models = ["SVM", "ANN", "NB", "LOGREG", "ADA", "RF", "DT", "KNN", "BAG-DT"]
    all_res = {}
    datasets = [dataset]
    for ds in datasets:
        print(ds)
        d = Data(ds, size)
        all_res[ds] = {}
        for im in init_models:
            print(im)
            m = Model(ds, im, only_beta=False, only_log=False)
            try:
                if im == "DT":
                    res, res_mean = m.run(d, tree.DecisionTreeClassifier(min_samples_leaf=10))
                elif im == "NB":
                    res, res_mean = m.run(d, GaussianNB())
                elif im == "SVM":
                    res, res_mean = m.run(d, SVC()) # decision_function
                elif im == "RF":
                    res, res_mean = m.run(d, RandomForestClassifier())
                elif im == "KNN":
                    res, res_mean = m.run(d, KNeighborsClassifier())
                elif im == "LOGREG":
                    res, res_mean = m.run(d, LogisticRegression()) # decision_function (predict_proba exists)
                elif im == "BAG-DT":
                    res, res_mean = m.run(d, BaggingClassifier(tree.DecisionTreeClassifier())) # decision_function (predict_proba exists)
                elif im == "ADA":
                    res, res_mean = m.run(d, AdaBoostClassifier()) # decision_function (predict_proba exists)
                elif im == "ANN":
                    res, res_mean = m.run(d, MLPClassifier())
            except:
                print("Didn't manage with model " + im)
                continue
            sys.stdout.flush()
            with open('results/models/may30_%s_%s.pkl' % (ds, im), 'wb') as output:
                pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)
            all_res[ds][im] = res
    return all_res


if __name__ == "__main__":
    size = int(args.size)
    dataset = int(args.dataset) - 1
    res = run(size, datasets[dataset])
    pickle.dump(res, open("results/res_%s_%s.pickle" % (size, datasets[dataset]), "wb"))
