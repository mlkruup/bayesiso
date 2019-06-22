import pandas as pd
import numpy as np
import pickle
import argparse

#parser = argparse.ArgumentParser(description='Combine results.')

#parser.add_argument('--dataset', dest='dataset', help='dataset number', type=int)

#args = parser.parse_args()


def get_data(data, dataset, model, method, score):
    return data[dataset][model][method][score]


def method_map(name):
    d = {"cal1": "Log", "cal2": "Log-Platt", "cal4": "Iso",
        "cal5": "Iso-Platt", "cal7": "Bayes-Iso", "cal8": "ENIR",
         "cal9": "ENIR-Platt", "cal10": "Beta", "cal11": "Beta-Platt", "full": "init", "pre": "pre"}
    for x in d:
        if x + "_" in name:
            return d[x]


def aggregate_scores(scores):
    if scores is None:
        return None, None, 0
    else:
        return np.mean(scores), np.median(scores), np.std(scores), len(scores)


def check_reverse_sigmoid(scores, pre_scores):
    new_scores = []
    changed = []
    different = []
    for i in range(len(scores)):
        if abs(scores[i] - pre_scores[i]) < 0.0001:
            new_scores.append(scores[i])
            changed.append(False)
            different.append(False)
        elif abs((1 - scores[i]) - pre_scores[i]) < 0.0001:
            new_scores.append(1 - scores[i])
            changed.append(True)
            different.append(False)
        else:
            new_scores.append(scores[i])
            changed.append(False)
            different.append(True)
    return new_scores, changed, different


def fill_dataset_mean(res, datasets, scores, methods, types, models, sizes):
    l = []
    l_mean = []
    for dataset in datasets:
        print(dataset)
        for score in scores:
            print(score)
            for method in methods:
                for t in types:
                    for model in models:
                        for size in sizes:
                            if method in ["full_", "pre_"]:
                                name = method + t
                            else:
                                name = method + size + "_" + t
                            try:
                                d = get_data(res, dataset, model, name, score)
                                changed = [False for i in range(25)]
                                different = [False for i in range(25)]
                                d_new = [None for i in range(25)]
                                if method in ["cal2_", "cal12_", "cal13_"] and score == "roc_auc":
                                    d_pre = get_data(res, dataset, model, "pre_" + t, score)
                                    d_new, changed, different = check_reverse_sigmoid(d, d_pre)
                                    mean, median, sd, count = aggregate_scores(d_new)
                                mean, median, sd, count = aggregate_scores(d)
                            except:
                                print(dataset, score, method, t, model, size, name)
                                raise Exception
                            l_mean.append([dataset, score, method.strip("_"), method_map(method),
                                           t, model, size, mean, median, sd, count, sum(changed), sum(different)])

                            # for the cv file
                            for i in range(len(d)):
                                l.append([dataset, score, method.strip("_"), method_map(method),
                                          t, model, size, i, d[i], d_new[i], changed[i], different[i]])

    df_mean = pd.DataFrame(l_mean,
                           columns=["data", "score", "method", "method_name", "type", "model", "size", "mean", "median", "sd",
                                    "count",
                                    "auc_changed", "auc_different"])
    df = pd.DataFrame(l, columns=["data", "score", "method", "method_name", "type", "model", "size", "cv", "value",
                                  "changed_value", "auc_changed", "auc_different"])

    return df, df_mean



datasets = ["sea", "breastw", "sonar", "heart", "planes", "house16", "cal_housing", "houses", "house8",
            "fried", "spectf_test", "austr", "spectf", "skin", "numerai"]
scores = ["acc", "bs", "roc_auc", "loglosse", "ece", "sm_top", "cm_top", "sm_bottom", "cm_bottom", "sm_top_cr", "cm_top_cr",
          "sm_bottom_cr", "cm_bottom_cr"]
methods = ["full_", "pre_", "cal1_", "cal2_", "cal4_", "cal5_", "cal7_", "cal8_", "cal9_", "cal10_", "cal11_"]
types = ["train", "big"]
models = ["SVM", "ANN", "NB", "DT", "RF", "KNN", "LOGREG", "BAG-DT", "ADA"]
sizes = ["100", "1000", "3000"]


res = {}
for dataset in datasets:
    with open("results/res_3000_%s.pickle" % dataset, 'rb') as pickle_file:
        temp = pickle.load(pickle_file)
        res[list(temp.keys())[0]] = temp[list(temp.keys())[0]]

df, df_mean = fill_dataset_mean(res, datasets, scores, methods, types, models, sizes)
df.to_csv("results/cv_results_correction.csv")
df_mean.to_csv("results/mean_results_correction.csv")
