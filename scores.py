from sklearn import metrics
import numpy as np
import pandas as pd


def accuracy(df, score_th=0.5):
    n = df.shape[0]
    match = (df["cl"] == df["score"].apply(lambda x: 1 if x >= score_th else 0)).sum()
    return match / n


def roc_auc(df):
    y = df["cl"]
    pred = df["score"]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def bs(df):
    n = df.shape[0]
    s = (df["score"] - df["cl"]).pow(2).sum()
    return s / n


def logloss(df, epsilon=False):
    # https://www.kaggle.com/wiki/LogarithmicLoss
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    n = df.shape[0]
    if epsilon:
        epsilon = 1e-15
    else:
        epsilon = 0
    s = (df["cl"] * np.log(np.minimum(1 - epsilon, np.maximum(df["score"], epsilon)))
        + (1 - df["cl"]) * np.log(1 - np.minimum(1 - epsilon, np.maximum(df["score"], epsilon)))).multiply(-1).sum()
    return s / n


def ece(df, bins = 15):
    df["cat"] = pd.cut(df.score, np.linspace(0, 1, (bins + 1)))
    df2 = df.groupby("cat").agg(["mean", "count"])
    df2.columns = df2.columns.droplevel()
    df2.columns = ["mean1", "count1", "mean2", "count2"]
    df2["prop"] = df2["count1"] / df.shape[0]
    ece_vals = ((df2["mean1"] - df2["mean2"]).abs() * df2["prop"])
    ece_0 = ece_vals[0]
    ece_15 = ece_vals[-1]
    ece = ece_vals.sum()
    assert(ece >= 0)
    return ece, ece_0, ece_15

def top100_perc(df):
    subdf_top = df.sort_values("score", ascending=False).copy().head(100)
    subdf_bottom = df.sort_values("score", ascending=True).copy().head(100)
    sm_top = subdf_top.score.mean()
    cm_top = subdf_top.cl.mean()
    sm_bottom = subdf_bottom.score.mean()
    cm_bottom = subdf_bottom.cl.mean()
    return {"sm_top": sm_top, "cm_top": cm_top, "sm_bottom": sm_bottom, "cm_bottom": cm_bottom}

def top_cr_perc(df):
    cr_pos = df["cl"].sum()
    cr_neg = df.shape[0] - cr_pos
    subdf_top = df.sort_values("score", ascending=False).copy().head(cr_pos)
    subdf_bottom = df.sort_values("score", ascending=True).copy().head(cr_neg)
    sm_top = subdf_top.score.mean()
    cm_top = subdf_top.cl.mean()
    sm_bottom = subdf_bottom.score.mean()
    cm_bottom = subdf_bottom.cl.mean()
    return {"sm_top": sm_top, "cm_top": cm_top, "sm_bottom": sm_bottom, "cm_bottom": cm_bottom}
