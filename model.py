import random

import numpy as np
import pandas as pd
from sklearn.base import clone

from betacal import BetaCalibration
from clipping_correction import ClippingCorrection
from enir2 import _MyENIRCalibration
from isocal2 import _MyIsotonicCalibration
from newcal import _MyIsotonicCalibration_NEW
from logcal import _MySigmoidCalibration
from scores import accuracy, roc_auc, bs, logloss, top100_perc, top_cr_perc, ece


class Model:
    def __init__(self, id, initial_mode_name, debug = False, only_new = False, only_beta = False, only_log = False):
        self.id = id
        self.initial_model_name = initial_mode_name
        self.pairs = []
        self.models = []
        self.pre_model = None
        self.full_model = None
        self.debug = debug
        self.only_new = only_new
        self.only_beta = only_beta
        self.only_log = only_log
        pass

    def train_initial(self, data, initial_model):
        copy_of_initial_model = clone(initial_model)
        full_model = initial_model.fit(data.get_full_train_x(), data.get_full_train_y())

        pre_cal_model = copy_of_initial_model.fit(data.get_pre_cal_train_x(), data.get_pre_cal_train_y())
        self.pre_model = pre_cal_model
        self.full_model = full_model

        return {"full": full_model, "pre": pre_cal_model}


    def train_calibration_new(self, data, pre_cal_model, size):

        # check if the scores on calibration data are different, otherwise it is not reasonable to calibrate
        if hasattr(pre_cal_model, "predict_proba"):
            preds = pre_cal_model.predict_proba(data.get_cal_train_x(size))
            scores = pd.Series(preds[:, 1]).tolist()
        elif hasattr(pre_cal_model, "decision_function"):
            preds = pre_cal_model.decision_function(data.get_cal_train_x(size))
            if self.only_log:
                scores = pd.Series(preds).tolist()
            else:
                scores = pd.Series(preds).apply(lambda x: 1 / (1 + np.e ** (-1 * x))).tolist()

        if min(scores) == max(scores):
            return None

        # generate different corrections
        y = data.get_cal_train_y(size)
        z = scores
        z_repl, y_repl, no_need1, no_need2 = data.group_scores(z, y, method="repl")
        z_join, y_join, nr0, nr1 = data.group_scores(z, y, method="join")
        z_platt, y_platt, no_need1, no_need2 = data.platts_correction(z, y)



        if self.only_beta:
            cal10 = BetaCalibration(sklearn_lr=False)
            cal10_model = cal10.fit(z_repl, y_repl)
            cal11 = BetaCalibration(sklearn_lr=False)
            cal11_model = cal11.fit(z_platt, y_platt)
            return {"cal10_" + str(size): cal10_model, "cal11_" + str(size): cal11_model}


        if self.only_log:
            cal1 = _MySigmoidCalibration()
            cal1_model = cal1.fit(z_repl, y_repl)
            cal2 = _MySigmoidCalibration()
            cal2_model = cal2.fit(z_platt, y_platt)
            return {"cal1_" + str(size): cal1_model, "cal2_" + str(size): cal2_model}

        # Logistic calibration
        if not self.only_new:
            cal1 = _MySigmoidCalibration()
            cal1_model = cal1.fit(z_repl, y_repl)
            cal2 = _MySigmoidCalibration()
            cal2_model = cal2.fit(z_platt, y_platt)
        # cal35_model = ClippingCorrection(cal1_model, 0.95)
        # cal36_model = ClippingCorrection(cal1_model, 0.99)
        # cal37_model = ClippingCorrection(cal1_model, 0.999)

        # Isotonic calibration
        cal4 = _MyIsotonicCalibration()
        cal4_model = cal4.fit(z_repl, y_repl)
        if not self.only_new:
            cal5 = _MyIsotonicCalibration()
            cal5_model = cal5.fit(z_platt, y_platt)
            cal7 = _MyIsotonicCalibration_NEW(distr=8, kind=2)
            cal7_model = cal7.fit(z_join, y_join, z_repl, y_repl, nr0=nr0, nr1=nr1)
            if self.debug:
                cal8 = _MyIsotonicCalibration_NEW(distr=9, kind=2)
                cal8_model = cal8.fit(z_join, y_join, z_repl, y_repl, nr0=nr0, nr1=nr1)
        else:
            cal8 = _MyIsotonicCalibration_NEW(distr=9, kind=2)
            cal8_model = cal8.fit(z_join, y_join, z_repl, y_repl, nr0=nr0, nr1=nr1)
        #cal65_model = ClippingCorrection(cal4_model, 0.95)
        #cal66_model = ClippingCorrection(cal4_model, 0.99)
        #cal67_model = ClippingCorrection(cal4_model, 0.999)

        # ENIR
        if not self.debug:
            if not self.only_new:
                cal8 = _MyENIRCalibration(self.id + "_" + str(size) + "_1", seed = random.randint(1, 10000))
                cal8_model = cal8.fit(z_repl, y_repl)
                cal9 = _MyENIRCalibration(self.id + "_" + str(size) + "_2", seed = random.randint(1, 10000))
                cal9_model = cal9.fit(z_platt, y_platt)
        # cal95_model = ClippingCorrection(cal8_model, 0.95)
        # cal96_model = ClippingCorrection(cal8_model, 0.99)
        # cal97_model = ClippingCorrection(cal8_model, 0.999)

        # Betacal
        if not self.debug:
            if not self.only_new:
                cal10 = BetaCalibration(sklearn_lr=False)
                cal10_model = cal10.fit(z_repl, y_repl)
                cal11 = BetaCalibration(sklearn_lr=False)
                cal11_model = cal11.fit(z_platt, y_platt)
        # cal125_model = ClippingCorrection(cal10_model, 0.95)
        # cal126_model = ClippingCorrection(cal10_model, 0.99)
        # cal127_model = ClippingCorrection(cal10_model, 0.999)


        if self.only_new:
            return {"cal4_" + str(size): cal4_model, "cal8_" + str(size): cal8_model}
        if self.debug:
            return {"cal1_" + str(size): cal1_model, "cal2_" + str(size): cal2_model, "cal4_" + str(size): cal4_model,
                    "cal5_" + str(size): cal5_model, "cal6_" + str(size): cal6_model, "cal7_" + str(size): cal7_model,
                    "cal8_" + str(size): cal8_model}

        return {"cal1_" + str(size): cal1_model, "cal2_" + str(size): cal2_model, "cal4_" + str(size): cal4_model,
                "cal5_" + str(size): cal5_model,"cal7_" + str(size): cal7_model,
                "cal8_" + str(size): cal8_model, "cal9_" + str(size): cal9_model,
                "cal10_" + str(size): cal10_model, "cal11_" + str(size): cal11_model}


    def test_full(self, model, x, y):
        if self.initial_model_name == "SVM":
            pred_scores = model.decision_function(x)
            scores = pd.Series(pred_scores).apply(lambda x: 1 / (1 + np.e ** (-1 * x))).tolist()
            score_df = pd.DataFrame({"score": pd.Series(scores), "cl": y})
        else:
            pred_scores = model.predict_proba(x)
            score_df = pd.DataFrame({"score": pd.Series(pred_scores[:, 1]), "cl": y})
        return score_df

    def test_cal_new(self, model, x, y, initial_model, cal_name):
        if hasattr(initial_model, "predict_proba"):
            preds = initial_model.predict_proba(x)
            scores = pd.Series(preds[:, 1]).tolist()
        elif hasattr(initial_model, "decision_function"):
            preds = initial_model.decision_function(x)
            if self.only_log:
                scores = pd.Series(preds).tolist()
            else:
                scores = pd.Series(preds).apply(lambda x: 1 / (1 + np.e ** (-1 * x))).tolist()

        pred_scores = model.predict(scores)
        score_df = pd.DataFrame({"score": pd.Series(pred_scores), "cl": y})

        return score_df


    def predict(self, data, models, pre_model = None, size = None):
        preds = {}
        for model in models:
            if "pre" in model:
                pre_pred_train = self.test_full(models[model], data.get_pre_cal_train_x(), data.get_pre_cal_train_y())
                pre_pred_big_test =  self.test_full(models[model], data.get_big_test_x(), data.get_big_test_y())
                preds[model + "_train"] = pre_pred_train
                preds[model + "_big"] = pre_pred_big_test
            if "cal" in model:
                cal_pred_train = self.test_cal_new(models[model], data.get_cal_train_x(size), data.get_cal_train_y(size), pre_model, model)
                cal_pred_big_test = self.test_cal_new(models[model], data.get_big_test_x(), data.get_big_test_y(), pre_model, model)
                preds[model + "_train"] = cal_pred_train
                preds[model + "_big"] = cal_pred_big_test
            else:
                full_pred_train = self.test_full(models[model], data.get_full_train_x(), data.get_full_train_y())
                cal_pred_big_test = self.test_full(models[model], data.get_big_test_x(), data.get_big_test_y())
                preds[model + "_train"] = full_pred_train
                preds[model + "_big"] = cal_pred_big_test
        return preds

    def evaluate(self, preds):
        res = {}
        for ds in preds:
            try:
                res[ds] = {}
                res[ds]["acc"] = accuracy(preds[ds])
                res[ds]["roc_auc"] = roc_auc(preds[ds])
                res[ds]["bs"] = bs(preds[ds])
                #res[ds]["logloss"] = logloss(preds[ds])
                res[ds]["loglosse"] = logloss(preds[ds], epsilon=True)
                res[ds]["ece"], res[ds]["ece0"], res[ds]["ece15"] = ece(preds[ds])
                top = top100_perc(preds[ds])
                top_cr = top_cr_perc(preds[ds])
                res[ds]["sm_top"] = top["sm_top"]
                res[ds]["cm_top"] = top["cm_top"]
                res[ds]["sm_bottom"] = top["sm_bottom"]
                res[ds]["cm_bottom"] = top["cm_bottom"]
                res[ds]["sm_top_cr"] = top_cr["sm_top"]
                res[ds]["cm_top_cr"] = top_cr["cm_top"]
                res[ds]["sm_bottom_cr"] = top_cr["sm_bottom"]
                res[ds]["cm_bottom_cr"] = top_cr["cm_bottom"]
            except Exception as e:
                print(e)
                print(preds[ds].head())
                print(self.initial_model_name, ds)
                raise Exception
        return res

    def run(self, data, initial_model):
        res = {}
        all_preds = {}
        all_models = {}
        for cv5o in range(1, 6):
            print(cv5o)
            data.set_5f_cv_outer(cv5o)
            res[cv5o] = {}
            all_preds[cv5o] = {}
            all_models[cv5o] = {}
            for cv5i in range(1, 6):
                print(cv5i, end=',')
                data.set_5f_cv_inner(cv5i)

                no_cal_models = self.train_initial(data, initial_model)
                models = {}
                preds = {}
                scores = {}

                preds["no_cal"] = self.predict(data, no_cal_models)
                scores["no_cal"] = self.evaluate(preds["no_cal"])

                for size in data.calibration_size:
                    models[size] = self.train_calibration_new(data, no_cal_models["pre"], size)

                    if models[size] is not None:
                        preds[size] = self.predict(data, models[size], no_cal_models["pre"], size)
                        scores[size] = self.evaluate(preds[size])
                    else:
                        preds[size] = None
                        scores[size] = None




                res[cv5o][cv5i] = scores
                all_preds[cv5o][cv5i] = preds
                all_models[cv5o][cv5i] = models
                if self.debug:
                    break
            if self.debug:
                break
            print("")
        self.preds = all_preds
        self.models = all_models
        return self.average(res)

    def average(self, scores):
        res = {}
        for i in range(1, 6):
            for j in range(1, 6):
                for size_class in scores[i][j]:
                    if scores[i][j][size_class] is None:
                        continue
                    for pred in scores[i][j][size_class]:
                        if pred not in res:
                            res[pred] = {}
                        for sc in scores[i][j][size_class][pred]:
                            if sc not in res[pred]:
                                res[pred][sc] = [scores[i][j][size_class][pred][sc]]
                            else:
                                res[pred][sc].append(scores[i][j][size_class][pred][sc])
                if self.debug:
                    break
            if self.debug:
                break

        res_mean = {}
        for pred in res:
            res_mean[pred] = {}
            for sc in res[pred]:
                res_mean[pred][sc] = sum(res[pred][sc]) / len(res[pred][sc])
        return res, res_mean



