import pandas as pd
import numpy as np
import subprocess
import numbers

class _MyENIRCalibration:
    def __init__(self, id = "whaat", seed = None, local = False):
        self.id = id
        self.seed = seed
        self.local = local

    def fit(self, z, y):
        self.model = self.ENIR_model(z, y)
        return self

    def ENIR_model(self, scores, classes):
        pd.DataFrame({"z": scores, "y": classes, "control": [self.seed for s in scores]}).to_csv("temp/enir_input_%s.csv" % self.id, index=False)
        # retcode = subprocess.call(["RScript","create_enir_model.R"])
        if self.local:
            retcode = subprocess.call(["/usr/local/bin/Rscript", "create_enir_model.R", self.id])
        else:
            retcode = subprocess.call(["Rscript", "create_enir_model.R", self.id])
        assert retcode == 0
        return 1

    def predict(self, z):
        pd.Series(z).to_csv("temp/enir_pred_input_%s.csv" % self.id, index=False)
        if self.local:
            retcode = subprocess.call(["/usr/local/bin/Rscript", "predict_enir.R", self.id])
        else:
            retcode = subprocess.call(["Rscript", "predict_enir.R", self.id]) # /usr/local/bin/
        assert retcode == 0
        #if retcode != 0:
        #    try:
        #        subprocess.check_output(["/usr/local/bin/Rscript", "predict_enir.R"])
        #    except subprocess.CalledProcessError as e:
        #        print(e)
        #        1 / 0
        res = pd.read_csv("temp/enir_output_%s.csv" % self.id, header=None)

        control = res[0][0]

        assert int(control) == self.seed

        _1scores = res[0][1:]


        _0scores = [1 - x for x in _1scores]

        return np.array(_1scores)
        #return np.array(list(zip(_0scores, _1scores)))