import pandas as pd

class ClippingCorrection:

    def __init__(self, calibrator, threshold):
        self.calibrator = calibrator
        self.threshold = threshold


    def predict(self, scores):
        pred_scores = pd.Series(self.calibrator.predict(scores))
        pred_scores[pred_scores >= self.threshold] = self.threshold
        pred_scores[pred_scores <= (1 - self.threshold)] = 1 - self.threshold
        return pred_scores

