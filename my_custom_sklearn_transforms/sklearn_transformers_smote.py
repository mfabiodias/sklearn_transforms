from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

class Smote(BaseEstimator, TransformerMixin):
    def __init__(self):
           self.smote = SMOTE()

    def fit(self, X, y):
           return self

    def transform(self, X, y):
        data = X.copy()
        targets = y.copy()
        data, targets = self.smote.fit_resample(data , targets)
        return data, targets
