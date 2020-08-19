from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

class Smote(BaseEstimator, TransformerMixin):
    def __init__(self):
           self.smote = SMOTE(SMOTE(kind='regular', n_jobs=-1))

    def fit(self, X, y=None):
           return self

    def transform(self, X, y):
        data = X.copy()
        targets = y.copy()
        data, targets = self.smote.fit_resample(data , targets)
        return data, targets