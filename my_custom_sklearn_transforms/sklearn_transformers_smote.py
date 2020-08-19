from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

import numpy as np

# class Smote(BaseEstimator, TransformerMixin):
#     def __init__(self, X, y):
#         self.smote = SMOTE()

#     def fit(self, X, y):
#         return self

#     def transform(self, X, y):
#         # data = X.copy()
#         # targets = y.copy()
#         # data, targets = self.smote.fit_resample(X, y)
#         # return data, targets
#         return  self.smote.fit_resample(X, y)

class Smote(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        # self.smote = SMOTE(kind='regular', n_jobs=-1).fit(X, y)
        self.smote = SMOTE()

        return self

    # def fit_transform(self, X, y):
    #     # dataX = np.ravel(X)
    #     # dataY = np.ravel(y)
    #     # self.fit(X, y)
    #     return self.smote.fit_resample(X, y)

    def transform(self, X):
        return self.smote.fit_resample(X, y)
