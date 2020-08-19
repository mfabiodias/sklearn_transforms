from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

# class Smote(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.smote = SMOTE()

#     def fit(self, X, y):
#         return self

#     def transform(self, X, y):
#         data = X.copy()
#         targets = y.copy()
#         data, targets = self.smote.fit_resample(data, targets)
#         return data, targets

class Smote(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print(X.shape, ' ', type(X)) # (57, 28)   <class 'numpy.ndarray'>
        print(len(y), ' ', type)     #    57      <class 'list'>
        self.smote = SMOTE()

        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.smote.fit_resample(X, y)

    def transform(self, X):
        return X
