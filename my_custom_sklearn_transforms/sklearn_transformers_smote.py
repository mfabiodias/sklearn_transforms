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

class smote(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        print(X.shape, ' ', type(X)) # (57, 28)   <class 'numpy.ndarray'>
        print(len(y), ' ', type)     #    57      <class 'list'>
        self.smote = SMOTE(kind='regular', n_jobs=-1).fit(X, y)

        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.smote.sample(X, y)

    def transform(self, X):
        return X
