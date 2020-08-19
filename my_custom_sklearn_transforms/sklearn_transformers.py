from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class MySmote(BaseEstimator, TransformerMixin):
    def init(self):
           self.smote = SMOTE(SMOTE(kind='regular', n_jobs=-1))

    def fit(self, X, y=None):
           return self

    def transform(self, X, y):
        data = X.copy()
        targets = y.copy()
        data, targets = self.smote.fit_resample(data , targets)
        return data, targets
