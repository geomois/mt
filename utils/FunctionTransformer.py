from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

class FunctionTransformerWithInverse(BaseEstimator, TransformerMixin):
    def __init__(self, func=None, inv_func=None, validate=False,
                 accept_sparse=False, pass_y=False):
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y
        self.func = func
        self.inv_func = inv_func

    def fit(self, X, y=None):
        if self.validate:
            check_array(X, self.accept_sparse)
        return self

    def transform(self, X, y=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.func is None:
            return X
        return self.func(X)

    def inverse_transform(self, X, y=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.inv_func is None:
            return X
        return self.inv_func(X)

