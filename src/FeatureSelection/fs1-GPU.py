import numpy as np
import pandas as pd
import cudf
import cuml
from sklearn.feature_selection import SelectKBest, f_classif

class MicroarrayFeatureSelector:
    """
    Class for performing feature selection on microarray data using SelectKBest and f_classif from scikit-learn.

    Parameters:
    -----------
    k : int or float, optional (default=None)
        Number of features to select. If None, half of the features will be selected.

    gpu : bool, optional (default=False)
        Whether to use GPU acceleration with cuDF and cuML.
    """

    def __init__(self, k=None, gpu=False):
        self.k = k
        self.gpu = gpu

    def fit_transform(self, X, y):
        """
        Perform feature selection on the given data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.

        Returns:
        --------
        X_new : array-like of shape (n_samples, k)
            The transformed input samples with only the selected features.
        """
        if self.gpu:
            X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
            y = cudf.Series(y)
        else:
            X = pd.DataFrame(X)

        n_samples, n_features = X.shape

        if self.k is None:
            self.k = n_features // 2

        if self.gpu:
            X = cuml.preprocessing.StandardScaler().fit_transform(X)
            selector = cuml.feature_selection.SelectKBest(cuml.feature_selection.f_classif, k=self.k)
            X_new = selector.fit_transform(X, y)
            X_new = X_new.to_pandas().values
        else:
            selector = SelectKBest(f_classif, k=self.k)
            X_new = selector.fit_transform(X, y)

        return X_new
