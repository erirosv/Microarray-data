from sklearn.feature_selection import SelectKBest, f_classif

class MicroarrayFeatureSelector:
    """
    Class for performing feature selection on microarray data using SelectKBest and f_classif from scikit-learn.

    Parameters:
    -----------
    k : int or float, optional (default=None)
        Number of features to select. If None, half of the features will be selected.
    """

    def __init__(self, k=None):
        self.k = k

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
        selector = SelectKBest(f_classif, k=self.k)
        X_new = selector.fit_transform(X, y)
        return X_new

# USAGE
# selector = MicroarrayFeatureSelector(k=1000)
# X_selected = selector.fit_transform(X_train, y_train)
