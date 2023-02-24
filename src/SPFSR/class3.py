import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SPFSR(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, max_iter=10, tol=1e-5):
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.n_features >= n_features:
            raise ValueError("Number of selected features must be less than the total number of features")

        # Initialize set of selected features with the first n_features indices
        selected_features = set(range(self.n_features))

        while len(selected_features) < n_features:
            # Find the point with minimum function value using parabolic interpolation
            x, f = self._parabolic_interpolation(X, y, selected_features)

            # Add the new index to the set of selected features
            selected_features.add(np.argmin(X[:, list(selected_features)] - x))

        # Store the indices of the selected features
        self.selected_indices_ = list(selected_features)

        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def _parabolic_interpolation(self, X, y, selected_features):
        indices = list(selected_features)
        x1, x2, x3 = X[:, np.random.choice(indices, size=3, replace=False)]
        f1, f2, f3 = [y[np.argmin(x)] for x in [x1, x2, x3]]
        for i in range(self.max_iter):
            # Perform parabolic interpolation
            numerator = (x2 - x1) * (x2 - x3) * (f2 - f3) + (x2 - x3) * (x3 - x1) * (f2 - f1) + (x3 - x1) * (x1 - x2) * (f3 - f1)
            denominator = (x2 - x1) * (x2 - x3) * (x3 - x1)
            x = 0.5 * (x1 + x2 - numerator / denominator)
            # Check for convergence
            if np.abs(x - x2) < self.tol:
                break
            # Evaluate function at new point
            f = y[np.argmin(X[:, indices] - x, axis=0)]
            # Update points for next iteration
            if f2 < f3:
                x1, x3 = x3, x2
                x2 = x
                f1, f3 = f3, f2
                f2 = f
            else:
                x2, x3 = x3, x1
                x1 = x
                f2, f1 = f1, f

        return x, f
