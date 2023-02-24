import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SPFSR(BaseEstimator, TransformerMixin):
    
    def __init__(self, k=10, max_iter=100, tol=1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.selected_features_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.k > n_features:
            self.k = n_features
        # Initialize feature set with all features
        selected_features = set(range(n_features))
        x_min, f_min = self._parabolic_interpolation(X, y, selected_features)
        for i in range(self.max_iter):
            # Randomly choose a feature to exclude
            excluded_feature = np.random.choice(list(selected_features))
            candidate_features = selected_features - {excluded_feature}
            # Perform parabolic interpolation on candidate features
            x, f = self._parabolic_interpolation(X, y, candidate_features)
            # Check for convergence
            if np.abs(x_min - x) < self.tol:
                break
            # Determine the points to retain for the next iteration
            if x_min > x:
                if f_min < f:
                    selected_features.remove(excluded_feature)
                    x_min, f_min = x, f
                else:
                    selected_features = candidate_features
                    x_min, f_min = x, f
            else:
                if f < f_min:
                    selected_features.add(excluded_feature)
                    x_min, f_min = x, f
        self.selected_features_ = list(selected_features)[:self.k]

    def transform(self, X):
        return X[:, self.selected_features_]

    def _parabolic_interpolation(self, X, y, selected_features):
        n_samples = X.shape[0]
        indices = list(selected_features)
        x1, x2, x3 = [X[:, i] for i in np.random.choice(indices, size=3, replace=False)]
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

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.indices = np.random.choice(X.shape[1], size=self.num_features, replace=False)
        selected_features = set(self.indices)
        n_samples = X.shape[0]
        x1, x2, x3 = [X[:, i] for i in np.random.choice(self.indices, size=3, replace=False)]
        f1, f2, f3 = [y[np.argmin(x)] for x in [x1, x2, x3]]
        for i in range(self.max_iter):
            numerator = (x2 - x1) * (x2 - x3) * (f2 - f3) + (x2 - x3) * (x3 - x1) * (f2 - f1) + (x3 - x1) * (x1 - x2) * (f3 - f1)
            denominator = (x2 - x1) * (x2 - x3) * (x3 - x1)
            x = 0.5 * (x1 + x2 - numerator / denominator)
            if np.abs(x - x2) < self.tol:
                break
            f = y[np.argmin(X[:, self.indices] - x, axis=0)]
            if f2 < f3:
                x1, x3 = x3, x2
                x2 = x
                f1, f3 = f3, f2
                f2 = f
            else:
                x2, x3 = x3, x1
                x1 = x
                f2, f1 = f1, f
            
            # Determine the points to retain for the next iteration
            if x < x2:
                if f < f2:
                    selected_features.add(self.indices[0])
                    self.indices[0] = np.argmin(X[:, self.indices[0]])
                else:
                    selected_features.discard(self.indices[2])
                    self.indices[2] = np.argmax(X[:, self.indices[2]])
            else:
                if f < f2:
                    selected_features.add(self.indices[2])
                    self.indices[2] = np.argmin(X[:, self.indices[2]])
                else:
                    selected_features.discard(self.indices[0])
                    self.indices[0] = np.argmax(X[:, self.indices[0]])
        return self
    
    def transform(self, X):
        return X[:, self.indices]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)