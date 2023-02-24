import numpy as np
import pandas as pd

class SPFSR:
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        n_features = X.shape[1]
        feature_indices = np.arange(n_features)
        x1, x3 = 0, n_features - 1
        x2 = (x1 + x3) / 2
        
        f = lambda x: self._objective_function(x, X, y)
        
        f1, f3 = f(x1), f(x3)
        f2 = f(x2)
        
        f_best = f2
        
        for i in range(n_features - self.k):
            # Use Fibonacci search to find the next point for parabolic interpolation
            d = (x3 - x1) / self._fibonacci(i + 2)
            x_min = x1 + d
            f_min = f(x_min)
            
            if x_min < x2:
                x1, x2 = x_min, x2 - d
                f1, f2 = f_min, f(x2)
            elif x_min > x2:
                x2, x3 = x_min, x2 + d
                f2, f3 = f_min, f(x2)
            else:
                x1, x3 = x2, x_min
                f1, f3 = f2, f_min
                x2 = (x1 + x3) / 2
                f2 = f(x2)
            
            # Determine the points to retain for the next iteration
            if x_min > x2:
                if f_min < f2:
                    x1, f1 = x2, f2
                    x2, f2 = x_min, f_min
                else:
                    x3, f3 = x_min, f_min
            else:
                if f_min < f2:
                    x3, f3 = x2, f2
                    x2, f2 = x_min, f_min
                else:
                    x1, f1 = x_min, f_min

            # Update the best function value so far
            if f2 > f_best:
                f_best = f2
        
        # Save the indices of the k selected features
        self.feature_indices_ = feature_indices[self._get_selected_indices(x2)]
    
    def _objective_function(self, x, X, y):
        """
        Computes the objective function value for a given feature subset x.
        """
        X_subset = X[:, x.astype(bool)]
        beta = np.linalg.lstsq(X_subset, y, rcond=None)[0]
        y_pred = X_subset @ beta
        return np.mean((y - y_pred) ** 2)
    
    def _fibonacci(self, n):
        """
        Computes the nth Fibonacci number.
        """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _get_selected_indices(self, x):
        """
        Returns the indices of the selected features based on the interpolated value of x.
        """
        return [int(np.floor(x)), int(np.ceil(x))]
    
    def transform(self, X):
        """
        Returns the data matrix X with only the selected features.
        """
        return X[:, self.feature_indices_]
