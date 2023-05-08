"""
NOTE: This code does not work on mac os, hence they do not support NVIDIA
"""

import numpy as np
import pandas as pd
import cupy as cp
import cudf

class SPFSR_GPU:
    def __init__(self, estimator, k=10, max_iter=100, tol=1e-4, gpu=False):
        self.estimator = estimator
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.gpu = gpu
    
    def _parabolic_interpolation(self, X, y, selected_features):
        n_samples = X.shape[0]
        indices = list(selected_features)
        x1, x2, x3 = [X[:, i] for i in cp.random.choice(indices, size=3, replace=False)]
        f1, f2, f3 = [y[cp.argmin(x)] for x in [x1, x2, x3]]
        for i in range(self.max_iter):
            # Perform parabolic interpolation
            numerator = (x2 - x1) * (x2 - x3) * (f2 - f3) + (x2 - x3) * (x3 - x1) * (f2 - f1) + (x3 - x1) * (x1 - x2) * (f3 - f1)
            denominator = (x2 - x1) * (x2 - x3) * (x3 - x1)
            x = 0.5 * (x1 + x2 - numerator / denominator)
            # Check for convergence
            if cp.abs(x - x2) < self.tol:
                break
            # Evaluate function at new point
            if self.gpu:
                f = y[cp.argmin(cp.asnumpy(cp.abs(X[:, indices] - x)), axis=0)]
            else:
                f = y[np.argmin(np.abs(X[:, indices] - x), axis=0)]
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
        if self.gpu:
            X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
            y = cudf.Series(y)
        else:
            X = pd.DataFrame(X)

        n_samples, n_features = X.shape
        selected_features = set(range(n_features))
        scores = np.zeros(n_features)
        support = np.zeros(n_features, dtype=bool)
        k = min(self.k, n_features)

        for i in range(k):
            scores.fill(0)
            X_selected = X.iloc[:, list(selected_features)]
            for j in range(n_features):
                if j not in selected_features:
                    X_temp = X_selected.copy()
                    X_temp.insert(len(X_temp.columns), str(j), X.iloc[:, j])
                    if self.gpu:
                        X_temp = cp.asarray(X_temp.values)
                    else:
                        X_temp = X_temp.values
                    score = self.estimator.score(X_temp, y)
                    scores[j] = score

            # Determine the feature with the minimum score
            if self.gpu:
                j_min = cp.argmin(scores)
            else:
                j_min = np.argmin(scores)

            # Update selected features and support
            selected_features.add(j_min)
            selected_features.discard(j)
            support[j_min] = True
            if self.verbose:
                print(f"Selected feature {i+1}: {j_min}, score: {scores[j_min]:.4f}")

        self.support_ = support
        return self