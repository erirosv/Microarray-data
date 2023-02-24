import numpy as np
import pandas as pd

def SPFSR_feature_selection(X, y, k, tol=1e-6):
    """
    Implements the Successive Parabolic Interpolation-Fibonacci Search-Ridder's Method (SPFSR) algorithm for feature
    selection using univariate t-tests. The algorithm selects k features that have the highest univariate t-test
    statistics with respect to the target variable.
    
    X: the input feature matrix, where each row represents an instance and each column represents a feature
    y: the target variable vector, where each element corresponds to the target value for the corresponding instance
    k: the number of features to select
    tol: the tolerance for the minimum
    
    Returns the indices of the k selected features.
    """
    # Define the golden ratio
    golden_ratio = (np.sqrt(5) - 1) / 2
    
    # Compute the univariate t-test statistics for each feature
    t_values, _ = np.apply_along_axis(lambda col: np.abs(np.mean(col[y == 1]) - np.mean(col[y == 0])) / np.sqrt(
        np.var(col[y == 1]) / np.sum(y == 1) + np.var(col[y == 0]) / np.sum(y == 0)), axis=0, arr=X.values)
    
    # Find the k features with the highest t-test statistics
    feature_indices = np.argsort(t_values)[::-1][:k]
    
    # Define the initial points for the Fibonacci search
    x = np.array([golden_ratio * (len(feature_indices) - 1), (1 - golden_ratio) * (len(feature_indices) - 1)])
    
    # Evaluate the function at the initial points
    f_x = -np.inf * np.ones_like(x)
    for i, x_i in enumerate(x):
        if x_i >= 0 and x_i < len(feature_indices) - 1:
            features = feature_indices[[int(np.floor(x_i)), int(np.ceil(x_i))]]
            f_x[i] = np.abs(np.corrcoef(X.iloc[:, features].T, y)[0, 1])
    
    # Keep track of the best function value so far
    f_best = np.max(f_x)
    
    # Define the initial points for the parabolic interpolation
    x1, x2, x3 = 0, np.mean(x), len(feature_indices) - 1
    f1, f2, f3 = f_x[0], f_x[1], f_x[-1]
    
    while abs(x3 - x1) > tol:
        # Determine the minimum point using parabolic interpolation
        x_min = x2 + (f2 - f1) * (x2 - x3) / ((f2 - f3) * (x2 - x1) - (f2 - f1) * (x2 - x3))
        
        # Evaluate the function at the minimum point
        if x_min >= 0 and x_min < len(feature_indices) - 1:
            features = feature_indices[[int(np.floor(x_min)), int(np.ceil(x_min))]]
            f_min = np.abs(np.corrcoef(X.iloc[:, features].T, y)[0, 1])
        else:
            f_min = -np.inf
        
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
        
    # Return the indices of the k selected features
    return feature_indices[[int(np.floor(x2)), int(np.ceil(x2))]]
