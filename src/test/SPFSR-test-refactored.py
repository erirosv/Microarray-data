import numpy as np

def SPFSR(f, a, b, tol=1e-6):
    """
    Implements the Successive Parabolic Interpolation-Fibonacci Search-Ridder's Method (SPFSR) algorithm to find the
    minimum of a unimodal function within the interval [a, b].
    
    f: the unimodal function to minimize
    a, b: the interval to search for the minimum within
    tol: the tolerance for the minimum
    
    Returns the approximate minimum of the function.
    """
    # Define the golden ratio
    golden_ratio = (np.sqrt(5) - 1) / 2
    
    # Define the initial points for the Fibonacci search
    x = np.array([a + golden_ratio * (b - a), a + (1 - golden_ratio) * (b - a)])
    
    # Evaluate the function at the initial points
    f_x = f(x)
    
    # Keep track of the best function value so far
    f_best = np.min(f_x)
    
    # Define the initial points for the parabolic interpolation
    x1, x2, x3 = a, np.mean(x), b
    f1, f2, f3 = f(x1), f(x2), f(x3)
    
    while abs(b - a) > tol:
        # Determine the minimum point using parabolic interpolation
        x_min = x2 + (f2 - f1) * (x2 - x3) / ((f2 - f3) * (x2 - x1) - (f2 - f1) * (x2 - x3))
        
        # Evaluate the function at the minimum point
        f_min = f(x_min)
        
        # Determine the points to retain for the next iteration
        if x_min > x2:
            if f_min < f2:
                x1, x2, x3 = x2, x_min, 2 * x2 - x_min
                f1, f2, f3 = f2, f_min, f(x3)
            else:
                x1, x2, x3 = x1, (x2 + x1) / 2, x2
                f1, f2, f3 = f1, f((x2 + x1) / 2), f2
        else:
            if f_min < f2:
                x1, x2, x3 = x1, x_min, x2
                f1, f2, f3 = f1, f_min, f2
            else:
                x1, x2, x3 = x2, x3, x_min
                f1, f2, f3 = f2, f3, f_min
        
        # Update the best function value
        f_best = np.min([f_best, f_min])
        
        # Use the Fibonacci search to find the next points for parabolic interpolation
        x = np.sort(x)
        x[0] = x[1] - (x[2] - x[1])
        x[1], x[2] = x[2], x[0]
        f_x = f(x)
        
        # Update the best function value
        f_best = np.min([f_best, np.min(f_x)])
        
    return (a + b) / 2  # Return the midpoint of the final interval as the approximate minimum
