import math

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
    golden_ratio = (math.sqrt(5) - 1) / 2
    
    # Define the initial points for the Fibonacci search
    x1 = a + golden_ratio * (b - a)
    x2 = a + (1 - golden_ratio) * (b - a)
    
    # Evaluate the function at the initial points
    f1 = f(x1)
    f2 = f(x2)
    
    # Keep track of the best function value so far
    f_best = min(f1, f2)
    
    # Define the initial points for the parabolic interpolation
    x3 = (a + b) / 2
    x4 = x3 + tol
    
    # Evaluate the function at the initial points
    f3 = f(x3)
    f4 = f(x4)
    
    while abs(b - a) > tol:
        # Determine the minimum point using parabolic interpolation
        x_min = x3 + (f3 - f4) * (x3 - x1) / ((f3 - f2) * (x3 - x4) - (f3 - f4) * (x3 - x2))
        
        # Evaluate the function at the minimum point
        f_min = f(x_min)
        
        # Determine the points to retain for the next iteration
        if x_min > x3:
            if f_min < f3:
                x2, x3, x4 = x3, x_min, x3 + 2 * (x3 - x_min)
                f2, f3, f4 = f3, f_min, f(x4)
            else:
                x1, x2, x3 = x2, x_min, x3
                f1, f2, f3 = f2, f_min, f3
        else:
            if f_min < f3:
                x1, x3, x2 = x1, x_min, x3
                f1, f3, f2 = f1, f_min, f3
            else:
                x1, x2, x3 = x2, x3, x_min
                f1, f2, f3 = f2, f3, f_min
        
        # Update the best function value
        f_best = min(f_best, f_min)
        
        # Use the Fibonacci search to find the next points for parabolic interpolation
        if f2 < f3:
            x1, x3 = x3, x2
            x2 = a + b - x3
            f1, f3 = f3, f2
            f2 = f(x2)
        else:
            x2, x3 = x3, x1
            x1 = a + b - x2
            f2, f1 = f1, f(x1)
    
    return (a + b) / 2  # Return the midpoint of the final interval as the approximate minimum
