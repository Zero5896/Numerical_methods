import numpy as np
from typing import Callable


def simpsons_rule_N( f, a, b, n=100):
        """Approximate the integral of `f` from `a` to `b` using Simpson's Rule with `n` subintervals."""
        h_n = (b - a) / n
        odd_terms = 0
        even_terms = 0
        
        for k in range(1, n // 2 + 1):
            odd_terms += f(a + (2 * k - 1) * h_n)
            even_terms += f(a + (2 * k) * h_n)
        
        odd_terms *= 4
        even_terms *= 2
        even_terms += f(a) + f(b) 
        
        approximation = odd_terms + even_terms
        approximation *= h_n / 3
        return approximation
    
    


def trapezoidal_rule_N(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Approximate the integral of a function `f` over the interval [a, b] using the trapezoidal rule.
    
    Parameters:
    - f (Callable[[np.ndarray], np.ndarray]): The function to integrate. It should accept a float or numpy array and return a float or numpy array.
    - a (float): The lower limit of the integration.
    - b (float): The upper limit of the integration.
    - n (int): The number of trapezoids (subintervals) to use.
    
    Returns:
    - float: The approximate value of the integral of `f` from `a` to `b`.
    
    Method:
    The function calculates the integral by summing the areas of `n` trapezoids formed under the curve of `f`.
    Each trapezoid's area is calculated using the formula: (f(x_i) + f(x_{i+1})) * h / 2,
    where `h` is the width of each subinterval, and x_i and x_{i+1} are the endpoints of the subinterval.
    
    Example:
    >>> f = lambda x: np.sin(x)  # A function that can handle numpy array inputs.
    >>> a, b, n = 0, np.pi, 1000  # Define the limits of integration and the number of subintervals
    >>> result = trapezoidal_rule_N(f, a, b, n)  # Calculate the integral using the trapezoidal rule
    >>> print(result)  # Print the result
    """
    # Step size (width of each trapezoid)
    h = (b - a) / n

    # Initialize the total sum of areas of trapezoids
    total = 0.5 * (f(a) + f(b))  # Start by adding half the first and last function values
    
    # Generate the intermediate points for evaluation
    x_values = np.linspace(a + h, b - h, n - 1)
    
    # Sum up all the function values at intermediate points
    total += np.sum(f(x_values))

    # Multiply the sum by the step size to get the integral
    total *= h

    return total



    

def Euler_method( f, x0, y0, x1, h):
    y = y0
    x = x0
    while x <= x1:
        y = y + h*f(x, y)
        x += h
    return y

def romberg_integration( f, a, b, max_steps, acc):
    '''If acc O(h**2j) => acc = j'''
    # Initialize the R matrix with zeros
    R = np.zeros((max_steps, max_steps))
    # Initial step size
    h = b - a

    # First trapezoidal rule
    R[0, 0] = (f(a) + f(b)) * h * 0.5
    # print_row(0, R[0])

    for i in range(1, max_steps):
        h /= 2
        # Composite trapezoidal rule for the current level
        total = sum(f(a + (k * h)) for k in range(1, 2**i, 2))
        R[i, 0] = 0.5 * R[i-1, 0] + total * h
        
        # Romberg extrapolation
        for k in range(1, i + 1):
            R[i, k] = R[i, k-1] + (R[i, k-1] - R[i-1, k-1]) / (4**k - 1)
        
        # print_row(i, R[i])

        # Check for convergence
        if i > 0 and abs(R[i, i] - R[i-1, i-1]) < acc:
            return R[i, i]
    
    # If the loop completes without returning, convergence wasn't reached
    print(f"Failed to converge within {max_steps} steps.")
    return R[max_steps-1, max_steps-1]  # Best estimate after max_steps