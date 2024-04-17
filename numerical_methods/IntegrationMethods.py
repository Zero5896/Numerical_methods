import numpy as np



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
    
    
def trapezoidal_rule_N( f, a, b, n):
    """Approximate the integral of `f` from `a` to `b` using the Trapezoidal Rule with `n` subintervals."""
    h = (b - a) / n
    total = 0
    for i in range(1, n):
        total += f(a + h * i)
    total += (f(a) + f(b)) / 2
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