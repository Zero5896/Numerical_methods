import numpy as np
from numpy import linalg as la
from prettytable import PrettyTable
from typing import Callable

resultados = PrettyTable(field_names=["i", "p", "f(p)"])
    
def fixed_point_iteration(f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Performs fixed-point iteration to find the fixed point of a given function.

    Parameters:
    - f (callable): The function for which the fixed point is sought.
    - x0 (numpy.ndarray): The initial guess for the fixed point.
    - tol (float, optional): The tolerance for convergence. Defaults to 1e-6.
    - max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

    Returns:
    - numpy.ndarray: The estimated fixed point.

    Raises:
    - ValueError: If the function does not converge within the maximum number of iterations.

    Fixed-point iteration is an iterative method used to find the fixed point of a function.
    The fixed point of a function f(x) is a value x* such that f(x*) = x*.

    The function iterates through the following steps:
    1. Start with an initial guess x0.
    2. Compute the next guess x_new = f(x).
    3. Repeat step 2 until the difference between x_new and x is within the specified tolerance tol, or until reaching the maximum number of iterations.

    If the function converges within the specified tolerance, the estimated fixed point is returned.
    If the function does not converge within the maximum number of iterations, a ValueError is raised.

    Example:
    >>> f = lambda x: np.cos(x)  # Define the function f(x) = cos(x)
    >>> initial_guess = np.array([1.0])  # Initial guess for the fixed point
    >>> fixed_point = fixed_point_iteration(f, initial_guess)  # Perform fixed-point iteration
    >>> fixed_point  # Display the estimated fixed point
    array([0.73908513])
    """
    x = np.array(x0)
    for i in range(max_iter):
        x_new = np.array(f(x))
        if np.allclose(x_new, x, atol=tol):
            return x_new
        x = x_new
    raise ValueError("Did not converge within the maximum number of iterations.")

def Newton_R( f, fp, p_0, TOL = 1e-6, N_0 = 1000):
    """Newton-Raphson method."""
    i = 1
    resultados.clear_rows()
    while i <= N_0:
        p = p_0 - f(p_0) / fp(p_0)
        resultados.add_row([i, p, f(p)])
        if abs(p - p_0) < TOL:
            print(resultados)
            print(f"ER = {abs(p - p_0) / abs(p) * 100}%")
            return p
        i += 1
        p_0 = p
    print(resultados)
    print(f"El método fracasó después de [{N_0}] iteraciones")
    return None
    
def secante( f, p_0, p_1, TOL = 1e-6, N_0 = 1000):
    """Secant method."""
    resultados.clear_rows()
    q_0 = f(p_0)
    q_1 = f(p_1)
    resultados.add_row([0, p_0, q_0])
    resultados.add_row([1, p_1, q_1])
    i = 2
    while i <= N_0:
        p = p_1 - q_1 * (p_1 - p_0) / (q_1 - q_0)
        resultados.add_row([i, p, f(p)])
        if abs(p - p_1) < TOL:
            print(resultados)
            print(f"ER = {abs(p - p_1) / abs(p) * 100}%")
            return p
        i += 1
        p_0 = p_1
        q_0 = q_1
        p_1 = p
        q_1 = f(p)
    else:
        print(resultados)
        print(f"El método fracasó después de [{N_0}] iteraciones")
        return None

def steffensen( f, x0, tol=1e-6, max_iter=100):
    """Steffensen's method for root finding."""
    x = x0
    for iter_count in range(max_iter):
        x_next = f(x)
        x_next_next = f(x_next)
        denominator = x_next_next - 2 * x_next + x
        if abs(denominator) < tol:  # Prevent division by zero
            print("Denominator too small. Method might not converge.")
            return None
        x_new = x - ((x_next - x) ** 2) / denominator
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print("Steffensen's method did not converge after", max_iter, "iterations.")
    return None