import numpy as np
from numpy import linalg as la
from prettytable import PrettyTable



resultados = PrettyTable(field_names=["i", "p", "f(p)"])
    
def fixed_point_iteration( f, x0: np.ndarray, tol=1e-6, max_iter=1000):
    x = np.array(x0)
    for i in range(max_iter):
        x_new = np.array(f(x))
        if np.allclose(x_new, x, atol=tol):
            return x_new
        x = x_new
    print("Did not converge within the maximum number of iterations.")
    return x

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