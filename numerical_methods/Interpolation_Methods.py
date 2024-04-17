import numpy as np


def lagrange_interpolation( X, m, x):
    """Calculate the Lagrange interpolation for a given x using point set X."""
    n = 1
    n2 = 1
    for i, x_i in enumerate(X):
        if i != m:
            n *= (x - x_i)
            n2 *= (X[m] - x_i)
    return n / n2

def piecewise_interpolation( X, Y, x):
    """Calculate the piecewise interpolation for a given x using points X and their values Y."""
    n = 0
    for i in range(len(X)):
        n += Y[i] * lagrange_interpolation(X, i, x)
    return n

def ddn( x, y):
    """Calculate Newton's divided differences for the given points x and their values y."""
    n = len(x)
    f = np.zeros((n, n))
    f[:,0] = y  # first column is y
    for i in range(1, n):
        for j in range(i, n):
            f[j, i] = (f[j, i-1] - f[j-1, i-1]) / (x[j] - x[j-i])
    np.set_printoptions(precision=4, suppress=True)  # Set print options for better formatting
    print("Divided Differences Table:")
    print(f)
    return f 


