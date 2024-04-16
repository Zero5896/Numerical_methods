import numpy as np
from numpy import linalg as la
from prettytable import PrettyTable


r = PrettyTable(field_names=['h', 'Progressive', 'Regressive'])

def dProg( f, x0, h=1e-5):
    """Calculate the progressive derivative of function f at x0 with step h."""
    return (f(x0 + h) - f(x0)) / h

def dReg( f, x0, h=1e-5):
    """Calculate the regressive derivative of function f at x0 with step h."""
    return (f(x0) - f(x0 - h)) / h

def dThreePoint( f, x0, h=1e-5):
    """Calculate the derivative of function f at x0 using the three-point formula."""
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def dFivePoint( f, x0, h=1e-3):
    """Calculate the derivative of function f at x0 using the five-point formula."""
    return (-f(x0 + 2*h) + 8*f(x0 + h) - 8*f(x0 - h) + f(x0 - 2*h)) / (12 * h)

def demonstrate_derivative( f, x0):
    """Demonstrate and tabulate progressive and regressive derivatives for different h."""
    for h in [10**(-i) for i in range(1, 5)]:
        prog = dProg(f, x0, h)
        reg = dReg(f, x0, h)
        r.add_row([h, prog, reg])
    print(r)
        