import numpy as np
from numpy import linalg as la
from prettytable import PrettyTable
import math
from prettytable import PrettyTable

class BaseNumericalMethod:
    """Base class for common utilities and setup for numerical methods."""
    pass


class InterpolationMethods(BaseNumericalMethod):
    """Interpolation methods for numerical analysis."""
    
    def lagrange_interpolation(self, X, m, x):
        """Calculate the Lagrange interpolation for a given x using point set X."""
        n = 1
        n2 = 1
        for i, x_i in enumerate(X):
            if i != m:
                n *= (x - x_i)
                n2 *= (X[m] - x_i)
        return n / n2

    def piecewise_interpolation(self, X, Y, x):
        """Calculate the piecewise interpolation for a given x using points X and their values Y."""
        n = 0
        for i in range(len(X)):
            n += Y[i] * self.lagrange_interpolation(X, i, x)
        return n

    def ddn(self, x, y):
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


class RootFindingMethods(BaseNumericalMethod):
    """Root-finding methods for numerical analysis."""

    def __init__(self):
        self.resultados = PrettyTable(field_names=["i", "p", "f(p)"])
        
    def fixed_point_iteration(self, f, x0: np.ndarray, tol=1e-6, max_iter=1000):
        x = np.array(x0)
        for i in range(max_iter):
            x_new = np.array(f(x))
            if np.allclose(x_new, x, atol=tol):
                return x_new
            x = x_new
        print("Did not converge within the maximum number of iterations.")
        return x

    def Newton_R(self, f, fp, p_0, TOL = 1e-6, N_0 = 1000):
        """Newton-Raphson method."""
        i = 1
        self.resultados.clear_rows()
        while i <= N_0:
            p = p_0 - f(p_0) / fp(p_0)
            self.resultados.add_row([i, p, f(p)])
            if abs(p - p_0) < TOL:
                print(self.resultados)
                print(f"ER = {abs(p - p_0) / abs(p) * 100}%")
                return p
            i += 1
            p_0 = p
        print(self.resultados)
        print(f"El método fracasó después de [{N_0}] iteraciones")
        return None
        
    def secante(self, f, p_0, p_1, TOL = 1e-6, N_0 = 1000):
        """Secant method."""
        self.resultados.clear_rows()
        q_0 = f(p_0)
        q_1 = f(p_1)
        self.resultados.add_row([0, p_0, q_0])
        self.resultados.add_row([1, p_1, q_1])
        i = 2
        while i <= N_0:
            p = p_1 - q_1 * (p_1 - p_0) / (q_1 - q_0)
            self.resultados.add_row([i, p, f(p)])
            if abs(p - p_1) < TOL:
                print(self.resultados)
                print(f"ER = {abs(p - p_1) / abs(p) * 100}%")
                return p
            i += 1
            p_0 = p_1
            q_0 = q_1
            p_1 = p
            q_1 = f(p)
        else:
            print(self.resultados)
            print(f"El método fracasó después de [{N_0}] iteraciones")
            return None

    def steffensen(self, f, x0, tol=1e-6, max_iter=100):
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

class IterativeLinearSystems(BaseNumericalMethod):
    """Iterative methods for solving systems of linear equations."""
    
    def jacobi(A, b, TOL=1e-2, N0=100):
        n = len(A)
        x = np.zeros(n)
        x0 = np.zeros(n)
        k = 1
        while k <= N0:
            print(f"Iteración {k}")
            for i in range(n):
                y = 0
                for j in range(n):
                    if j != i:
                        y += A[i, j] * x0[j]
                x[i] = (-y + b[i]) / A[i, i]
            print(f"x{k} = {x}")
            if la.norm(x - x0) < TOL:
                ER = la.norm(x - x0) / la.norm(x) * 100
                print(f"Error relativo = {ER:.2f}%")
                return x
            k += 1
            x0 = np.copy(x)
        print(f"El método fracasó después de {N0} iteraciones")


class NumericalIntegrationMethods(BaseNumericalMethod):
    """Class for numerical integration methods."""
    


    def simpsons_rule_N(self, f, a, b, n=100):
        """Approximate the integral of `f` from `a` to `b` using Simpson's Rule with `n` subintervals."""
        h_n = (b - a) / n
        odd_terms = 0
        even_terms = 0
        
        for k in range(1, n // 2 + 1):
            odd_terms += f(a + (2 * k - 1) * h_n)
            even_terms += f(a + (2 * k) * h_n)
        
        odd_terms *= 4
        even_terms *= 2
        even_terms += f(a) + f(b)  # Ensure this matches your formula's needs
        
        approximation = odd_terms + even_terms
        approximation *= h_n / 3
        return approximation
    
    
    def trapezoidal_rule_N(self, f, a, b, n):
        """Approximate the integral of `f` from `a` to `b` using the Trapezoidal Rule with `n` subintervals."""
        h = (b - a) / n
        total = 0
        for i in range(1, n):
            total += f(a + h * i)
        total += (f(a) + f(b)) / 2
        total *= h
        return total

        
    
    def Euler_method(self, f, x0, y0, x1, h):
        y = y0
        x = x0
        while x <= x1:
            y = y + h*f(x, y)
            x += h
        return y

    def romberg_integration(self, f, a, b, max_steps, acc):
        '''If acc O(h**2j) => acc = j'''
        # Initialize the R matrix with zeros
        R = np.zeros((max_steps, max_steps))
        # Initial step size
        h = b - a
   
        # First trapezoidal rule
        R[0, 0] = (f(a) + f(b)) * h * 0.5
        # self.print_row(0, R[0])

        for i in range(1, max_steps):
            h /= 2
            # Composite trapezoidal rule for the current level
            total = sum(f(a + (k * h)) for k in range(1, 2**i, 2))
            R[i, 0] = 0.5 * R[i-1, 0] + total * h
            
            # Romberg extrapolation
            for k in range(1, i + 1):
                R[i, k] = R[i, k-1] + (R[i, k-1] - R[i-1, k-1]) / (4**k - 1)
            
            # self.print_row(i, R[i])

            # Check for convergence
            if i > 0 and abs(R[i, i] - R[i-1, i-1]) < acc:
                return R[i, i]
        
        # If the loop completes without returning, convergence wasn't reached
        print(f"Failed to converge within {max_steps} steps.")
        return R[max_steps-1, max_steps-1]  # Best estimate after max_steps
    
    
class NumericalDerivationMethods(BaseNumericalMethod):
    def __init__(self):
    
        self.r = PrettyTable(field_names=['h', 'Progressive', 'Regressive'])

    def dProg(self, f, x0, h=1e-5):
        """Calculate the progressive derivative of function f at x0 with step h."""
        return (f(x0 + h) - f(x0)) / h

    def dReg(self, f, x0, h=1e-5):
        """Calculate the regressive derivative of function f at x0 with step h."""
        return (f(x0) - f(x0 - h)) / h

    def dThreePoint(self, f, x0, h=1e-5):
        """Calculate the derivative of function f at x0 using the three-point formula."""
        return (f(x0 + h) - f(x0 - h)) / (2 * h)

    def dFivePoint(self, f, x0, h=1e-3):
        """Calculate the derivative of function f at x0 using the five-point formula."""
        return (-f(x0 + 2*h) + 8*f(x0 + h) - 8*f(x0 - h) + f(x0 - 2*h)) / (12 * h)

    def demonstrate_derivative(self, f, x0):
        """Demonstrate and tabulate progressive and regressive derivatives for different h."""
        for h in [10**(-i) for i in range(1, 5)]:
            prog = self.dProg(f, x0, h)
            reg = self.dReg(f, x0, h)
            self.r.add_row([h, prog, reg])
        print(self.r)
        


