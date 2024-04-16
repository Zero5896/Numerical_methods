import numpy as np
from numpy import linalg as la



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