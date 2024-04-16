# Numerical Methods Library
## Introduction
This Python library provides a collection of numerical methods for interpolation, root finding, numerical integration, and solving systems of linear equations. It's designed to assist in educational and research activities related to numerical analysis and scientific computing.

## Features

- Interpolation Methods: Lagrange interpolation, Piecewise interpolation, and Newton's divided differences.
- Root Finding Methods: Fixed-point iteration, Newton-Raphson method, Secant method, and Steffensen's method.
- Numerical Integration Methods: Simpson's rule, Trapezoidal rule, and Romberg integration.
- Iterative Linear Systems: Methods like Jacobi for solving linear systems iteratively.
- Numerical Derivation Methods: Various derivative approximation techniques.


## Dependencies
- Python 3.x
- NumPy
- PrettyTable
- Math


## Installation
Clone this repository or download the files directly:

```bash

git clone https://github.com/Zero5896/Numerical_methods.git
cd numerical-methods-library
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Usage
To use the library, import the necessary classes from the modules and create instances as needed. Below are some examples of how to use different methods in this library:

### Example: Using Root Finding Methods
```python

from numerical_methods import RootFindingMethods


root_finder = RootFindingMethods()
f = lambda x: x**2 - 4
fp = lambda x: 2*x
initial_guess = 2
tolerance = 1e-6
root = root_finder.Newton_R(f, fp, initial_guess, TOL=tolerance)
print("Root found:", root)
```
### Example: Using Numerical Integration Methods
```python

from numerical_methods import NumericalIntegrationMethods

integrator = NumericalIntegrationMethods()
f = lambda x: x**2
a = 0
b = 1
approximation = integrator.simpsons_rule_N(f, a, b)
print("Approximated integral:", approximation)
```
## Contibuting
Contributions are welcome! If you'd like to contribute, please fork the repository and use a pull request to add your contributions. If you have any suggestions or issues, please open an issue in the repository.
