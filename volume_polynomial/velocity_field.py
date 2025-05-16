import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import norm
from scipy.spatial import Rectangle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import product
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

def generate_data(f, n, p, noise_std=0.1, seed=0):
    np.random.seed(seed)
    X = np.random.randn(p, n)
    Y_clean = np.array([f(x) for x in X])
    noise = np.random.randn(p, n) * noise_std
    Y = Y_clean + noise
    return X, Y

def erf_space_transform(x):
    return norm.cdf(x)

def erf_space_transform_jacobian(x):
    return np.diag(norm.pdf(x))

def fit_constrained_polynomial(X_u, Y_u, degree):
    assert Y_u.shape == X_u.shape
    n = X_u.shape[1]
    models = []
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    
    X_poly = poly.fit_transform(X_u)
    bc_scaling = 1 / (np.multiply(X_u, 1-X_u))
    Y_u_unconst = np.multiply(Y_u, bc_scaling)

    for i in range(n):
        regressor = LinearRegression()        
        regressor.fit(X_poly, Y_u_unconst[:, i])
        coeffs = regressor.coef_

        models.append((coeffs, poly))
    return models

if __name__ == "__main__":
    # Parameters
    n = 3
    p = 10
    degree = 3

    # Generate data
    def example_function(x):
        return np.sin(x) * 1 / (1 + x**2)
    X, Y = generate_data(example_function, n=n, p=p)

    print("X:\n", X)
    print("Y:\n", Y)

    # Transform to unit box
    X_u = erf_space_transform(X)
    Y_u = erf_space_transform_jacobian(X) * Y

    # Fit constrained model
    models = fit_constrained_polynomial(X_u, Y_u, degree=degree)
