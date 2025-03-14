import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt

# Definitions
n = 2
m = 2

# Useful functions
def std_gaussian_quantile(x):
    return np.sqrt(2) * sp.erfinv(2*x - 1)

def std_gaussian_cdf(x):
    return 0.5 * (1 + sp.erf(x / np.sqrt(2)))

def std_gaussian_quantile_deriv(x):
    return 1 / (1 / np.sqrt(2 * np.pi) * np.exp(-(sp.erfinv(2*x - 1))**2))

def std_gaussian_cdf_deriv(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

## Initial state and noise distribution cdf and quantiles ##

def Cx0(x):
    return np.array([std_gaussian_cdf(x[0]), std_gaussian_cdf(x[1])])

def Qx0(yx):
    return np.array([std_gaussian_quantile(yx[0]), std_gaussian_quantile(yx[0])])

def Cw(w):
    return np.array([std_gaussian_cdf(w[0]), std_gaussian_cdf(w[1])])

def Qw(yw):
    return np.array([std_gaussian_quantile(yw[0]), std_gaussian_quantile(yw[0])])

## Dynamics and the inverse dynamics ##

def g(x, w):
    def g0(x, w):
        return np.sin(w[0]) * x[0]**3 + w[1]**2

    def g1(x, w):
        return w[0]**3 * 0.5 * x[1] + 1/10 * x[0]**4 + 10 * np.sin(2 * x[0])

    return np.array([g0(x, w), g1(x, w)])

def g_inv(xp, w):
    def g0_inv(xp, w):
        return (xp[0] - w[1]**2)**(1/3)

    def g1_inv(xp, w):
        return -2 / (w[0]**3) * xp[1] - (1/10 * g0_inv(xp, w)**4 + 10 * np.sin(2 * g0_inv(xp, w)))

    return np.array([g0_inv(xp, w), g1_inv(xp, w)])

## Transformation functions to the uniform domain ##

def Phi(x0, w):
    return np.array([Cx0(x0), Cw(w)])

def Phi_inv(yx, yw):
    return np.array([Qx0(yx), Cw(yw)])

## Flow Functions ##

def Gok(x0, w, k):
    xi = x0
    for i in range(0, k):
        xi = g(xi, w)
    return xi, w

def Gok_inv(xk, w, k):
    xi = xk
    for i in range(k, 0, -1):
        xi = g_inv(xi, w)
    return xi, w

## Plot the distributions

def plot_state_dist(ax, k : int, resolution : int, x_bounds : np.ndarray, w_bounds : np.ndarray):
    X0_0, X0_0, W0, W1 = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], resolution), 
                                 np.linspace(x_bounds[2], x_bounds[3], resolution), 
                                 np.linspace(w_bounds[0], w_bounds[1], resolution), 
                                 np.linspace(w_bounds[2], w_bounds[3], resolution))
    

def main():
    n = 2
    m = 2


if __name__ == "__main__":
    main()