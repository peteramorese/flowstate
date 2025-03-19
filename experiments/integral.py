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
    return [std_gaussian_cdf(x[0]), std_gaussian_cdf(x[1])]

def Qx0(yx):
    return [std_gaussian_quantile(yx[0]), std_gaussian_quantile(yx[0])]

def Cw(w):
    return [std_gaussian_cdf(w[0]), std_gaussian_cdf(w[1])]

def Qw(yw):
    return [std_gaussian_quantile(yw[0]), std_gaussian_quantile(yw[0])]

## Dynamics and the inverse dynamics ##

def g(w, x):
    def g0(x, w):
        return np.sin(w[0]) * x[0]**3 + w[1]**2

    def g1(x, w):
        return w[0]**3 * 0.5 * x[1] + 1/10 * x[0]**4 + 10 * np.sin(2 * x[0])

    return [g0(x, w), g1(x, w)]

def g_inv(w, xp):
    def g0_inv(w, xp):
        return (xp[0] - w[1]**2)**(1/3)

    def g1_inv(w, xp):
        return -2 / (w[0]**3) * xp[1] - (1/10 * g0_inv(w, xp)**4 + 10 * np.sin(2 * g0_inv(w, xp)))

    return [g0_inv(w, xp), g1_inv(w, xp)]

## Transformation functions to the uniform domain ##

def Phi(w, x0):
    return [Cw(w), Cx0(x0)]

def Phi_inv(yw, yx):
    return [Cw(yw), Qx0(yx)]

## Flow Functions ##

def Gok(w, x0, k):
    xi = x0
    for i in range(0, k):
        xi = g(w, xi)
    return [w, xi]

def Gok_inv(w, xk, k):
    xi = xk
    for i in range(k, 0, -1):
        xi = g_inv(w, xi)
    return [w, xi]

## Jacobians ##

def J_G(w, x):
    J = np.zeros(n + m, n + m)
    
    # dw0/dw0
    J[0, 0] = 1

    # dw1/dw1
    J[1, 1] = 1

    # dg0/dw0
    J[2, 0] = np.cos(w[0]) * x[0]**3

    # dg0/dw1
    J[2, 1] = 2 * w[1]

    # dg0/dx0
    J[2, 2] = 3 * np.sin(w[0]) * x[0]**2

    # dg0/dx1
    J[2, 3] = 0

    # dg1/dw0
    J[3, 0] = 3 * w[0]**2 * x[1] / 2

    # dg1/dw1
    J[3, 1] = 0

    # dg1/dx0
    J[3, 2] = 4 / 10 * x[0]**3 + 2 * 10 * np.cos(2 * x[0])

    # dg1/dx1
    J[3, 3] = w[0]**3 / 2

    return J

def J_Phi(yw, yx):
    J = np.zeros(n + m, n + m)

    # dw0/dyw0
    J[0, 0] = std_gaussian_quantile_deriv(yw[0])

    # dw0/dyw1
    J[0, 1] = 0

    # dw1/dyw0
    J[1, 0] = 0 

    # dw1/dyw1
    J[1, 1] = std_gaussian_quantile_deriv(yw[1])

    # dx0/dyx0
    J[2, 2] = std_gaussian_quantile_deriv(yx[0])

    # dx0/dyx1
    J[2, 3] = 0

    # dx1/dyx0
    J[3, 2] = 0

    # dx1/dyx1
    J[3, 3] = std_gaussian_quantile_deriv(yx[1])

    return J

## Plot the distributions

def plot_state_dist(ax, k : int, resolution : int, x_bounds : np.ndarray, w_bounds : np.ndarray):
    Xk0, Xk1, W0, W1 = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], resolution), 
                                 np.linspace(x_bounds[2], x_bounds[3], resolution), 
                                 np.linspace(w_bounds[0], w_bounds[1], resolution), 
                                 np.linspace(w_bounds[2], w_bounds[3], resolution))
    


def main():
    n = 2
    m = 2


if __name__ == "__main__":
    main()