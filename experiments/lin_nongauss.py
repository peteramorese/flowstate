import numpy as np
from scipy import special as sp
from scipy import spatial
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import vegas

from problem import Problem
from distribution_functions import laplace_cdf, laplace_quantile, laplace_cdf_deriv, laplace_quantile_deriv
import visualizers, integrators

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

n = 2
m = n # additive noise

#s = int(input("seed: "))
#np.random.seed(s)
np.random.seed(17)
A = np.random.uniform(-1, 2, (n, n)) #+ np.random.uniform(1, 2, (n, n))
A_inv = np.linalg.inv(A)


## Initial state and noise distribution cdf and quantiles ##

def Cx0(x):
    return (laplace_cdf(x[0]), laplace_cdf(x[1]))

def Qx0(yx):
    return (laplace_quantile(yx[0]), laplace_quantile(yx[1]))

def Cw(w):
    return (laplace_cdf(w[0]), laplace_cdf(w[1]))

def Qw(yw):
    return (laplace_quantile(yw[0]), laplace_quantile(yw[1]))

## Dynamics and the inverse dynamics ##

def g(w, x):
    # x' = Ax + w
    return tuple(np.matmul(A, np.array(x).T) + np.array(w))

def g_inv(w, xp):
    # x = A^-1 (x' - w)
    return tuple(np.matmul(A_inv, np.array(xp).T - np.array(w).T))

## Transformation functions to the uniform domain ##

def Phi(w, x0):
    return (Cw(w), Cx0(x0))

def Phi_inv(yw, yx):
    return (Qw(yw), Qx0(yx))

## Flow Functions ##

def G(w, x):
    return (w, g(x))

def G_inv(w, xp):
    return (w, g_inv(w, xp))

## Jacobians ##

def J_G(w, x):
    return np.block([[np.identity(m), np.zeros((m, n))], [np.identity(n), A]])

def J_Phiinv(yw, yx):
    sz = len(yw) + len(yx)
    J = np.zeros((sz, sz))

    # dw0/dyw0
    J[0, 0] = laplace_quantile_deriv(yw[0])

    # dw0/dyw1
    J[0, 1] = 0

    # dw1/dyw0
    J[1, 0] = 0 

    # dw1/dyw1
    J[1, 1] = laplace_quantile_deriv(yw[1])

    # dx0/dyx0
    J[2, 2] = laplace_quantile_deriv(yx[0])

    # dx0/dyx1
    J[2, 3] = 0

    # dx1/dyx0
    J[3, 2] = 0

    # dx1/dyx1
    J[3, 3] = laplace_quantile_deriv(yx[1])

    return J

# For now use matrix inverse cus im lazy
def J_Ginv(w, xp):
    return np.linalg.inv(J_G(*G_inv(w, xp)))

# For now use matrix inverse cus im lazy
def J_Phi(w, x0):
    return np.linalg.inv(J_Phiinv(*Phi(w, x0)))

def main():
    prob = Problem(n, m, g, g_inv, Qx0, Qw, Phi, Phi_inv, J_Ginv, J_Phi)    

    resolution_x = 20
    resolution_w = 20
    resolution_yx = 40
    resolution_yw = 40
    n_plot_samples = 10000
    n_int_samples = 100000
    x_bounds = 5.0 * np.array([-1, 1, -1, 1])
    w_bounds = 3.0 * np.array([-1, 1, -1, 1])

    #resolution_w = 20
    #resolution_yw = 30
    #w_bounds = [w_test[0], w_test[0], w_test[1], w_test[1]]


    region = spatial.Rectangle([1.5, 0.5], [2, 1])
    #region = spatial.Rectangle([-3, -3], [3, 3])
    
    #w_test_0, w_test_1 = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10), indexing='ij')
    #for idx in np.ndindex(w_test_0.shape):
    #    w_test = (w_test_0[idx], w_test_1[idx])

    K = 6
    fig, axes = plt.subplots(2, K)
    #fig.suptitle(f"w_test = {w_test}")
    for k in range(0, K):
        print("\nTime step ", k, " out of ", K - 1)
        visualizers.plot_state_dist(axes[0, k], prob, k, resolution_x, resolution_w, x_bounds, w_bounds)#, w_test=w_test)
        visualizers.plot_state_dist_empirical(axes[1, k], prob, k, n_plot_samples, x_bounds)#, w_test=w_test)
        visualizers.plot_region(axes[1, k], region)
        

        # Monte Carlo Probability
        t_i = time.time()
        P_mc = integrators.mc_prob(prob, region, k, n_int_samples)
        comp_time = time.time() - t_i
        print("   MC probability:        ", P_mc, "   [", comp_time, "s]")

        # Numerical grid integration of the density
        t_i = time.time()
        P_grid_density = integrators.density_AQ_integral(prob, region, k, w_bounds)
        comp_time = time.time() - t_i
        print("   Density grid integral: ", P_grid_density, "   [", comp_time, "s]")

        # Numerical Monte Carlo integration of the density
        t_i = time.time()
        P_mc_density = integrators.density_mc_integral(prob, region, k, w_bounds)
        comp_time = time.time() - t_i
        print("   Density mc integral:    ", P_mc_density, "   [", comp_time, "s]")

        # Grid sum of the volume
        t_i = time.time()
        P_vol = integrators.volume_grid_sum(prob, region, k, resolution_yx, resolution_yw)
        comp_time = time.time() - t_i
        print("   Volume grid sum:       ", P_vol, "   [", comp_time, "s]")
    plt.show()

if __name__ == "__main__":
    main()