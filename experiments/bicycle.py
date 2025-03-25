import numpy as np
from scipy import special as sp
from scipy import spatial
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import vegas

from problem import Problem
from distribution_functions import std_gaussian_cdf, std_gaussian_quantile, std_gaussian_cdf_deriv, std_gaussian_quantile_deriv
import visualizers, integrators

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

##### Kinematic bicycle #####
#
# 3D states / 2D uncertain parameters
#
# x[0] = x postion
# x[1] = y position
# x[2] = heading (theta)
# w[0] = velocity
# w[1] = steering angle
#
#############################

# Parameters of the dynamics
delta_t = 0.1
wheelbase = 0.5
velocity_scale = 20.0
steering_angle_scale = 0.01

# Parameters of noise
x_sigma = 0.3
y_sigma = 0.3
theta_sigma = 0.3
velocity_sigma = 1.5
steering_angle_sigma = 0.5

## Initial state and noise distribution cdf and quantiles ##

def Cx0(x):
    return (std_gaussian_cdf(x[0], sigma=x_sigma), std_gaussian_cdf(x[1], sigma=y_sigma), std_gaussian_cdf(x[2], sigma=theta_sigma))

def Qx0(yx):
    return (std_gaussian_quantile(yx[0], sigma=x_sigma), std_gaussian_quantile(yx[1], sigma=y_sigma), std_gaussian_quantile(yx[2], sigma=theta_sigma))

def Cw(w):
    return (std_gaussian_cdf(w[0], sigma=velocity_sigma), std_gaussian_cdf(w[1], sigma=steering_angle_sigma))

def Qw(yw):
    return (std_gaussian_quantile(yw[0], sigma=velocity_sigma), std_gaussian_quantile(yw[1], sigma=steering_angle_sigma))

## Dynamics and the inverse dynamics ##

def g(w, x):
    def g0(x, w):
        return x[0] + velocity_sigma * (w[0] + 1) * np.cos(x[2]) * delta_t

    def g1(x, w):
        return x[1] + velocity_sigma * (w[0] + 1) * np.sin(x[2]) * delta_t

    def g2(x, w):
        return x[2] + velocity_sigma * (w[0] + 1) / wheelbase * np.tan(steering_angle_scale * w[1]) * delta_t

    return (g0(x, w), g1(x, w), g2(x, w))

def g_inv(w, xp):
    def g0_inv(w, xp):
        return xp[0] - velocity_sigma * (w[0] + 1) * np.cos(xp[2]) * delta_t

    def g1_inv(w, xp):
        return xp[1] - velocity_sigma * (w[0] + 1) * np.cos(xp[2]) * delta_t

    def g2_inv(w, xp):
        return xp[2] - velocity_sigma * (w[0] + 1) / wheelbase * np.tan(steering_angle_scale * w[1]) * delta_t

    return (g0_inv(w, xp), g1_inv(w, xp), g2_inv(w, xp))

## Transformation functions to the uniform domain ##

def Phi(w, x0):
    return (Cw(w), Cx0(x0))

def Phi_inv(yw, yx):
    return (Qw(yw), Qx0(yx))


## Jacobians ##

def J_G(w, x):
    J = np.zeros((5, 5))
    
    # dw0/dw0
    J[0, 0] = 1

    # dw1/dw1
    J[1, 1] = 1

    # dg0/dw0
    J[2, 0] = velocity_sigma * np.cos(x[2]) * delta_t

    # dg0/dw1
    J[2, 1] = 0

    # dg0/dx0
    J[2, 2] = 1

    # dg0/dx1
    J[2, 3] = 0

    # dg0/dx2
    J[2, 4] = -velocity_sigma * (w[0] + 1) * np.sin(x[2]) * delta_t

    # dg1/dw0
    J[3, 0] = velocity_sigma * np.sin(x[2]) * delta_t

    # dg1/dw1
    J[3, 1] = 0

    # dg1/dx0
    J[3, 2] = 0

    # dg1/dx1
    J[3, 3] = 1

    # dg1/dx2
    J[3, 4] = velocity_sigma * (w[0] + 1) * np.cos(x[2]) * delta_t

    # dg2/dw0
    J[4, 0] = velocity_sigma * w[0] / wheelbase * np.tan(steering_angle_scale * w[1]) * delta_t

    # dg2/dw1
    J[4, 1] = steering_angle_scale * velocity_sigma * (w[0] + 1) / (wheelbase * np.cos(steering_angle_scale * w[1])**2) * delta_t

    # dg2/dx0
    J[4, 2] = 0

    # dg2/dx1
    J[4, 3] = 0

    # dg2/dx2
    J[4, 4] = 1

    return J

def J_Phiinv(yw, yx):
    sz = len(yw) + len(yx)
    J = np.zeros((sz, sz))

    # dw0/dyw0
    J[0, 0] = std_gaussian_quantile_deriv(yw[0])

    # dw1/dyw1
    J[1, 1] = std_gaussian_quantile_deriv(yw[1])

    # dx0/dyx0
    J[2, 2] = std_gaussian_quantile_deriv(yx[0], sigma=x_sigma)

    # dx1/dyx1
    J[3, 3] = std_gaussian_quantile_deriv(yx[1], sigma=y_sigma)

    # dx2/dyx2
    J[4, 4] = std_gaussian_quantile_deriv(yx[2], sigma=theta_sigma)

    return J

def G(w, x):
    return (w, g(x))

def G_inv(w, xp):
    return (w, g_inv(w, xp))


# For now use matrix inverse cus im lazy
def J_Ginv(w, xp):
    return np.linalg.inv(J_G(*G_inv(w, xp)))

# For now use matrix inverse cus im lazy
def J_Phi(w, x0):
    return np.linalg.inv(J_Phiinv(*Phi(w, x0)))

def simulate_mc(ax :plt.Axes, prob : Problem, k : int, n_samples : int, position_bounds : np.ndarray):
    xk_samples = prob.sample_empirical(k, n_samples)
    ax.scatter(xk_samples[0, :], xk_samples[1, :], s=.05)
    ax.set_xlim((position_bounds[0], position_bounds[1]))
    ax.set_ylim((position_bounds[2], position_bounds[3]))
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

def main():
    n, m = 3, 2
    prob = Problem(n, m, g, g_inv, Phi, Phi_inv, J_Ginv, J_Phi)    

    resolution_x = 20
    resolution_w = 20
    resolution_yx = 10
    resolution_yw = 10
    n_plot_samples = 10000
    n_int_samples = 100000
    x_bounds = 3.0 * np.array([-1, 1, -1, 1, -1, 1])
    w_bounds = 3.0 * np.array([-1, 1, -1, 1])

    #resolution_w = 20
    #resolution_yw = 30
    #w_bounds = [w_test[0], w_test[0], w_test[1], w_test[1]]


    region = spatial.Rectangle([2.5, 1.0, -2*np.pi], [5, 2.5, 2*np.pi])
    #region = spatial.Rectangle([-3, -3], [3, 3])
    
    #w_test_0, w_test_1 = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10), indexing='ij')
    #for idx in np.ndindex(w_test_0.shape):
    #    w_test = (w_test_0[idx], w_test_1[idx])

    K = 4
    fig, axes = plt.subplots(1, K)
    #fig.suptitle(f"w_test = {w_test}")
    for k in range(0, 2*K, 2):
        print("\nTime step ", k, " out of ", K - 1)
        #visualizers.plot_state_dist(axes[0, k], prob, k, resolution_x, resolution_w, x_bounds, w_bounds)#, w_test=w_test)
        #visualizers.plot_state_dist_empirical(axes[1, k], prob, k, n_plot_samples, x_bounds)#, w_test=w_test)
        simulate_mc(axes[k], prob, k, n_plot_samples, position_bounds=[-1, 10, -10, 10])
        visualizers.plot_region(axes[k], region)
        

        # Monte Carlo Probability
        t_i = time.time()
        P_mc = integrators.mc_prob(prob, region, k, n_int_samples)
        comp_time = time.time() - t_i
        print("   MC probability:        ", P_mc, "   [", comp_time, "s]")

        # Numerical grid integration of the density
        t_i = time.time()
        P_grid_density = integrators.density_grid_integral(prob, region, k, w_bounds)
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