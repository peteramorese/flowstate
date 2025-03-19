import numpy as np
from scipy import special as sp
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Definitions
n = 2
m = 2

# Parameters of the dynamics
a1 = 1/2.5
a2 = 1/40
a3 = 1/10

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
    return (std_gaussian_cdf(x[0]), std_gaussian_cdf(x[1]))

def Qx0(yx):
    return (std_gaussian_quantile(yx[0]), std_gaussian_quantile(yx[1]))

def Cw(w):
    return (std_gaussian_cdf(w[0]), std_gaussian_cdf(w[1]))

def Qw(yw):
    return (std_gaussian_quantile(yw[0]), std_gaussian_quantile(yw[1]))

## Dynamics and the inverse dynamics ##

def g(w, x):
    def g0(x, w):
        return a1 * (np.sin(w[0]) + 2) * np.arcsinh(x[0]) + w[1]**2

    def g1(x, w):
        return (w[0]**2 + 1) / 2 * x[1] + a2 * x[0]**4 + a3 * np.sin(2 * x[0])

    return (g0(x, w), g1(x, w))

def g_inv(w, xp):
    def g0_inv(w, xp):
        return np.sinh((xp[0] - w[1]**2) / (a1 * (np.sin(w[0]) + 2)))

    def g1_inv(w, xp):
        return -2 / (w[0]**2 + 1) * (xp[1] - a2 * g0_inv(w, xp)**4 + a3 * np.sin(2 * g0_inv(w, xp)))

    return (g0_inv(w, xp), g1_inv(w, xp))

## Transformation functions to the uniform domain ##

def Phi(w, x0):
    return (Cw(w), Cx0(x0))

def Phi_inv(yw, yx):
    return (Cw(yw), Qx0(yx))

## Flow Functions ##

def G(w, x):
    return (w, g(x))

def G_inv(w, xp):
    return (w, g_inv(w, xp))

def Gok(w, x0, k):
    xi = x0
    for i in range(0, k):
        xi = g(w, xi)
    return (w, xi)

def Gok_inv(w, xk, k):
    xi = xk
    for i in range(k, 0, -1):
        xi = g_inv(w, xi)
    return (w, xi)

## Jacobians ##

def J_G(w, x):
    J = np.zeros((n + m, n + m))
    
    # dw0/dw0
    J[0, 0] = 1

    # dw1/dw1
    J[1, 1] = 1

    # dg0/dw0
    J[2, 0] = a1 * np.cos(w[0]) * np.arcsinh(x[0])

    # dg0/dw1
    J[2, 1] = 2 * w[1]

    # dg0/dx0
    J[2, 2] = a1 * (np.sin(w[0]) + 2) / np.sqrt(x[0]**2 + 1)

    # dg0/dx1
    J[2, 3] = 0

    # dg1/dw0
    J[3, 0] = w[0] * x[1]

    # dg1/dw1
    J[3, 1] = 0

    # dg1/dx0
    J[3, 2] = a2 * 4 * x[0]**3 + 2 * a3 * np.cos(2 * x[0])

    # dg1/dx1
    J[3, 3] = (w[0]**2 + 1) / 2

    return J

def J_Phiinv(yw, yx):
    J = np.zeros((n + m, n + m))

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

# For now use matrix inverse cus im lazy
def J_Ginv(w, xp):

    w, x = G_inv(w, xp)
    #print("input w: ", w, ", x:", x)
    #print("test: \n", J_G(*G_inv(w, xp)))
    return np.linalg.inv(J_G(*G_inv(w, xp)))

# For now use matrix inverse cus im lazy
def J_Phi(w, x0):
    return np.linalg.inv(J_Phiinv(*Phi(w, x0)))

## Density at time k ##

def p(w, xk, k):
    #print("\n EVAL w: ", w, ", x:", xk)
    try:
        # Invert to x0
        J = np.identity(n + m)
        for i in range(k, 0, -1):
            J = np.matmul(J, J_Ginv(*Gok_inv(w, xk, i - 1)))

        # Invert to y
        J = np.matmul(J, J_Phi(*Gok_inv(w, xk, k)))

        return np.abs(np.linalg.det(J))
    except np.linalg.LinAlgError:
        return 0

## Sample states at time k ##

def sample_empirical(k : int, n_samples : int):
    xk_samples = np.zeros((2, n_samples))
    for s in range(0, n_samples):
        yw_rand = np.random.uniform(0, 1, 2)
        yx_rand = np.random.uniform(0, 1, 2)
        w, xk = Gok(*Phi_inv(yw_rand, yx_rand), k)
        xk_samples[:, s] = xk

        ## Test specific w #####################################
        #w, x0 = Phi_inv(yw_rand, yx_rand)
        #w, xk = Gok(w_test, x0, k)
        #xk_samples[:, s] = xk
    return xk_samples


## Plot the distributions
w_test = (1.5, 1.0)

def plot_state_dist(ax :plt.Axes, k : int, resolution_x : int, resolution_w : int, x_bounds : np.ndarray, w_bounds : np.ndarray):
    #W0, W1, Xk0, Xk1 = np.meshgrid(np.linspace(w_bounds[0], w_bounds[1], resolution_w), 
    #                             np.linspace(w_bounds[2], w_bounds[3], resolution_w), 
    #                             np.linspace(x_bounds[0], x_bounds[1], resolution_x), 
    #                             np.linspace(x_bounds[2], x_bounds[3], resolution_x), indexing='ij')
    
    # Test specific w #####################################3
    W0, W1, Xk0, Xk1 = np.meshgrid(np.linspace(w_test[0], w_test[0], 1), 
                                 np.linspace(w_test[1], w_test[1], 1), 
                                 np.linspace(x_bounds[0], x_bounds[1], resolution_x), 
                                 np.linspace(x_bounds[2], x_bounds[3], resolution_x), indexing='ij')

    p_values_w_x = np.empty_like(W0)

    for idx in np.ndindex(p_values_w_x.shape):
        w = (W0[idx], W1[idx])
        xk = (Xk0[idx], Xk1[idx])
        p_values_w_x[idx] = p(w, xk, k)

    #p_values_w_x = np.vectorize(lambda w0, w1, xk0, xk1: p((w0, w1), (xk0, xk1), k))(W0, W1, Xk0, Xk1)

    p_values_x = np.nansum(p_values_w_x, axis=(0, 1))

    Xk0, Xk1 = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], resolution_x), np.linspace(x_bounds[2], x_bounds[3], resolution_x))
    ax.contourf(Xk0, Xk1, p_values_x.T, levels=50, cmap='viridis')
    print("p_values_x: \n", p_values_x)
    #ax.colorbar(label='p(x0, x1)')
    
def plot_state_dist_empirical(ax :plt.Axes, k : int, n_samples : int, x_bounds : np.ndarray, w_bounds : np.ndarray):
    xk_samples = sample_empirical(k, n_samples)
    ax.scatter(xk_samples[0, :], xk_samples[1, :], s=.05)
    ax.set_xlim((x_bounds[0], x_bounds[1]))
    ax.set_ylim((x_bounds[2], x_bounds[3]))

def plot_region(ax :plt.Axes, region : spatial.Rectangle):
    # Make sure the region is just over x
    assert region.m == 2

    rect_patch = patches.Rectangle(
        region.mins,  # Bottom-left corner (x_min, y_min)
        region.maxes[0] - region.mins[0],  # Width
        region.maxes[1] - region.mins[1],  # Height
        linewidth=2,
        edgecolor="red",
        facecolor="red",
        alpha=0.5  # Opacity (0 = fully transparent, 1 = fully opaque)
    )

    ax.add_patch(rect_patch)

## Monte Carlo Probability ##

def mc_prob(region : spatial.Rectangle, k, n_samples):
    # Make sure the region is just over x
    assert region.m == 2

    xk_samples = sample_empirical(k, n_samples)
    contained = (region.mins[:, None] <= xk_samples) & (region.maxes[:, None] >= xk_samples)

    n_contained = np.sum(np.all(contained, axis=0))

    # empirical probability
    return n_contained / n_samples

def numerical_integral(region : spatial.Rectangle, k):
    pass

def main():
    resolution_x = 30
    resolution_w = 40
    n_plot_samples = 10000
    n_int_samples = 100000
    x_bounds = 3.0 * np.array([-1, 1, -1, 1])
    w_bounds = -10.0 * np.array([-1, 1, -1, 1])

    K = 4
    fig, axes = plt.subplots(2, K)

    region = spatial.Rectangle([1.5, 0.5], [2, 1])
    
    for k in range(0, K):
        print("Time step ", k, " out of ", K - 1)
        #plot_state_dist(axes[0, k], k, resolution_x, resolution_w, x_bounds, w_bounds)
        plot_state_dist_empirical(axes[1, k], k, n_plot_samples, x_bounds, w_bounds)
        plot_region(axes[1, k], region)
        print("   MC probability: ", mc_prob(region, k, n_int_samples))

    plt.show()

if __name__ == "__main__":
    main()