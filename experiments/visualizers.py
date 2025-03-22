import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import spatial

from problem import Problem
import integrators


## Plot the distributions

def plot_state_dist(ax :plt.Axes, prob : Problem, k : int, resolution_x : int, resolution_w : int, x_bounds : np.ndarray, w_bounds : np.ndarray, w_test = None):
    if w_test:
        W0, W1, Xk0, Xk1 = np.meshgrid(np.linspace(w_test[0], w_test[0], 1), 
                                    np.linspace(w_test[1], w_test[1], 1), 
                                    np.linspace(x_bounds[0], x_bounds[1], resolution_x), 
                                    np.linspace(x_bounds[2], x_bounds[3], resolution_x), indexing='ij')
        dw0 = 1 
        dw1 = 1 
    else:
        W0, W1, Xk0, Xk1 = np.meshgrid(np.linspace(w_bounds[0], w_bounds[1], resolution_w), 
                                    np.linspace(w_bounds[2], w_bounds[3], resolution_w), 
                                    np.linspace(x_bounds[0], x_bounds[1], resolution_x), 
                                    np.linspace(x_bounds[2], x_bounds[3], resolution_x), indexing='ij')
        dw0 = W0[(1,)*4] - W0[(0,)*4]
        dw1 = W1[(1,)*4] - W1[(0,)*4]
    
    p_values_w_x = np.empty_like(W0)

    for idx in np.ndindex(p_values_w_x.shape):
        w = (W0[idx], W1[idx])
        xk = (Xk0[idx], Xk1[idx])
        p_values_w_x[idx] = prob.p(w, xk, k)

    #p_values_w_x = np.vectorize(lambda w0, w1, xk0, xk1: p((w0, w1), (xk0, xk1), k))(W0, W1, Xk0, Xk1)
    
    p_values_x = dw0 * dw1 * np.nansum(p_values_w_x, axis=(0, 1))

    Xk0, Xk1 = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], resolution_x), np.linspace(x_bounds[2], x_bounds[3], resolution_x))
    ax.contourf(Xk0, Xk1, p_values_x.T, levels=50, cmap='viridis')
    
def plot_state_dist_empirical(ax :plt.Axes, prob : Problem, k : int, n_samples : int, x_bounds : np.ndarray, w_test = None):
    xk_samples = prob.sample_empirical(k, n_samples, w_test)
    ax.scatter(xk_samples[0, :], xk_samples[1, :], s=.05)
    ax.set_xlim((x_bounds[0], x_bounds[1]))
    ax.set_ylim((x_bounds[2], x_bounds[3]))

def plot_region(ax :plt.Axes, region : spatial.Rectangle):

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
