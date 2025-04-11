import numpy as np
from scipy import spatial
import time
import matplotlib.pyplot as plt

import integrators, distribution_functions

import lin_gauss, lin_nongauss, nonlin, bicycle

def main():
    epsilon = 1e-10
    region = spatial.Rectangle([1.5, 0.5], [2, 1])
    w_bounds = distribution_functions.std_gaussian_quantile(epsilon) * np.array([-1, 1, -1, 1])

    prob = nonlin.prob
    k = 4

    # Numerical Monte Carlo integration of the density

    #P_true = integrators.density_mc_integral(prob, region, k, w_bounds)

    #n_samples_true = 5000000
    #P_true = integrators.mc_prob(prob, region, k, n_samples_true)
    #print("True probability for ", n_samples_true, " samples: ", P_true)

    resolutions = list(range(1, 41))
    #grid_ss_errors = list()
    #grid_u_errors = list()
    #for resolution in resolutions:
    #    print("Computing resolution ", resolution)
    #    P_grid_ss = integrators.density_grid_sum(prob, region, k, resolution_xk=resolution, resolution_yw=resolution, epsilon=epsilon)
    #    P_grid_u = integrators.volume_grid_sum(prob, region, k, resolution_yx=resolution, resolution_yw=resolution)
    #    grid_ss_err = np.abs(P_grid_ss - P_true) / P_true
    #    grid_u_err = np.abs(P_grid_u - P_true) / P_true
    #    print("     Grid state space error:   ", grid_ss_err)
    #    print("     Grid uniform space error: ", grid_u_err)
    #    grid_ss_errors.append(grid_ss_err)
    #    grid_u_errors.append(grid_u_err)

    grid_ss_errors = [1, 1, 1, 1, 23.72, 16.58, 7.346, 3.036, 1.8617, 0.216, 0.75225, 0.3949, 0.1995, 0.05305, 0.1731, 0.30894, 0.21334, 0.08567, 0.06841, 0.09885, 0.05831, 0.13795, 0.083434, 0.07115, 0.03185, 0.03144, 0.06356, 0.06322, 0.06708, 0.04103, 0.03535, 0.05920, 0.05673, 0.03755, 0.02505, 0.04094, 0.03398, 0.03734, 0.04095, 0.03354]
    grid_u_errors = [1, 1, 1, 1, 0.487, 0.996, 0.3414, 0.535, 0.487, 0.239, 0.1489, 0.02689, 0.26542, 0.21611, 0.1061, 0.02584, 0.06028, 0.12114, 0.12376, 0.10273, 0.05333, 0.00714, 0.022408, 0.05589, 0.08099, 0.07859, 0.04456, 0.02907, 0.03715, 0.03718, 0.04515, 0.02894, 0.03267, 0.03902, 0.04705, 0.03907, 0.03085, 0.02782, 0.02848, 0.03632]
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(resolutions, grid_ss_errors, label="Grid State Space")
    ax.plot(resolutions, grid_u_errors, label="Grid Uniform Space")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Percent Error")
    ax.set_ylim((0,1))
    ax.legend()
    plt.show()


    


if __name__ == "__main__":
    main()