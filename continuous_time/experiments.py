import numpy as np
import sympy as sp
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle
from itertools import product

from velocity_field import VelocityField
from pdf import visualize_2D_pdf
from box_flow_algo import sample_box_flow_algo
import integrators
import visualizers as vis


def region_post_sampling(vf : VelocityField, dt : float, region : Rectangle, n_samples : int, direction = 1, epsilon : float = 0):
    """
    Propagate an over-estimate of the region in the velocity field
    """
    
    def sample_extrema_on_surface(f, fixed_dim : int, fixed_coord : float, maximize = True):
        if maximize:
            max_val = -np.inf
            for _ in range(n_samples):
                x = np.random.uniform(region.mins, region.maxes)
                x[fixed_dim] = fixed_coord
                fx = f(x)
                if fx > max_val:
                    max_val = fx
            return max_val
        else:
            min_val = np.inf
            for _ in range(n_samples):
                x = np.random.uniform(region.mins, region.maxes)
                x[fixed_dim] = fixed_coord
                fx = f(x)
                if fx < min_val:
                    min_val = fx
            return min_val

    new_region = copy.deepcopy(region)


    for d in range(vf.dim):
        vi = lambda x : vf.velocity(x, t=None, i=d)

        # Compute the "max" deviation of the upper surface
        new_region.maxes[d] += direction * dt * sample_extrema_on_surface(vi, d, region.maxes[d], True) + epsilon
        # Compute the "max" deviation of the lower surface
        new_region.mins[d] += direction * dt * sample_extrema_on_surface(vi, d, region.mins[d], False) - epsilon

    return new_region


if __name__ == "__main__":
    x = sp.symbols('x:2')
    t = sp.symbols('t')

    #u0 = -1/3 * x[1]**3 + .1 * x[1] * x[0]
    #u1 = 1/2 * x[0]**2 + x[1]**3 * x[0]
    u0 = 2*sp.erf(x[1])
    u1 = sp.atan(1/5 * x[0] * x[1]) + 2 * x[0] 

    vf = VelocityField(x, [u0, u1])
    dt = 0.001
    timesteps = 100

    target_region = Rectangle(mins=[0, 0], maxes=[1, 1])
    integral_result = vf.volume_time_derivative(target_region)

    fig = plt.figure()
    ax = fig.gca()

    fig_bounds = 5*np.array([-1, 1, -1, 1])

    # Show the v field
    vf.visualize(ax, fig_bounds)

    # Show the initial region
    vis.show_2D_region(ax, target_region, color='red')
    tf_region = vis.RegionBoundaryDiscretization(target_region, n=20)
    for _ in range(timesteps):
        tf_region.flow_backward(vf, dt)
        vis.show_2D_transformed_region(ax, tf_region=tf_region, color='red')

    # Show the PDF
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    visualize_2D_pdf(ax, vf, dt, timesteps, bounds=fig_bounds)

    # Compute integrals
    P_mc = integrators.mc_prob(target_region, vf, dt, timesteps, 100000)
    print("Monte Carlo probability:    ", P_mc)
    P_vegas = integrators.density_mc_integral(target_region, vf, dt, timesteps, 5000)
    print("Vegas integral probability: ", P_vegas)

    # Box algorithm
    def box_propagator(region : Rectangle):
        return region_post_sampling(vf, dt, region, 1000, direction=-1, epsilon=0.1)

    # Show the box algo steps
    #fig, axes = plt.subplots(nrows=1, ncols=timesteps)
    #for ax in axes:
    #    vf.visualize(ax, fig_bounds)
    #P_box_algo = sample_box_flow_algo(target_region, 0.01, vf, dt, timesteps, box_propagator, axes)

    P_box_algo = sample_box_flow_algo(target_region, 0.01, vf, dt, timesteps, box_propagator)
    print("Box algo probability:       ", P_box_algo)

    plt.show()