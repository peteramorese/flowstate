import numpy as np
import sympy as sp
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle
from itertools import product

from velocity_field import VelocityField
from pdf import visualize_2D_pdf, std_gaussian_integral_hyperrectangle
from box_flow_algo import naive_box_flow_algo, smart_box_flow_algo, evaluate_on_grid
import integrators
import visualizers as vis



if __name__ == "__main__":
    x = sp.symbols('x:2')
    t = sp.symbols('t')

    # Nonlinear field

    # Constants
    a00, a10, a11 = 0.5, -1, 2

    u0 = a00 * x[1] / (1 + sp.exp(-x[0])) + sp.sin(x[1])
    u1 = a10 * sp.log(sp.exp(x[0]) + 1) + a11 * x[0] / (1 + sp.exp(-x[1])) 

    jacobian_bound = np.abs(np.array([
        [1/4 * a00, 1],             # L(v0 wrt x0), L(v0 wrt x1)
        [1/4 * a10, 1/4 * a11]      # L(v1 wrt x0), L(v1 wrt x1)
    ]))
    divergence_lipschitz_bounds = np.abs(np.array([
        0.1 * a00 + 0,              # L(div(v) wrt x0) = L(dv0dx0 wrt x0) + L(dv1dx1 wrt x0)
        0 + 0.1 * a11               # L(div(v) wrt x1) = L(dv0dx0 wrt x1) + L(dv1dx1 wrt x1)
    ]))

    ## Trivial field
    #u0 = x[1]
    #u1 = x[0]
    #jacobian_bound = np.array([[0, 0], [0, 0]])
    #divergence_lipschitz_bounds = np.array([0, 0])

    vf = VelocityField(x, [u0, u1])
    dt = 0.1
    timesteps = 10

    target_region = Rectangle(mins=[1, 1], maxes=[2, 2])
    #target_region = Rectangle(mins=[-100, -100], maxes=[100, 100])

    fig = plt.figure()
    ax_vf = fig.gca()

    fig_bounds = 5*np.array([-1, 1, -1, 1])

    # Show the v field
    vf.visualize(ax_vf, fig_bounds)

    # Show the initial region
    vis.show_2D_region(ax_vf, target_region, color='red')
    tf_region = vis.RegionBoundaryDiscretization(target_region, n=20)
    alpha_values = np.linspace(0.3, 0.05, timesteps)
    for k in range(timesteps):
        tf_region.flow_backward(vf, dt)
        vis.show_2D_transformed_region(ax_vf, tf_region=tf_region, color='red', alpha=alpha_values[k])
        #vis.show_2D_transformed_region(ax_vf, tf_region=tf_region, color='green', alpha=alpha_values[-(k + 1)])
    vis.show_2D_transformed_region(ax_vf, tf_region=tf_region, color='green', alpha=0.7)

    # Show the PDF
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    visualize_2D_pdf(ax, vf, dt, timesteps, bounds=fig_bounds)

    # Compute integrals
    P_mc = integrators.mc_prob(target_region, vf, dt, timesteps, 100000)
    print("Monte Carlo probability:            ", P_mc)
    P_vegas = integrators.density_mc_integral(target_region, vf, dt, timesteps, 5000)
    print("Vegas integral probability:         ", P_vegas)

    # Naive box algorithm
    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, fig_bounds)
    axes = None
    P_box_algo_naive = naive_box_flow_algo(target_region, vf, dt, timesteps, jacobian_bound, axes=axes)
    print("Naive box algo probability:         ", P_box_algo_naive)

    # Smart box algorithm
    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, fig_bounds)
    axes = None
    P_box_algo_smart = smart_box_flow_algo(target_region, vf, dt, timesteps, jacobian_bound, divergence_lipschitz_bounds, axes=axes)
    print("Smart box algo probability:         ", P_box_algo_smart)

    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, fig_bounds)
    #algo = lambda region_cell, axs : smart_box_flow_algo(region_cell, vf, dt, timesteps, jacobian_bound, divergence_lipschitz_bounds, axes=axs)
    #P_box_algo_smart_grid = evaluate_on_grid(target_region, resolution=30, algorithm=algo, axes=axes)
    #print("Smart box algo (grid) probability:  ", P_box_algo_smart_grid)

    #evolved_trivial_region = Rectangle(mins=(target_region.mins - dt * timesteps), maxes=(target_region.maxes - dt * timesteps))
    #print("\nevolved trivial region: ",evolved_trivial_region)
    #vis.show_2D_region(ax_vf, evolved_trivial_region)
    #trivial_field_prob = std_gaussian_integral_hyperrectangle(evolved_trivial_region)
    #print("trivial field prob: ", trivial_field_prob)




    plt.show()