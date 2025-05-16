import numpy as np
import sympy as sp
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle
from itertools import product

from velocity_field import VelocityField
from pdf import visualize_2D_pdf, std_gaussian_integral_hyperrectangle, std_gaussian_cdf
from box_flow_algo import naive_box_flow_algo, smart_box_flow_algo, evaluate_on_grid, min_div_integ_bound, adversarial_div_integ_bound
import integrators
import visualizers as vis



if __name__ == "__main__":
    x = sp.symbols('x:2')
    t = sp.symbols('t')

    # Nonlinear field

    # Constants
    a00, a01, a10, a11 = 0.1, 6, -1, 5

    # v0 = du0/dx1 = a00 / (1 + exp(-x[0])) - a01 * cos(a01 * x[1])
    u0 = a00 * x[1] / (1 + sp.exp(-x[0])) + sp.sin(a01 * x[1])
    # v1 = du1/dx0 = a10 * exp(x[0]) / (exp(x[0]) + 1) + a11 / (1 + exp(-x[1]))
    u1 = a10 * sp.log(sp.exp(x[0]) + 1) + a11 * x[0] / (1 + sp.exp(-x[1])) 

    # Lipschitz constants for velocity
    velocity_jacobian_bound = np.abs(np.array([
        [1/4 * np.abs(a00), np.abs(a01)],             # L(v0 wrt x0), L(v0 wrt x1)
        [np.abs(a10), 1/4 * np.abs(a11)]      # L(v1 wrt x0), L(v1 wrt x1)
    ]))
    divergence_lipschitz_bounds = np.abs(np.array([
        0.1 * np.abs(a00) + 0,              # L(div(v) wrt x0) = L(dv0dx0 wrt x0) + L(dv1dx1 wrt x0)
        0 + 0.1 * np.abs(a11)               # L(div(v) wrt x1) = L(dv0dx0 wrt x1) + L(dv1dx1 wrt x1)
    ]))
    velocity_magnitude_bounds = np.abs(np.array([
        np.abs(a00) + np.abs(a01),            # >= |v0|
        np.abs(a10) + np.abs(a11)   # >= |v1|
    ]))

    # Create the box field using the boundary condition envelopes
    box_u0 = (x[0] - x[0]**2) * u0
    box_u1 = (x[1] - x[1]**2) * u1

    # Create the box field velocity jacobian bound
    box_velocity_jacobian_bound = velocity_jacobian_bound + 4 * np.diag(velocity_magnitude_bounds)
    box_divergence_lipschitz_bounds = np.abs(np.array([
        (2 * velocity_magnitude_bounds[0] + 8 * velocity_jacobian_bound[0, 0] + 0.1 * a00) + (4 * velocity_jacobian_bound[1, 0] + 0),
        (2 * velocity_magnitude_bounds[1] + 8 * velocity_jacobian_bound[1, 1] + 0.1 * a11) + (4 * velocity_jacobian_bound[0, 1] + 0),
    ]))

    ## Trivial field
    #u0 = x[1]
    #u1 = x[0]
    #jacobian_bound = np.array([[0, 0], [0, 0]])
    #divergence_lipschitz_bounds = np.array([0, 0])

    vf = VelocityField(x, [box_u0, box_u1])
    dt = 0.001
    timesteps = 300
    #dt = 0.0001
    #timesteps = 10000

    target_region = Rectangle(mins=[0, 0], maxes=[1, 1])

    def convert_region_to_erf_space(region : Rectangle):
        new_mins = [std_gaussian_cdf(m) for m in region.mins]
        new_maxes = [std_gaussian_cdf(m) for m in region.maxes]
        return Rectangle(mins=new_mins, maxes = new_maxes)

    target_region = convert_region_to_erf_space(target_region)

    fig = plt.figure()
    ax_vf = fig.gca()

    fig_bounds = 5 * np.array([-1, 1, -1, 1])
    erf_space_bounds = np.array([0, 1, 0, 1])

    # Show the v field
    vf.visualize(ax_vf, erf_space_bounds)

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

    visualize_2D_pdf(ax, vf, dt, timesteps, bounds=fig_bounds, erf_space=True, resolution=50)

    # Compute integrals
    N = 100
    t_start = time.time()
    P_mc = integrators.mc_prob(target_region, vf, dt, timesteps, N, erf_space=True)
    t_end = time.time() - t_start
    confidence_bound = integrators.calculate_confidence_bounds(P_mc, N, alpha=0.05, method='hoeffding')
    print(t_end, "s  Monte Carlo probability:            ", P_mc, " (upper 95 conf bound: ", confidence_bound[1], ")")
    #P_vegas = integrators.density_mc_integral(target_region, vf, dt, timesteps, 5000, erf_space=True)
    #print("Vegas integral probability:         ", P_vegas)

    # Naive box algorithm
    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, erf_space_bounds)
    axes = None
    P_box_algo_naive = naive_box_flow_algo(target_region, vf, dt, timesteps, box_velocity_jacobian_bound, axes=axes, erf_space=True)
    print("Naive box algo probability:         ", P_box_algo_naive)

    # Smart box algorithm
    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, erf_space_bounds)
    axes = None
    t_start = time.time()
    P_box_algo_smart_md = smart_box_flow_algo(target_region, vf, dt, timesteps, box_velocity_jacobian_bound, box_divergence_lipschitz_bounds, divergence_integrator=min_div_integ_bound, erf_space=True, axes=axes)
    t_end = time.time() - t_start
    print(t_end, "s Smart box algo probability (min div):         ", P_box_algo_smart_md)

    t_start = time.time()
    P_box_algo_smart_ad = smart_box_flow_algo(target_region, vf, dt, timesteps, box_velocity_jacobian_bound, box_divergence_lipschitz_bounds, divergence_integrator=adversarial_div_integ_bound, erf_space=True, axes=axes)
    t_end = time.time() - t_start
    print(t_end, "s Smart box algo probability (adv div):         ", P_box_algo_smart_ad)

    #fig, axes = plt.subplots(nrows=1, ncols=(timesteps + 1))
    #for ax in axes:
    #    vf.visualize(ax, erf_space_bounds)
    #algo = lambda region_cell, axs : smart_box_flow_algo(region_cell, vf, dt, timesteps, jacobian_bound, divergence_lipschitz_bounds, axes=axs)
    #P_box_algo_smart_grid = evaluate_on_grid(target_region, resolution=30, algorithm=algo, axes=axes)
    #print("Smart box algo (grid) probability:  ", P_box_algo_smart_grid)

    #evolved_trivial_region = Rectangle(mins=(target_region.mins - dt * timesteps), maxes=(target_region.maxes - dt * timesteps))
    #print("\nevolved trivial region: ",evolved_trivial_region)
    #vis.show_2D_region(ax_vf, evolved_trivial_region)
    #trivial_field_prob = std_gaussian_integral_hyperrectangle(evolved_trivial_region)
    #print("trivial field prob: ", trivial_field_prob)




    plt.show()