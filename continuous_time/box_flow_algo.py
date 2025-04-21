import numpy as np
import sympy as sp
import copy
from scipy.spatial import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from velocity_field import VelocityField
from pdf import std_gaussian_integral_hyperrectangle, standard_multivariate_gaussian_pdf, std_gaussian_pdf_min_val
import visualizers as vis

def sample_box_flow_algo(target_region : Rectangle, sample_volume : float, vf : VelocityField, dt : float, timesteps : int, box_propagator, axes : list[plt.Axes] = None):
    if axes is not None and vf.dim != 2:
        raise ValueError("Cannot visualize algorithm with dim != 2")

    volume_t = target_region.volume()
    region_t = copy.deepcopy(target_region)

    # Move a point forward to the target time step and check if it is contained within the target region
    def point_contained(k : int, z_k : np.ndarray):
        z_k = z_k.copy()

        #print("   point contained: ", k)
        for _ in range(timesteps, k, -1):
            z_k += vf.velocity(z_k) * dt
        #print("target region: ", target_region, " z_k: ", z_k, " inside? ", np.all((z_k >= target_region.mins) & (z_k <= target_region.maxes)))
        #input("...")
        return np.all((z_k >= target_region.mins) & (z_k <= target_region.maxes))
        
    def create_cell_region(center : np.ndarray):
        half_length = sample_volume ** (1 / vf.dim) / 2
        return Rectangle(maxes=(center + half_length), mins=(center - half_length))

    if axes is not None:
        true_region = vis.RegionBoundaryDiscretization(target_region)

    ax = None
    for k in range(timesteps, -1, -1):

        # Visualize current region
        if axes is not None:
            ax = axes[timesteps-k]
            vis.show_2D_region(ax, region_t, color='red', alpha=0.1)
            vis.show_2D_transformed_region(ax, true_region, color='purple')
            true_region.flow_backward(vf, dt)

        # Calculate the free volume estimate
        free_volume = region_t.volume() - volume_t
        print("k: ", k, "region_t vol: ", region_t.volume(), " free vol: ", free_volume, " n_samples: ", np.floor(free_volume / sample_volume))
        #n_samples = int(np.floor(free_volume / sample_volume))
        n_samples = int(np.floor())
        if n_samples < 0:
            print("WARNING: negative free volume")
            n_samples = 0


        # Calculate the volume estimate in the next time step as the volume time derivative of the current box
        volume_tp = volume_t + vf.volume_time_derivative(region_t) * dt

        # Sample centerpoints for the negative space cells
        #print("target region: ", target_region.mins, ", ", target_region.maxes)
        #print("region_t: ", region_t.mins, ", ", region_t.maxes)
        cell_samples = np.random.uniform(low=region_t.mins, high=region_t.maxes, size=(n_samples, vf.dim))
        #print("cell samples: ", cell_samples)

        if k > 0:
            for sample_center in cell_samples:
                if not point_contained(k, sample_center):
                    neg_cell = create_cell_region(sample_center)
                    volume_tp -= vf.volume_time_derivative(neg_cell) * dt

                    if ax:
                        ax.scatter(sample_center[0], sample_center[1], s=1, c='grey')
                        #vis.show_2D_region(ax, neg_cell, color='grey', alpha=1)


            # Calculate the next bounding box
            region_tp = box_propagator(region_t)
            #print("    NEW REGION: ", region_tp)

            # Update the region and volume estimate
            #print("volume_t: ", volume_t, " volume_tp: ", volume_tp)
            volume_t = volume_tp
            region_t = region_tp
        else:
            #print("INTEGRATING GAUSSIAN OVER: ", region_t)
            probability = std_gaussian_integral_hyperrectangle(region_t)
            #print("probability before removing negative cells: ", probability)
            
            mc_integral = 0
            for sample_center in cell_samples:
                if not point_contained(k, sample_center):
                    #probability -= std_gaussian_integral_hyperrectangle(neg_cell)
                    mc_integral += standard_multivariate_gaussian_pdf(sample_center) 

                    if ax:
                        neg_cell = create_cell_region(sample_center)
                        #ax.scatter(sample_center[0], sample_center[1], s=1, c='grey')
                        vis.show_2D_region(ax, neg_cell, color='grey', alpha=1)

            return probability

def naive_box_flow_algo(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, jacobian_bound : np.ndarray, axes : list[plt.Axes] = None):
    region_t = copy.deepcopy(target_region)

    def propagate_region(current_region : Rectangle):
        new_region_mins = current_region.mins.copy()
        new_region_maxes = current_region.maxes.copy()

        #print("Input region: ", current_region.mins, ", ", current_region.maxes)
        
        half_diag = 0.5 * (current_region.maxes - current_region.mins) 

        for i in range(vf.dim):

            center_u = current_region.mins + half_diag + (half_diag[i] * np.eye(1, vf.dim, i)[0])
            center_l = current_region.mins + half_diag - (half_diag[i] * np.eye(1, vf.dim, i)[0])

            #print("half diag: ", half_diag, " jacobian bounds: ", jacobian_bound)
            #print("center u: ", center_u, ", center l: ", center_l)
            vi_center_u = -vf.velocity(center_u, t=None, i=i)
            vi_center_l = -vf.velocity(center_l, t=None, i=i)

            # Check the extrema values over each surface dimension to find the worst case
            extremum = -np.inf # Absolute value of the lipschitz magnitude
            for j in range(vf.dim):
                if j == i:
                    continue
                extremum_j = half_diag[j] * jacobian_bound[i, j]
                if extremum_j > extremum:
                    extremum = extremum_j
                
            # Add the extremum to move the bounds of the new rectangle

            new_region_maxes[i] += (vi_center_u + extremum) * dt
            new_region_mins[i] += (vi_center_l - extremum) * dt
            #print("extremum: ", extremum)
            #print("vi_center_u: ", vi_center_u, " center_l: ", vi_center_l)
            #print(" dim ", i, " new bounds: ", new_region_mins[i], " ", new_region_maxes[i])
            #input("...")
        
        #print("output region: ", new_region_mins, ", ", new_region_maxes)
        return Rectangle(maxes=new_region_maxes, mins=new_region_mins)
    
    if axes is not None:
        true_region = vis.RegionBoundaryDiscretization(target_region)
    ax = None

    for k in range(timesteps, -1, -1):
        # Visualize current region
        if axes is not None:
            ax = axes[timesteps-k]
            vis.show_2D_region(ax, region_t, color='red', alpha=0.1)
            vis.show_2D_transformed_region(ax, true_region, color='purple', alpha=0.5)
            true_region.flow_backward(vf, dt)

        if k > 0:
            region_t = propagate_region(region_t)
        else:
            return std_gaussian_integral_hyperrectangle(region_t)

def smart_box_flow_algo(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, jacobian_bound : np.ndarray, divergence_lipschitz_bounds : np.ndarray, axes : list[plt.Axes] = None):
    def propagate_region(current_region : Rectangle):
        new_region_mins = current_region.mins.copy()
        new_region_maxes = current_region.maxes.copy()

        #print("Input region: ", current_region.mins, ", ", current_region.maxes)
        
        half_diag = 0.5 * (current_region.maxes - current_region.mins) 

        for i in range(vf.dim):

            center_u = current_region.mins + half_diag + (half_diag[i] * np.eye(1, vf.dim, i)[0])
            center_l = current_region.mins + half_diag - (half_diag[i] * np.eye(1, vf.dim, i)[0])

            #print("half diag: ", half_diag, " jacobian bounds: ", jacobian_bound)
            #print("center u: ", center_u, ", center l: ", center_l)
            vi_center_u = -vf.velocity(center_u, t=None, i=i)
            vi_center_l = -vf.velocity(center_l, t=None, i=i)

            # Check the extrema values over each surface dimension to find the worst case
            extremum = -np.inf # Absolute value of the lipschitz magnitude
            for j in range(vf.dim):
                if j == i:
                    continue
                extremum_j = half_diag[j] * jacobian_bound[i, j]
                if extremum_j > extremum:
                    extremum = extremum_j
                
            # Add the extremum to move the bounds of the new rectangle

            new_region_maxes[i] += (vi_center_u + extremum) * dt
            new_region_mins[i] += (vi_center_l - extremum) * dt
            #print("extremum: ", extremum)
            #print("vi_center_u: ", vi_center_u, " center_l: ", vi_center_l)
            #print(" dim ", i, " new bounds: ", new_region_mins[i], " ", new_region_maxes[i])
            #input("...")
        
        #print("output region: ", new_region_mins, ", ", new_region_maxes)
        return Rectangle(maxes=new_region_maxes, mins=new_region_mins)

    def get_min_divergence(region : Rectangle):
        half_diag = 0.5 * (region.maxes - region.mins)
        center_point = half_diag + region.mins
        div_at_center = vf.divergence(center_point)

        extremum = -np.inf # Absolute value of the lipschitz magnitude
        for i in range(vf.dim):
            extremum_i = half_diag[i] * divergence_lipschitz_bounds[i]
            if extremum_i > extremum:
                extremum = extremum_i
        
        return div_at_center - extremum

    if axes is not None:
        true_region = vis.RegionBoundaryDiscretization(target_region)
    ax = None

    region_t = copy.deepcopy(target_region)

    true_region_vol_bound = target_region.volume()

    for k in range(timesteps, -1, -1):
        # Visualize current region
        if axes is not None:
            ax = axes[timesteps-k]
            vis.show_2D_region(ax, region_t, color='red', alpha=0.1)
            vis.show_2D_transformed_region(ax, true_region, color='purple', alpha=0.5)
            true_region.flow_backward(vf, dt)

        if k > 0:
            min_div = get_min_divergence(region_t)
            true_region_vol_bound += dt * (vf.volume_time_derivative(region_t) - min_div * (region_t.volume() - true_region_vol_bound))

            region_t = propagate_region(region_t)
        else:
            return std_gaussian_integral_hyperrectangle(region_t) - true_region_vol_bound * std_gaussian_pdf_min_val(region_t)