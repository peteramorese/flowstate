import numpy as np
import sympy as sp
import copy
from scipy.spatial import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from velocity_field import VelocityField
from pdf import std_gaussian_integral_hyperrectangle
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
    for k in range(timesteps, 0, -1):
        # Visualize current region
        if axes is not None:
            ax = axes[timesteps-k]
            vis.show_2D_region(ax, region_t, color='red', alpha=0.1)
            vis.show_2D_transformed_region(ax, true_region, color='purple')
            true_region.flow_backward(vf, dt)

        # Calculate the free volume estimate
        free_volume = region_t.volume() - volume_t
        print("k: ", k, "region_t vol: ", region_t.volume(), " free vol: ", free_volume, " n_samples: ", np.floor(free_volume / sample_volume))
        n_samples = int(np.floor(free_volume / sample_volume))
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

        if k > 1:
            for sample_center in cell_samples:
                if not point_contained(k, sample_center):
                    neg_cell = create_cell_region(sample_center)
                    volume_tp -= vf.volume_time_derivative(neg_cell) * dt

                    if ax:
                        ax.scatter(sample_center[0], sample_center[1], s=1, c='grey')
                        #vis.show_2D_region(ax, neg_cell, color='grey', alpha=1)


            # Calculate the next bounding box
            region_tp = box_propagator(region_t)

            # Update the region and volume estimate
            print("volume_t: ", volume_t, " volume_tp: ", volume_tp)
            volume_t = volume_tp
            region_t = region_tp
        else:
            probability = std_gaussian_integral_hyperrectangle(region_t)
            print("probability before removing negative cells: ", probability)
            for sample_center in cell_samples:
                if not point_contained(k, sample_center):
                    neg_cell = create_cell_region(sample_center)
                    probability -= std_gaussian_integral_hyperrectangle(neg_cell)
            return probability

    
