import numpy as np
import sympy as sp
from scipy.spatial import Rectangle

from velocity_field import VelocityField
from pdf import std_gaussian_integral_hyperrectangle

def sample_box_flow_algo(target_region : Rectangle, sample_volume : float, vf : VelocityField, dt : float, timesteps : int, box_propagator):
    volume_t = target_region.volume()
    region_t = target_region

    # Move a point forward to the target time step and check if it is contained within the target region
    def point_contained(k : int, z_t : np.ndarray):
        for _ in range(timesteps, k, -1):
            z_t += vf.velocity(z_t) * dt
        return np.all(z_t >= target_region.mins & z_t <= target_region.maxes)
        
    def create_cell_region(center : np.ndarray):
        half_length = sample_volume ** (1 / vf.dim) / 2
        return Rectangle(maxes=center + half_length, mins=center - half_length)

    for k in range(timesteps, 0, -1):
        # Calculate the free volume estimate
        free_volume = region_t.volume() - volume_t
        n_samples = np.floor(free_volume / sample_volume)


        # Calculate the volume estimate in the next time step as the volume time derivative of the current box
        volume_tp = vf.volume_time_derivative(region_t) * dt

        # Sample centerpoints for the negative space cells
        cell_samples = np.random.uniform(low=region_t.mins, high=region_t.maxes, size=(n_samples, vf.dim))
        neg_cell_samples = list()

        if k > 1:
            for sample_center in cell_samples:
                if point_contained(k, sample_center):
                    neg_cell = create_cell_region(sample_center)
                    volume_tp -= vf.volume_time_derivative(neg_cell) * dt

            # Calculate the next bounding box
            region_tp = box_propagator(region_t)

            # Update the region and volume estimate
            volume_t = volume_tp
            region_t = region_tp
        else:
            probability = std_gaussian_integral_hyperrectangle(region_t)
            for sample_center in cell_samples:
                if point_contained(k, sample_center):
                    neg_cell = create_cell_region(sample_center)
                    probability -= std_gaussian_integral_hyperrectangle(neg_cell)
        
        return probability

    
