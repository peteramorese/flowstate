import numpy as np
from scipy.spatial import Rectangle
import vegas

from velocity_field import VelocityField
from pdf import pdf

def mc_prob(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, n_samples : int = 10000):
    # sample from standard gaussian initial distribution
    z_t_samples = np.random.randn(n_samples, vf.dim)

    def point_contained(z_t : np.ndarray):
        for _ in range(timesteps):
            z_t += vf.velocity(z_t) * dt
        return np.all((z_t >= target_region.mins) & (z_t <= target_region.maxes))
        
    n_contained = 0
    for z_0 in z_t_samples:
        if point_contained(z_0):
            n_contained += 1

    # empirical probability
    return n_contained / n_samples

def density_mc_integral(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, n_eval : int = 10000):
    bounds = [(l, u) for l, u in zip(target_region.mins, target_region.maxes)]
    integ = vegas.Integrator(bounds)
    result = integ(lambda x : pdf(x, vf, dt, timesteps), nitn=10, neval=n_eval)
    return result