import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Rectangle
from scipy.stats import multivariate_normal
from scipy import special
from typing import Tuple
from itertools import product

from velocity_field import VelocityField

def standard_multivariate_gaussian_pdf(x : np.ndarray):
    n = x.shape[0]
    norm_const = 1 / ((2 * np.pi) ** (n / 2))
    exponent = -0.5 * np.dot(x, x)
    return norm_const * np.exp(exponent)

def std_gaussian_cdf(x : np.ndarray) -> float:
    """
    Compute the cumulative distribution function (CDF) of the standard multivariate Gaussian
    at point x
    """
    return 0.5 * (1 + special.erf(x / np.sqrt(2)))
    


def std_gaussian_integral_hyperrectangle(region : Rectangle) -> float:
    """
    Compute the integral of the standard multivariate Gaussian distribution
    over a hyperrectangle region
    
    Returns:
    --------
    float
        The probability mass of the standard multivariate Gaussian within the hyperrectangle
    """
    
    # For standard multivariate Gaussian (mean=0, cov=I), 
    # the integral factorizes into a product of univariate integrals
    
    # Convert to cumulative distribution function (CDF) using error function
    # For each dimension: Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
    cdf_maxs = 0.5 * (1 + special.erf(region.maxes / np.sqrt(2)))
    cdf_mins = 0.5 * (1 + special.erf(region.mins / np.sqrt(2)))
    
    # The probability in each dimension is the difference between upper and lower CDFs
    probs_per_dim = cdf_maxs - cdf_mins
    
    # The joint probability is the product of the marginal probabilities
    # because dimensions are independent in standard multivariate Gaussian
    total_prob = np.prod(probs_per_dim)
    
    return total_prob

def std_gaussian_pdf_min_val(region : Rectangle):
    vertices = list(product(*zip(region.mins, region.maxes)))
    min_density = np.inf
    for vertex in vertices:
        density = standard_multivariate_gaussian_pdf(np.array(vertex))
        if density < min_density:
            min_density = density
    return min_density

def pdf(x : np.ndarray, vf : VelocityField, dt : float, timesteps : int, erf_space = False):
    """
    Compute the probability density found by flowing mass backwards to std gaussian
    through a given velocity field
    """
    # Set the initial z to x
    z_t = x
    log_px = 0
    for _ in range(timesteps):
        log_px -= vf.divergence(z_t) * dt
        z_t -= vf.velocity(z_t) * dt
        if erf_space:
            z_t = np.clip(z_t, 0, 1)
    
    if not erf_space:
        log_px += np.log(standard_multivariate_gaussian_pdf(z_t))
    return np.exp(log_px)
    
def visualize_2D_pdf(ax : plt.Axes, vf : VelocityField, dt : float, timesteps : int, bounds : list, resolution = 100, erf_space = False):
    X0, X1 = np.meshgrid(np.linspace(bounds[0], bounds[1], resolution), np.linspace(bounds[2], bounds[3], resolution))
    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            x = np.array([X0[i, j],X1[i, j]])
            if erf_space:
                u = std_gaussian_cdf(x)
                Z[i, j] = standard_multivariate_gaussian_pdf(x) * pdf(np.array(u), vf, dt, timesteps, erf_space=erf_space)
            else:
                Z[i, j] = pdf(np.array(x), vf, dt, timesteps, erf_space=erf_space)
             
    
    #ax.contourf(X0, X1, Z, levels=100, cmap='viridis')
    ax.plot_surface(X0, X1, Z, vmin=0, vmax=Z.max(), cmap='magma')
    ax.set_title("Target Density")
    return ax