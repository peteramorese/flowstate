import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle
from itertools import product

from velocity_field import VelocityField
from pdf import visualize_2D_pdf

def region_post_sampling(vf : VelocityField, dt : float, region : Rectangle, n_samples : int, epsilon : float):
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

    new_region = region

    for d in range(vf.dim):
        vi = lambda x : vf.velocity(x, t=None, i=d)

        # Compute the "max" deviation of the upper surface
        new_region.maxes[d] += dt * sample_extrema_on_surface(vi, d, region.maxes[d], True)
        # Compute the "max" deviation of the lower surface
        new_region.mins[d] += dt * sample_extrema_on_surface(vi, d, region.mins[d], False)

    return new_region

################################## Visualization ##################################
def show_2D_region(ax : plt.Axes, region : Rectangle, color='blue', alpha=0.2):
    """
    Plots a 2D region
    """
    if region.m != 2:
        raise ValueError("Region must be 2D.")
    
    width = region.maxes[0] - region.mins[0]
    height = region.maxes[1] - region.mins[1]
    
    rect = patches.Rectangle(
        (region.mins[0], region.mins[1]), width, height,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha
    )
    ax.add_patch(rect)
    
    return ax

class RegionBoundaryDiscretization:
    def __init__(self, region : Rectangle, n=20):
        bottom = np.linspace(region.mins, [region.maxes[0], region.mins[1]], n, endpoint=False)
        right = np.linspace([region.maxes[0], region.mins[1]], region.maxes, n, endpoint=False)
        top = np.linspace(region.maxes, [region.mins[0], region.maxes[1]], n, endpoint=False)
        left = np.linspace([region.mins[0], region.maxes[1]], region.mins, n, endpoint=False)
        self.boundary_points = np.vstack([bottom, right, top, left])
    
    def flow(self, vf : VelocityField, dt : float, t : float = None):
        self.boundary_points += np.array([dt * vf.velocity(bi, t) for bi in self.boundary_points])

def show_2D_transformed_region(ax : plt.Axes, tf_region : RegionBoundaryDiscretization, color='blue', alpha=0.2):
    #ax.scatter(tf_region.boundary_points[0], tf_region.boundary_points[1])
    polygon = patches.Polygon(
        tf_region.boundary_points,
        closed=True,
        edgecolor=color,
        facecolor=color,
        linewidth=1.5,
        alpha=alpha
    )
    ax.add_patch(polygon)

###################################################################################

if __name__ == "__main__":
    x = sp.symbols('x:2')
    t = sp.symbols('t')

    #u0 = -1/3 * x[1]**3 + .1 * x[1] * x[0]
    #u1 = 1/2 * x[0]**2 + x[1]**3 * x[0]
    u0 = 2*sp.erf(x[1]) 
    u1 = sp.atan(1/5 * x[0] * x[1])

    vf = VelocityField(x, [u0, u1])

    dt = 0.1

    initial_region = Rectangle([0.3, -0.3], [0.2, -0.4])
    integral_result = vf.volume_time_derivative(initial_region)
    print("Integral result: ", integral_result)

    fig = plt.figure()
    ax = fig.gca()

    fig_bounds = 5*np.array([-1, 1, -1, 1])

    # Show the v field
    vf.visualize(ax, fig_bounds)

    # Show the initial region
    show_2D_region(ax, initial_region, color='red')
    tf_region = RegionBoundaryDiscretization(initial_region, n=20)

    timesteps = 10
    for _ in range(timesteps):
        tf_region.flow(vf, dt)
        show_2D_transformed_region(ax, tf_region=tf_region, color='red')

    # Show the PDF
    fig = plt.figure()
    ax = fig.gca()

    visualize_2D_pdf(ax, vf, dt, timesteps, bounds=fig_bounds)

    plt.show()