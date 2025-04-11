import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle
from itertools import product

def evaluate_integral(antideriv, region : Rectangle):
    bounds = [(l, u) for l, u in zip(region.mins, region.maxes)]
    
    vertices = list(product(*bounds))
    
    integral = 0

    for vertex in vertices:
        # Count how many lower bounds are used in this vertex
        num_lower_bounds = sum(1 for i, v in enumerate(vertex) if v == bounds[i][0])
    
        # Assign the sign based on parity
        sign = (-1) ** num_lower_bounds
        
        # Add the contribution of this vertex
        integral += sign * antideriv(*np.array(vertex))
    return integral

class VelocityField:
    def __init__(self, x_symbols : list, div_anti_derivatives : list, t_symbol : sp.Symbol = None):
        assert len(x_symbols) == len(div_anti_derivatives)
        self.dim = len(div_anti_derivatives)
        self.x_symbols = x_symbols
        self.div_anti_derivatives = div_anti_derivatives

        symbols = tuple(self.x_symbols) + (t_symbol,) if t_symbol else tuple(self.x_symbols) 
        self.div_anti_derivatives_num = [sp.lambdify(symbols, antid, modules='numpy') for antid in self.div_anti_derivatives]
        self.t_symbol = t_symbol

        self.v = list()
        for i, div_anti_deriv in enumerate(self.div_anti_derivatives):
            mixed_partial = div_anti_deriv
            for j, xj_sym in enumerate(self.x_symbols):
                if j != i: # Skip the divergence derivative to get just the velocity
                    mixed_partial = sp.diff(mixed_partial, xj_sym)

            print(f"v{i}(x, t) = ", mixed_partial)
            
            # Convert to numerical function
            vi = sp.lambdify(symbols, mixed_partial, modules='numpy')
            self.v.append(vi)

    def velocity(self, x : np.ndarray, t : float = None, i : int = None):
        assert (t == None) == (self.t_symbol == None)
        assert len(x) == self.dim

        if i:
            return self.v[i](*x, t) if self.t_symbol else self.v[i](*x)
        else:
            vel = np.zeros(x.shape)
            for i, vi in enumerate(self.v):
                vel[i] = vi(*x, t) if self.t_symbol else vi(*x)
            return vel
    
    def volume_time_derivative(self, region : Rectangle):
        # Evaluate the antiderivative of the divergence of the velocity field which
        # is the time rate of change of the volume of the input region in the v field
        
        # Integral of divergence splits in to n integrals over each individual antiderivative
        integral = 0
        for antideriv in self.div_anti_derivatives_num:
            print("  integral result = ", evaluate_integral(antideriv, region))
            integral += evaluate_integral(antideriv, region)
        return integral

    def visualize(self, ax : plt.Axes, bounds, disc = 20, t = None):
        assert self.dim == 2
        X0, X1 = np.meshgrid(np.linspace(bounds[0], bounds[1], disc), np.linspace(bounds[2], bounds[3], disc));
        V0 = self.v[0](X0, X1)
        V1 = self.v[1](X0, X1)

        ax.quiver(X0, X1, V0, V1)
        ax.set_xlabel("x_0")
        ax.set_xlabel("x_1")
        ax.set_title("Velocity Field")
        ax.axis('equal')
        ax.grid(True)

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

## Visualization ##
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

if __name__ == "__main__":
    x = sp.symbols('x:2')
    t = sp.symbols('t')

    u0 = -1/3 * x[1]**3 + .1 * x[1] * x[0]
    u1 = 1/2 * x[0]**2 + x[1]**3 * x[0]

    vf = VelocityField(x, [u0, u1])

    dt = 0.3

    initial_region = Rectangle([0.3, -0.3], [0.2, -0.4])
    integral_result = vf.volume_time_derivative(initial_region)
    print("Integral result: ", integral_result)

    fig = plt.figure()
    ax = fig.gca()

    # Show the v field
    vf.visualize(ax, [-1, 1, -1, 1])

    # Show the initial region
    show_2D_region(ax, initial_region, color='red')
    tf_region = RegionBoundaryDiscretization(initial_region, n=20)

    timesteps = 10
    for _ in range(timesteps):
        tf_region.flow(vf, dt)
        show_2D_transformed_region(ax, tf_region=tf_region, color='red')


    plt.show()