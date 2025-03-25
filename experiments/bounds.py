import numpy as np
from scipy import special as sp
from scipy import spatial
import matplotlib.pyplot as plt
from itertools import product

from problem import Problem
import distribution_functions as fcns


def std_gaussian_quantile_bounds(min : float, max : float, mu = 0, sigma = 1):
    assert min < max

    # Edge cases near 0 or 1
    if min < 1e-10:
        min = 1e-10
    if max > 1-1e-10:
        max = 1-1e-10

    if max <= 0.5: # region is on the LHS of 1/2
        u_slope = fcns.std_gaussian_quantile_deriv(min, mu, sigma)
        u_intercept = fcns.std_gaussian_quantile(min, mu, sigma) - u_slope * min
        l_slope = (fcns.std_gaussian_quantile(max, mu, sigma) - fcns.std_gaussian_quantile(min, mu, sigma)) / (max - min)
        l_intercept = fcns.std_gaussian_quantile(min, mu, sigma) - l_slope * min
    elif min >= 0.5: # region is on the RHS of 1/2
        l_slope = fcns.std_gaussian_quantile_deriv(max, mu, sigma)
        l_intercept = fcns.std_gaussian_quantile(max, mu, sigma) - l_slope * max
        u_slope = (fcns.std_gaussian_quantile(max, mu, sigma) - fcns.std_gaussian_quantile(min, mu, sigma)) / (max - min)
        u_intercept = fcns.std_gaussian_quantile(max, mu, sigma) - u_slope * max
    else: # region is in between
        assert False
    
    return (l_slope, l_intercept), (u_slope, u_intercept)

def rect_intersection(r1 : spatial.Rectangle, r2 : spatial.Rectangle):
    assert r1.m == r2.m
    intersection_min = np.maximum(r1.mins, r2.mins)
    intersection_max = np.minimum(r1.maxes, r2.maxes)

    if np.any(intersection_min > intersection_max):
        return None
    
    return spatial.Rectangle(intersection_min, intersection_max)

# Translate and scale rectangle
def transform_rect(r : spatial.Rectangle, scale : np.ndarray, translation : np.ndarray):
    return spatial.Rectangle(r.mins * scale + translation, r.maxes * scale + translation)

class GaussianCellPropagator:
    def __init__(self, prob : Problem, lipschitz_constant : float):
        self.prob = prob
        self.lipschitz_constant = lipschitz_constant

    def propagate_state_cell(self, cell : spatial.Rectangle, k : int):
        """
        Compute the post() of a cell in the state-noise space using Lipschitz theorem
        """
        composed_lipschitz_const = self.lipschitz_constant**k

        # Create a cell for just the state since the noise dimensions stay the same
        state_cell = spatial.Rectangle(cell.mins[self.prob.m:], cell.maxes[self.prob.m:])

        pre_cell_radius = 0.5 * np.linalg.norm(state_cell.maxes - state_cell.mins)
        vertices = list()
        for vertex_coords in product(*[(state_cell.mins[i], state_cell.maxes[i]) for i in range(self.prob.n)]):
            vertices.append(np.array(vertex_coords))
        
        # Propagate each vertex through the dynamics
        print("vertex[0]: ", vertices[0])
        post_vertices = np.array([self.prob.g(x) for x in vertices])

        min_coords = np.min(post_vertices, axis=0)
        max_coords = np.max(post_vertices, axis=0)

        # Expand the post cell by the radius
        expanded_min_coords = min_coords - composed_lipschitz_const * pre_cell_radius
        expanded_max_coords = max_coords + composed_lipschitz_const * pre_cell_radius

        return spatial.Rectangle(expanded_min_coords, expanded_max_coords)


#def gaussian_post_cell_intersection_vol(region : spatial.Rectangle, grid_cell : spatial.Rectangle, mu : np.ndarray, sigma : np.ndarray)















if __name__ == "__main__":
    pass
    #xvec = np.linspace(0, 1, 100)

    #fig = plt.figure()        
    #ax = fig.gca()

    #ax.plot(xvec, fcns.std_gaussian_quantile(xvec))

    ## Region
    #min, max = 0.8, 0.9
    #mu, sigma = 0, 1
    #l_bounds, u_bounds = std_gaussian_quantile_bounds(min, max, mu, sigma)
    #region_xvec = np.linspace(min, max, 10)

    #ax.plot(region_xvec, l_bounds[0] * region_xvec + l_bounds[1], linestyle='--')
    #ax.plot(region_xvec, u_bounds[0] * region_xvec + u_bounds[1], linestyle='--')
    #ax.axvline(x=min, color='k')
    #ax.axvline(x=max, color='k')
    

    #plt.show()
