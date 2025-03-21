import numpy as np
from scipy import special as sp
from scipy import spatial
import scipy.integrate as spi
import vegas

from problem import Problem

################################## INTEGRATION METHODS ##################################

## Monte Carlo Probability ##

def mc_prob(prob : Problem, region : spatial.Rectangle, k, n_samples):
    # Make sure the region is just over x
    assert region.m == 2

    xk_samples = prob.sample_empirical(k, n_samples)
    contained = (region.mins[:, None] <= xk_samples) & (region.maxes[:, None] >= xk_samples)

    n_contained = np.sum(np.all(contained, axis=0))

    # empirical probability
    return n_contained / n_samples

## Numerical grid integration of the density ##

def density_grid_integral(prob : Problem, region : spatial.Rectangle, k, w_bounds : np.ndarray):
    mins = np.append(w_bounds[::2], region.mins)
    maxes = np.append(w_bounds[1::2], region.maxes)
    bounds = [(l, u) for l, u in zip(mins, maxes)]
    options_x = {'epsabs': 1e-2, 'epsrel': 1e-2, 'limit': 20}  # Increase error tolerance, reduce subdivisions
    options_w = {'epsabs': 1e-2, 'epsrel': 1e-2, 'limit': 100}  # Increase error tolerance, reduce subdivisions
    options = [options_x, options_x, options_w, options_w]
    return spi.nquad(lambda w0, w1, xk0, xk1: prob.p((w0, w1), (xk0, xk1), k), bounds, opts=options)

## Monte Carlo integration of the density

def density_mc_integral(prob : Problem, region : spatial.Rectangle, k, w_bounds : np.ndarray):
    mins = np.append(w_bounds[::2], region.mins)
    maxes = np.append(w_bounds[1::2], region.maxes)
    bounds = [(l, u) for l, u in zip(mins, maxes)]
    #print("bounds: ", bounds)
    #input("...")
    integ = vegas.Integrator(bounds)
    result = integ(lambda wx: prob.p((wx[0], wx[1]), (wx[2], wx[3]), k), nitn=10, neval=10000)
    return result

## Numerical grid sum of the volume ##

def volume_grid_sum(prob : Problem, region : spatial.Rectangle, k, resolution_yx : int, resolution_yw : int):
    YW0, YW1, YX0, YX1 = np.meshgrid(np.linspace(0, 1, resolution_yw), 
                                 np.linspace(0, 1, resolution_yw), 
                                 np.linspace(0, 1, resolution_yx), 
                                 np.linspace(0, 1, resolution_yx), indexing='ij')

    volume = 0

    for idx in np.ndindex((resolution_yw - 1, resolution_yw - 1, resolution_yx - 1, resolution_yx - 1)):
        yw = (YW0[idx], YW1[idx])
        yx = (YX0[idx], YX1[idx])
        w, xk = prob.Gok(*prob.Phi_inv(yw, yx), k)
        if (xk[0] > region.mins[0]) and (xk[0] < region.maxes[0]) and (xk[1] > region.mins[1]) and (xk[1] < region.maxes[1]):
            next_idx = tuple(np.array(idx) + 1)
            dyw0 = YW0[next_idx] - YW0[idx]
            dyw1 = YW1[next_idx] - YW1[idx]
            dyx0 = YX0[next_idx] - YX0[idx]
            dyx1 = YX1[next_idx] - YX1[idx]
            volume += dyw0 * dyw1 * dyx0 * dyx1
    return volume

