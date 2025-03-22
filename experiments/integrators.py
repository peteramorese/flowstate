import numpy as np
from scipy import special as sp
from scipy import spatial
import scipy.integrate as spi
import vegas

from problem import Problem

################################## INTEGRATION METHODS ##################################

## Monte Carlo Probability ##

def mc_prob(prob : Problem, region : spatial.Rectangle, k, n_samples):
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
    options_x = {'epsabs': 1e-2, 'epsrel': 1e-2, 'limit': 30}  # Increase error tolerance, reduce subdivisions
    options_w = {'epsabs': 1e-2, 'epsrel': 1e-2, 'limit': 30}  # Increase error tolerance, reduce subdivisions
    options = prob.m * [options_w] + prob.n * [options_x]
    return spi.nquad(lambda *wx: prob.p(tuple(wx[:prob.m]), tuple(wx[prob.m:]), k), bounds, opts=options)

## Monte Carlo integration of the density

def density_mc_integral(prob : Problem, region : spatial.Rectangle, k, w_bounds : np.ndarray):
    mins = np.append(w_bounds[::2], region.mins)
    maxes = np.append(w_bounds[1::2], region.maxes)
    bounds = [(l, u) for l, u in zip(mins, maxes)]
    #print("bounds: ", bounds)
    #input("...")
    integ = vegas.Integrator(bounds)
    result = integ(lambda wx: prob.p(tuple(wx[:prob.m]), tuple(wx[prob.m:]), k), nitn=10, neval=10000)
    return result

## Numerical grid sum of the volume ##

def volume_grid_sum(prob : Problem, region : spatial.Rectangle, k, resolution_yx : int, resolution_yw : int):
    linspaces = prob.m * [np.linspace(0, 1, resolution_yw)] + prob.n * [np.linspace(0, 1, resolution_yx)]
    YWX = np.meshgrid(*linspaces, indexing='ij')

    volume = 0

    for idx in np.ndindex(tuple(prob.m * [resolution_yw - 1] + prob.n * [resolution_yx - 1])):
        yw = tuple(YWi[idx] for YWi in YWX[:prob.m])
        yx = tuple(YXi[idx] for YXi in YWX[prob.m:])
        #print("yx shape: ", yx.shape)
        w, xk = prob.Gok(*prob.Phi_inv(yw, yx), k)
        #print("xk: ", xk)
        xk = np.array(xk)
        #print("region.mins: ", region.mins, ", xk: ", xk)
        if np.all((region.mins <= xk) & (region.maxes >= xk)):
            next_idx = tuple(np.array(idx) + 1)
            
            # Compute the volume of the cell
            d_volume = 1
            for mesh in YWX:
                #print("diff: ", mesh[next_idx] - mesh[idx])
                d_volume *= mesh[next_idx] - mesh[idx]

            volume += d_volume
    return volume

