import numpy as np
from scipy import special as sp

# Gaussian
def std_gaussian_quantile(x, mu = 0, sigma = 1):
    return mu + sigma * np.sqrt(2) * sp.erfinv(2*x - 1)

def std_gaussian_cdf(x, mu = 0, sigma = 1):
    return 0.5 * (1 + sp.erf((x - mu) / (sigma * np.sqrt(2))))

def std_gaussian_quantile_deriv(x, mu = 0, sigma = 1):
    return np.sqrt(2 * np.pi) * sigma * np.exp(sp.erfinv(2*x - 1)**2)

def std_gaussian_cdf_deriv(x, mu = 0, sigma = 1):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Laplace
def laplace_quantile(x, mu = 0, b = 1):
    if x <= 0.5:
        return mu + b * np.log(2 * x) 
    else:
        return mu - b * np.log(2 - 2 * x) 

def laplace_cdf(x, mu = 0, b = 1):
    if x <= mu:
        return 0.5 * np.exp(1 / b * (x - mu))
    else:
        return 1 - 0.5 * np.exp(-1 / b * (x - mu))

def laplace_quantile_deriv(x, mu = 0, b = 1):
    if x <= 0.5:
        return b / x
    else:
        return -b / (1 - x)

def laplace_cdf_deriv(x, mu = 0, b = 1):
    return 0.5 * np.exp(1 / b * (x - mu))

