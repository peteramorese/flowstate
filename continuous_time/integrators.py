import numpy as np
from scipy.spatial import Rectangle
from scipy import optimize
import vegas

from velocity_field import VelocityField
from pdf import pdf

def mc_prob(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, n_samples : int = 10000, erf_space = False):
    # sample from standard gaussian initial distribution
    if erf_space:
        z_t_samples = np.random.uniform(0, 1, (n_samples, vf.dim))
    else:
        z_t_samples = np.random.randn(n_samples, vf.dim)

    def point_contained(z_t : np.ndarray):
        for _ in range(timesteps):
            z_t += vf.velocity(z_t) * dt
            if erf_space:
                z_t = np.clip(z_t, 0, 1)
        return np.all((z_t >= target_region.mins) & (z_t <= target_region.maxes))
        
    n_contained = 0
    for z_0 in z_t_samples:
        if point_contained(z_0):
            n_contained += 1

    # empirical probability
    return n_contained / n_samples

def density_mc_integral(target_region : Rectangle, vf : VelocityField, dt : float, timesteps : int, n_eval : int = 10000, erf_space = False):
    bounds = [(l, u) for l, u in zip(target_region.mins, target_region.maxes)]
    integ = vegas.Integrator(bounds)
    result = integ(lambda x : pdf(x, vf, dt, timesteps, erf_space=erf_space), nitn=10, neval=n_eval)
    return result

def calculate_confidence_bounds(p_hat, n, alpha=0.05, method='bernstein'):
    """
    Calculate concentration bounds for Monte Carlo probability estimation.
    
    Parameters:
    -----------
    p_hat : float
        The empirical estimate of the probability.
    n : int
        Number of samples.
    alpha : float
        Desired confidence level (e.g., 0.05 for 95% confidence).
    method : str
        Concentration inequality to use: 'hoeffding', 'bernstein', or 'chernoff'.
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound) for the probability estimate.
    """
    if not 0 <= p_hat <= 1:
        raise ValueError("p_hat must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    if method == 'hoeffding':
        # Hoeffding's inequality
        epsilon = np.sqrt(np.log(2/alpha) / (2 * n))
        lower_bound = max(0, p_hat - epsilon)
        upper_bound = min(1, p_hat + epsilon)
        
    elif method == 'bernstein':
        # Bernstein's inequality
        # We need to solve for epsilon in:
        # alpha = 2*exp(-n*epsilon^2 / (2*p_hat*(1-p_hat) + 2*epsilon/3))
        
        def bernstein_equation(eps):
            if p_hat == 0 or p_hat == 1:
                variance_term = 1e-10  # Small value to avoid division by zero
            else:
                variance_term = 2 * p_hat * (1 - p_hat)
            return 2 * np.exp(-n * eps**2 / (variance_term + 2*eps/3)) - alpha
        
        # Find the root of this equation
        try:
            epsilon = optimize.brentq(bernstein_equation, 0, 1)
        except ValueError:
            epsilon = np.sqrt(np.log(2/alpha) / (2 * n))  # Fallback to Hoeffding
        
        lower_bound = max(0, p_hat - epsilon)
        upper_bound = min(1, p_hat + epsilon)
        
    elif method == 'chernoff':
        # Chernoff bounds
        # Different formulas for upper and lower tails
        
        if p_hat == 0:
            # If empirical probability is 0, use special case
            upper_delta = 3 * np.log(1/alpha) / n
            lower_bound = 0
            upper_bound = min(1, upper_delta)
        elif p_hat == 1:
            # If empirical probability is 1, use special case
            lower_delta = 3 * np.log(1/alpha) / n
            lower_bound = max(0, 1 - lower_delta)
            upper_bound = 1
        else:
            # Upper tail: P(p_hat ≥ (1+delta)*p) ≤ exp(-p*delta^2*n/3)
            # Solve for delta: alpha = exp(-p_hat*delta^2*n/3)
            upper_delta = np.sqrt(3 * np.log(1/alpha) / (p_hat * n))
            
            # Lower tail: P(p_hat ≤ (1-delta)*p) ≤ exp(-p*delta^2*n/2)
            # Solve for delta: alpha = exp(-p_hat*delta^2*n/2)
            lower_delta = np.sqrt(2 * np.log(1/alpha) / (p_hat * n))
            
            # Ensure deltas are capped at reasonable values
            upper_delta = min(upper_delta, 1)
            lower_delta = min(lower_delta, 1)
            
            upper_bound = min(1, p_hat * (1 + upper_delta))
            lower_bound = max(0, p_hat * (1 - lower_delta))
    else:
        raise ValueError("Method must be 'hoeffding', 'bernstein', or 'chernoff'")
    
    return lower_bound, upper_bound