import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
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

        # Velocity components
        self.v = list()

        # Divergence 
        div_sym = None
        for i, div_anti_deriv in enumerate(self.div_anti_derivatives):
            mixed_partial = div_anti_deriv
            for j, xj_sym in enumerate(self.x_symbols):
                if j != i: # Skip the divergence derivative to get just the velocity
                    mixed_partial = sp.diff(mixed_partial, xj_sym)

            print(f"v{i}(x, t) = ", mixed_partial)
            
            # Convert to numerical function
            vi = sp.lambdify(symbols, mixed_partial, modules='numpy')
            self.v.append(vi)

            # Take the derivative previously excluded to get the divergence term
            div_term = sp.diff(mixed_partial, self.x_symbols[i])
            if div_sym:
                div_sym += div_term
            else:
                div_sym = div_term
        
        self.div = sp.lambdify(symbols, div_sym, modules='numpy')

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
        
    def divergence(self, x : np.ndarray, t : float = None):
        return self.div(*x)
    
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
        X0, X1 = np.meshgrid(np.linspace(bounds[0], bounds[1], disc), np.linspace(bounds[2], bounds[3], disc))
        V0 = self.v[0](X0, X1)
        V1 = self.v[1](X0, X1)

        ax.quiver(X0, X1, V0, V1)
        ax.set_xlabel("x_0")
        ax.set_xlabel("x_1")
        ax.set_title("Velocity Field")
        ax.axis('equal')
        ax.grid(True)