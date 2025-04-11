import numpy as np
from scipy import special as sp
from scipy import spatial
from typing import Callable

class Problem:
    def __init__(self, n : int, m : int, g : Callable, g_inv : Callable, Qx0 : Callable, Qw : Callable, Phi : Callable, Phi_inv : Callable, J_Ginv : Callable, J_Phi):
        self.n = n
        self.m = m
        self.g = g
        self.g_inv = g_inv
        self.Qx0 = Qx0
        self.Qw = Qw
        self.Phi = Phi
        self.Phi_inv = Phi_inv
        self.J_Ginv = J_Ginv
        self.J_Phi = J_Phi
    
    ## Flow Functions ##

    def Gok(self, w, x0, k):
        xi = x0
        for i in range(0, k):
            xi = self.g(w, xi)
        return (w, xi)

    def Gok_inv(self, w, xk, k):
        xi = xk
        for i in range(k, 0, -1):
            xi = self.g_inv(w, xi)
        return (w, xi)

    # Sample states at time k 
    def sample_empirical(self, k : int, n_samples : int, w_test = None):
        xk_samples = np.zeros((self.n, n_samples))
        for s in range(0, n_samples):
            yw_rand = np.random.uniform(0, 1, self.m)
            yx_rand = np.random.uniform(0, 1, self.n)
            if w_test:
                w, x0 = self.Phi_inv(yw_rand, yx_rand)
                w, xk = self.Gok(w_test, x0, k)
            else:
                w, xk = self.Gok(*self.Phi_inv(yw_rand, yx_rand), k)
                xk_samples[:, s] = xk

            xk_samples[:, s] = xk
        return xk_samples
    
    # Density at time k 
    def p(self, w, xk, k):
        try:
            # Invert to x0
            J = np.identity(self.n + self.m)
            for i in range(k, 0, -1):
                J = np.matmul(J, self.J_Ginv(*self.Gok_inv(w, xk, i - 1)))

            # Invert to y
            J = np.matmul(J, self.J_Phi(*self.Gok_inv(w, xk, k)))

            density = np.abs(np.linalg.det(J))
            return density if not np.isnan(density) else 0
        except np.linalg.LinAlgError:
            return 0


