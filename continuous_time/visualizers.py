import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Rectangle

from velocity_field import VelocityField

class RegionBoundaryDiscretization:
    def __init__(self, region : Rectangle, n=20):
        bottom = np.linspace(region.mins, [region.maxes[0], region.mins[1]], n, endpoint=False)
        right = np.linspace([region.maxes[0], region.mins[1]], region.maxes, n, endpoint=False)
        top = np.linspace(region.maxes, [region.mins[0], region.maxes[1]], n, endpoint=False)
        left = np.linspace([region.mins[0], region.maxes[1]], region.mins, n, endpoint=False)
        self.boundary_points = np.vstack([bottom, right, top, left])
    
    def flow_forward(self, vf : VelocityField, dt : float, t : float = None):
        self.boundary_points += np.array([dt * vf.velocity(bi, t) for bi in self.boundary_points])

    def flow_backward(self, vf : VelocityField, dt : float, t : float = None):
        self.boundary_points -= np.array([dt * vf.velocity(bi, t) for bi in self.boundary_points])


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