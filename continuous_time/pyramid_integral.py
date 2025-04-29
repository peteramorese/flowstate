import numpy as np
from scipy.spatial import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

def integral_comparison(f, region : Rectangle, n_samples : int = 10000):
    # Dimension
    n = region.m

    # Generate uniform random samples in the domain [0, 1]^n then scale to [mins, maxs]
    unit_samples = np.random.rand(n_samples, n)
    samples = region.mins + (region.maxes - region.mins) * unit_samples

    # Evaluate the function on all samples
    values = np.apply_along_axis(f, 1, samples)

    # Compute the volume of the integration region
    volume = np.prod(region.maxes - region.mins)

    # Monte Carlo estimate
    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_samples)

    #print(f"Estimated integral over {n}D region: {integral:.6f}")
    #print(f"Standard error: {std_error:.2e}")
    return integral

def plot_function_2d(f, region : Rectangle, resolution=100):
    """
    Plots f(x) where x ∈ ℝ² → ℝ over a rectangular grid.
    
    Parameters:
    - f: function taking a 2D vector [x, y]
    - x_min, x_max: bounds for x-axis
    - y_min, y_max: bounds for y-axis
    - resolution: number of grid points per axis
    """
    x = np.linspace(region.mins[0], region.maxes[0], resolution)
    y = np.linspace(region.mins[1], region.maxes[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Plot of f(x, y)')
    plt.tight_layout()

def volume_hyperpyramid(base : Rectangle, height : float):
    return base.volume() * height / (base.m + 1)

def adversarial_volume(region : Rectangle, facet_slope : float, apex_point : float, target_region_volume : float):
    """
    Calculate the adversarial volume of a chopped hyperpyramid (prism) with equal facet slope in each dimension.
    """
    center_point = (region.mins + region.maxes) / 2
    origin_centered_region = copy.deepcopy(region)
    origin_centered_region.mins -= center_point
    origin_centered_region.maxes -= center_point

    if apex_point - facet_slope * max(origin_centered_region.maxes) < 0 and apex_point > 0:
        # The volume changes sign somewhere
        l_positive = apex_point / facet_slope
    else:
        l_positive = np.inf

    sorted_dims = sorted(range(region.m), key=lambda i: origin_centered_region.mins[i])
    sorted_centered_region = Rectangle(mins=origin_centered_region.mins[sorted_dims], maxes=origin_centered_region.maxes[sorted_dims])
    print("sorted_centered_region: ", sorted_centered_region)

    def create_inner_region(d):
        inner_region = copy.deepcopy(sorted_centered_region)
        for i in range(d):
            inner_region.mins[i] = sorted_centered_region.mins[d]
            inner_region.maxes[i] = sorted_centered_region.maxes[d]

    # Find which iteration the target region volume occurs in
    total_region_volume = region.volume()
    target_region_iteration = sorted_centered_region.m - 1
    for d in range(1, sorted_centered_region.m):
        inner_region = create_inner_region(d)
        #print("  d: ", d, " inner_region: ", inner_region)
        #print("    outter volume: ", total_region_volume - inner_region.volume())
        if total_region_volume - inner_region.volume() > target_region_volume:
            target_region_iteration = d - 1
            break
    #print("target_region_iteration: ", target_region_iteration)

    # Calculate the block volume of previous iterations
    block_volume = 0
    for d in range(target_region_iteration):
        inner_region = create_inner_region(d) 
        pyramid_region = Rectangle(mins=inner_region.mins[:d+1], maxes=inner_region.maxes[:d+1])
        rectangular_region = Rectangle(mins=inner_region.mins[d+1:], maxes=inner_region.maxes[d+1:])

        pyramid_height = 0.5 * facet_slope * (sorted_centered_region.maxes[0] - sorted_centered_region.mins[0])
        inner_pyramid_height = 0.5 * (inner_region.maxes[0] - inner_region.mins[0]) * L

        pyramid_volume = volume_hyperpyramid(pyramid_region, pyramid_height) - volume_hyperpyramid
        block_volume   


    def edge_volume(l, d):
        inner_region = create_inner_region(d) 
        # Split the prism into pyramid dimensions and rectangular dimensions
        pyramid_region = Rectangle(mins=inner_region.mins[d+1:], maxes=inner_region.maxes[d+1:])
        rectangular_region = Rectangle(mins=inner_region.mins[d+1:], maxes=inner_region.maxes[d+1:])
        return 
            


if __name__ == "__main__":
    dim = 2
    region = Rectangle(mins=[0, 0, 0], maxes=[1.2, 2, 0.5])
    print("original region: ", region)
    L = 1
    c = np.zeros((1,dim))
    f_of_c = 1

    adversarial_volume(region, L, f_of_c, 1.19)

    #def f(x):
    #    return f_of_c - np.linalg.norm(x - c, ord=1)

    #plot_function_2d(f, region, 100)
    
    ## Integral (numerical)
    #def numerical_area(l):
    #    inner_region = copy.deepcopy(region)
    #    inner_region.mins += l 
    #    inner_region.maxes -= l 
    #    num_samples = 10000
    #    return integral_comparison(f, region, n_samples=num_samples) - integral_comparison(f, inner_region, n_samples=num_samples)
    
    #def analytical_area(l):
    #    inner_region = copy.deepcopy(region)
    #    inner_region.mins += l 
    #    inner_region.maxes -= l 

    #    pyramid_height = 0.5 * (region.maxes[0] - region.mins[0]) * L
    #    #inner_pyramid_height = (0.5 * (region.maxes[0] - region.mins[0]) - l) * L
    #    inner_pyramid_height = 0.5 * (inner_region.maxes[0] - inner_region.mins[0]) * L
    #    #print("region: ", region, " vol: ", region.volume(), " height: ", pyramid_height, "")
    #    #print("term1 ", volume_hyperpyramid(region, pyramid_height), " term2: ", volume_hyperpyramid(inner_region, inner_pyramid_height), " term3 ", inner_region.volume() * (pyramid_height - inner_pyramid_height))
    #    offset = (f_of_c - pyramid_height) * region.volume()
    #    #print("vtotal: ", volume_hyperpyramid(region, pyramid_height), " vin: ", volume_hyperpyramid(inner_region, inner_pyramid_height), " vinb: ", inner_region.volume() * (pyramid_height - inner_pyramid_height))
    #    return volume_hyperpyramid(region, pyramid_height) - volume_hyperpyramid(inner_region, inner_pyramid_height) - inner_region.volume() * (pyramid_height - inner_pyramid_height) #+ offset

    ##analytical_area(0.5)

    #l_ls = np.linspace(0, 0.5 * (region.maxes[0] - region.mins[0]), 100)

    #area_num = [numerical_area(l) for l in l_ls]
    #area_ana = [analytical_area(l) for l in l_ls]

    #print("act vol: ", volume_hyperpyramid(region, 0.5 * (region.maxes[0] - region.mins[0]) * L))


    #plt.figure()
    #plt.plot(l_ls, area_num)
    #plt.plot(l_ls, area_ana)

    #plt.show()


    
