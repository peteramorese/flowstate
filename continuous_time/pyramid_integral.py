import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

def volume_hyperpyramid(base : Rectangle, height : float):
    return base.volume() * height / (base.m + 1)

def scale_nd_rectangle(region : Rectangle, scales : np.ndarray):
    """
    Scales a region by given scale factors.
    """

    center = (region.mins + region.maxes) / 2
    half_sizes = (region.maxes - region.mins) / 2

    new_half_sizes = half_sizes * scales

    scaled_region = Rectangle(mins=center - new_half_sizes, maxes=center + new_half_sizes)

    volume_change = region.volume() / scaled_region.volume()

    return scaled_region, volume_change

def adversarial_integral(region : Rectangle, facet_slope : float, apex_value : float, target_region_volume : float):
    """
    Calculate the adversarial integral of a chopped hyperpyramid (prism) with equal facet slope in each dimension.
    """
    #print("target reigion vol: ", target_region_volume, " region vol: ", region.volume())
    total_volume = region.volume()
    assert target_region_volume <= total_volume 

    # Center the region about the origin
    center_point = (region.mins + region.maxes) / 2
    origin_centered_region = copy.deepcopy(region)
    origin_centered_region.mins -= center_point
    origin_centered_region.maxes -= center_point

    # Re-orient the rectangle such that the lengths decrease as dimensions increase
    sorted_dims = sorted(range(region.m), key=lambda i: origin_centered_region.mins[i])
    sorted_centered_region = Rectangle(mins=origin_centered_region.mins[sorted_dims], maxes=origin_centered_region.maxes[sorted_dims])

    def create_mass_polynomial(base_dim, apex_value):
        # Construct the volume polynomial
        
        # Coefficient counts the number of edge pieces are added equal to number of facets on n-rectangle
        p = Polynomial([2 * base_dim])

        # Multiply the integral regions (dA)
        p *= Polynomial([0, -2])**(base_dim - 1)
        
        # Multiply by the linear function integrand
        p *= Polynomial([apex_value, facet_slope])
    
        # Integrate the polynomial to get the volume polynomial
        return p.integ()
    
    sorted_region_lengths = sorted_centered_region.maxes - sorted_centered_region.mins
    np.append(sorted_region_lengths, 1) # Append 1 for cleaner code
    previous_volume = 0 # Track the volume of the previous iterations
    previous_mass = 0 # Track the cumulative integral value
    for i in range(sorted_centered_region.m):

        volume_polynomial = create_mass_polynomial(i + 1, apex_value)
        
        # Check if the target_region volume is beyond this iteration by just checking the endpoint
        if i < (sorted_centered_region.m - 1):
            iteration_volume_upper_bound = previous_volume + (sorted_region_lengths[i]**(i+1) * np.prod(sorted_region_lengths[i+1:]) - sorted_region_lengths[i+1]**(i+2) * np.prod(sorted_region_lengths[i+2:]))
        else:
            iteration_volume_upper_bound = total_volume
        
        # If the target region volume exceeds the iteration upper bound, simply move to the next iteration 
        # otherwise, then we need to solve for indeterminate dimension defining the interior region
        if target_region_volume > iteration_volume_upper_bound:
            previous_volume = iteration_volume_upper_bound

            # Add the integral over the volume for this iteration
            mass_coefficient = np.prod(sorted_region_lengths[i+1:])
            previous_mass += mass_coefficient * (volume_polynomial(-sorted_region_lengths[i + 1] / 2) - volume_polynomial(-sorted_region_lengths[i] / 2))
            continue
        else:
            # Solve for the indeterminate dimension
            coefficient = np.prod(sorted_region_lengths[i+1:])
            l = 0.5*(sorted_region_lengths[i] - ((total_volume - target_region_volume) / coefficient)**(1/(i+1)))
            
            # Calculate the integral without including the interior region (and future iterations)
            mass_coefficient = np.prod(sorted_region_lengths[i+1:])
            return previous_mass + mass_coefficient * (volume_polynomial(l - sorted_region_lengths[i] / 2) - volume_polynomial(-sorted_region_lengths[i] / 2))


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
    ax.set_xlim(region.mins[0], region.maxes[0])
    ax.set_ylim(region.mins[1], region.maxes[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Plot of f(x, y)')
    ax.set_aspect('equal')
    plt.tight_layout()

        
    




            


if __name__ == "__main__":
    #region = Rectangle(mins=[0, 0], maxes=[2, 1])
    #region = Rectangle(mins=[-0.6, -1, -0.25], maxes=[0.6, 1, 0.25])
    region = Rectangle(mins=[0.3, -1, -0.25, 5], maxes=[0.9, 2, 0.28, 7])
    print("original region: ", region)

    L = 1
    c = region.mins + (region.maxes - region.mins) / 2
    f_of_c = 0.5

    ## iteration 3
    #test_l = 0.1
    #target_vol = 1.2 - (0.5 - 2 * test_l)**3
    #inner_region = Rectangle(mins=[-0.25 + test_l, -0.25 + test_l, -0.25 + test_l], maxes = [0.25 - test_l, 0.25 - test_l, 0.25 - test_l])

    ## iteration 2
    #test_l = .6 - .25
    #target_vol = 1.2 - 0.5*(1.2 - 2 * test_l)**2
    #inner_region = Rectangle(mins=[-0.6 + test_l, -0.6 + test_l, -0.25], maxes = [0.6 - test_l, 0.6 - test_l, 0.25])

    target_vol = region.volume()
    #print("target_vol:", target_vol, "inner region vol: ", inner_region.volume())
    mass = adversarial_integral(region, L, f_of_c, target_vol)
    print("target vol:", target_vol, " adversarial mass: ", mass)

    def f(x):
        return f_of_c - np.linalg.norm(x - c, ord=np.inf)

    #print("inner region: ", inner_region)
    #true_val = integral_comparison(f, region, n_samples=300000) - integral_comparison(f, inner_region, n_samples=300000)
    true_val = integral_comparison(f, region, n_samples=300000) 
    print("Total region integral: ", true_val)

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

    plt.show()


    
