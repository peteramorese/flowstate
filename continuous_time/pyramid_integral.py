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
    return base.volume() * height / base.m

if __name__ == "__main__":
    dim = 2
    region = Rectangle(mins=[-1, -1], maxes=[1, 1])
    print("original region: ", region)
    L = 1
    c = np.zeros((1,dim))
    f_of_c = 1

    def f(x):
        return f_of_c - np.linalg.norm(x - c, ord=1)

    plot_function_2d(f, region, 100)
    

    # Integral (numerical)
    def numerical_area(l):
        inner_region = copy.deepcopy(region)
        inner_region.mins += l 
        inner_region.maxes -= l 
        num_samples = 10000
        return integral_comparison(f, region, n_samples=num_samples) - integral_comparison(f, inner_region, n_samples=num_samples)
    
    def analytical_area(l):
        inner_region = copy.deepcopy(region)
        inner_region.mins += l 
        inner_region.maxes -= l 

        pyramid_height = 0.5 * (region.maxes[0] - region.mins[0]) * L
        inner_pyramid_height = (0.5 * (region.maxes[0] - region.mins[0]) - l) * L
        #print("region: ", region, " vol: ", region.volume(), " height: ", pyramid_height, "")
        #print("term1 ", volume_hyperpyramid(region, pyramid_height), " term2: ", volume_hyperpyramid(inner_region, inner_pyramid_height), " term3 ", inner_region.volume() * (pyramid_height - inner_pyramid_height))
        offset = (f_of_c - pyramid_height) * region.volume()
        return volume_hyperpyramid(region, pyramid_height) - volume_hyperpyramid(inner_region, inner_pyramid_height) - inner_region.volume() * (pyramid_height - inner_pyramid_height) + offset

    l_ls = np.linspace(0, 0.5 * (region.maxes[0] - region.mins[0]), 100)

    area_num = [numerical_area(l) for l in l_ls]
    area_ana = [analytical_area(l) for l in l_ls]

    print("act vol: ", volume_hyperpyramid(region, 0.5 * (region.maxes[0] - region.mins[0]) * L))


    plt.figure()
    plt.plot(l_ls, area_num)
    plt.plot(l_ls, area_ana)

    plt.show()


    
