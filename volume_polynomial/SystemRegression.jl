using Random
using Distributions
using LinearAlgebra
using MultivariatePolynomials
using DynamicPolynomials
using Plots
pyplot()
#using StatsBase
#using MLJLinearModels
#using MLJModelInterface
#using IterTools

# Generate synthetic data
function generate_data(f, d, n; domain_std=1.0, noise_std=0.1, seed=0)
    Random.seed!(seed)
    X = randn(n, d) .* domain_std
    Y_clean = zeros(n, d)
    for i in 1:n
        Y_clean[i, :] = f(X[i, :])
    end
    noise = randn(n, d) .* noise_std
    Y = Y_clean + noise
    return X, Y
end

function visualize_data(X, Y, title="")
    @assert size(X, 2) == 2
    p1 = scatter(X[:, 1], X[:, 2], Y[:, 1], markersize=4, title="f1(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f1")
    p2 = scatter(X[:, 1], X[:, 2], Y[:, 2], markersize=4, title="f2(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f2")
    display(plot!(p1, p2, layout=(1, 2), size=(800, 400), title=title))
end

function plot_polynomial_surface(p, x, y, xlim, ylim; n_points=50, kwargs...)
    # Create grid points
    xvals = range(xlim[1], xlim[2], length=n_points)
    yvals = range(ylim[1], ylim[2], length=n_points)
    
    # Store z values in matrix
    z_values = [p(x => x_val, y => y_val) for y_val in yvals, x_val in xvals]
    
    # Create the surface plot
    return Plots.surface(xvals, yvals, z_values; 
                   xlabel="$(x)", ylabel="$(y)", zlabel="p($(x),$(y))",
                   title="Polynomial Surface: $p",
                   kwargs...)
end

# Gaussian CDF transform (erf-space)
function erf_space_transform(x)
    return cdf.(Normal(), x)
end

# Jacobian of the erf-space transform
function erf_space_transform_jacobian(x)
    return Diagonal(pdf.(Normal(), x))
end

function poly_regression(polyvars, X::Matrix{Float64}, y::Vector{Float64}; deg::Int=2)
    n, d = size(X)

    # Create symbolic variables x₁, x₂, ..., x_d

    # Get all monomials up to total degree `deg`
    mons = monomials(polyvars, 0:deg)

    # Build the design matrix A (n × num_monomials)
    A = zeros(n, length(mons))
    for i in 1:n
        xi = X[i, :]
        subst = Dict(polyvars[j] => xi[j] for j in 1:d)
        A[i, :] = [subs(m, subst...) for m in mons]
    end

    # Solve least squares to find coefficients
    coeffs = A \ y

    # Form the polynomial from monomials and fitted coefficients
    p = sum(coeffs[i] * mons[i] for i in 1:length(mons))

    return p
end

function system_regression(X_u, Y_u, degree)
    @assert size(X_u) == size(Y_u)

    d = size(X_u, 2)
    @polyvar x[1:d]

    model = Vector{Polynomial}()
    
    bc_scaling = 1 ./(X_u .* (1 .- X_u))
    Y_u_unconst = Y_u .* bc_scaling

    for i in 1:n
        y = Y_u_unconst[:, i]
        p = poly_regression(x, X_u, y, deg=degree) 
        p_bc = x[i] * (1 - x[i]) * p 
        push!(model, p_bc)
    end
    return model
end


# Example usage
d = 2
n = 1000
degree = 3

example_function(x) = sin.(4 * x) .* 1 ./ (1 .+ (.5*x).^2)
X, Y = generate_data(example_function, d, n, noise_std=0.01)

#visualize_data(X, Y)

# Transform to unit box
X_u = erf_space_transform.(X)
Y_u = zeros(size(Y))
for i in 1:size(X, 1)
    J = erf_space_transform_jacobian(X[i, :])
    Y_u[i, :] = J * Y[i, :]
end

#visualize_data(X_u, Y_u, "Erf space transformed data")

# Fit constrained model
#models = fit_constrained_polynomial(X_u, Y_u, degree)












#@polyvar x[1:2]
#
#X = randn(n, 2)
#y = .4 * X[:,1].^3 + 2 * X[:,2].^2 .+ 5 .+ 0.1 * randn(n)
#
## Perform polynomial regression
#p = poly_regression(x[1:2], X, y, deg=3)
#
#p_true = .4 * x[1]^3 + 2 * x[2]^2 .+ 5
#
#println("Fitted polynomial:")
#println(p)
#
#p1 = scatter(X[:, 1], X[:, 2], y, markersize=4, title="f1(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f1")
#p2 = plot_polynomial_surface(p, x[1], x[2], (-3, 3), (-3, 3), title="Polynomial Surface")
#p3 = plot_polynomial_surface(p_true, x[1], x[2], (-3, 3), (-3, 3), title="True Polynomial Surface")
#display(plot(p1, p2, p3, layout=(1, 3), size=(800, 400)))
