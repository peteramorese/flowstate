using DynamicPolynomials
using MultivariatePolynomials
using LazySets
using IterTools

struct VolumePolynomial
    x_vars::Vector{Variable}
    t_var::Variable
    p::AbstractPolynomialLike
end

struct IntegratorPolynomial
    x_vars::Vector{Variable}
    t_var::Variable
    p_antideriv::AbstractPolynomialLike
end

function divergence(x::Vector{Variable}, Φ::AbstractPolynomialLike)
    divergence_poly = 0
    for i in 1:length(x)
        divergence_poly += differentiate(Φ, x[i])
    end
    return divergence_poly
end

function reynolds_operator(x::Vector{Variable}, Φ::AbstractPolynomialLike, v::Vector{AbstractPolynomialLike})
    Φ_scaled_field = Φ .* v
    return divergence(x, Φ_scaled_field)
end


function compute_coefficients(x::Vector{Variable}, model::Vector{AbstractPolynomialLike}, degree::Int=1)
    Φ_i = 1
    coefficients = []
    for i in 1:degree
        ϕ_ip1 = reynolds_operator(x, ϕ_i, model)
        ϕ_i = ϕ_ip1
        push!(coefficients, Φ_i)
    end
    return coefficients
end

function create_volume_polynomial(
        x::Vector{Variable}, 
        t::Variable,
        model::Vector{AbstractPolynomialLike}, 
        degree::Int=1)

    coefficients = compute_coefficients(x, model, degree)

    t_monoms = monomials(t, 1:degree)

    return t_monoms' * coefficients
end

function create_integrator_polynomial(volume_polynomial::VolumePolynomial)
    x_vars = volume_polynomial.x_vars
    t_var = volume_polynomial.t_var

    # Create the antiderivative polynomial
    p_antideriv = volume_polynomial.p
    for x_var in x_vars
        p_antideriv = antidifferentiate(p_antideriv, x_var)
    end

    return IntegratorPolynomial(x_vars, t_var, p_antideriv)
end

function evaluate_integral(antideriv, region::Hyperrectangle{Float64})
    center = region.center
    radius = region.radius

    n = length(center)
    integral = 0.0
    for bits in Iterators.product((0:1 for _ in 1:n)...)
        vertex = [center[i] + (2*bits[i]-1)*radius[i] for i in 1:n]
        sign = (-1)^sum(bits)
        integral += sign * antideriv(vertex)
    end

    return integral
end

function density(x_eval::Vector{Float64}, t_eval::Float64, volume_polynomial::VolumePolynomial)
    subst = Dict(volume_polynomial.x_vars[i] => x_eval[i] for i in 1:length(volume_polynomial.x_vars))
    merge!(subst, Dict(volume_polynomial.t_var => t_eval))
    return subs(volume_polynomial.p, subst...)
end

function probability(region::Hyperrectangle{Float64}, integ_polynomial::IntegratorPolynomial)
end



