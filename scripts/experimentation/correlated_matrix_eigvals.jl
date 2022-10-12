@doc raw"""
    Calculation of the eigenvalues of time
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, Distributions, LinearAlgebra

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.Metaprogramming

# Generated correlated pair of random variable with given correlation
function correlated_pair(ρ::Real, dist::Distribution=Normal())
    # Uncorrelated pair
    φ = rand(dist, 2)
    # Create correlated pair
    θ = 0.5 * asin(ρ)
    ϕ = [φ[1] * sin(θ) + φ[2] * cos(θ),
        φ[1] * cos(θ) + φ[2] * sin(θ)]
    return ϕ
end

# Create 2x2 correlated matrix
function correlated_matrix(ρ::Real, dist::Distribution=Normal())
    ϕ = correlated_pair(ρ, dist)
    ψ = correlated_pair(ρ, dist)
    return vcat(ϕ', ψ')
end

# Normaliza time series matrix
@inline normalize_ts_matrix(M_ts::AbstractMatrix) = hcat(
    map(eachcol(M_ts)) do xⱼ
        xⱼ_avg = mean(xⱼ)
        (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg, corrected=false)
    end...)

# Create cross correlation matrix
@inline function cross_correlation_matrix(M_ts::AbstractMatrix)
    # Number of steps = number of rows
    n_steps = size(M_ts, 1)
    return (1 / n_steps) .* Symmetric(transpose(M_ts) * M_ts)
end

# eigvals(ρ, N) = map(eigvals, [cross_correlation_matrix(normalize_ts_matrix(correlated_matrix(ρ))) for _ ∈ 1:N])
