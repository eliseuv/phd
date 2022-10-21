using Base: permutecols!!, copymutable
using Gadfly: GuideElement
@doc raw"""
    Calculation of the eigenvalues of time
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Random, Statistics, Distributions, LinearAlgebra, DataFrames, Gadfly, Cairo

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.Metaprogramming

# Generated correlated pair of random variable with given correlation
@inline function correlated_pair(ρ::Real, dist::Distribution=Normal())
    # Uncorrelated pair
    φ = rand(dist, 2)
    # Create correlated pair
    θ = 0.5 * asin(ρ)
    ϕ = (φ[1] * sin(θ) + φ[2] * cos(θ),
        φ[1] * cos(θ) + φ[2] * sin(θ))
    return ϕ
end

# Create matrix with multiple pairs of correlated time series
@inline correlated_ts_matrix(ρ::Real, M::Integer=2, n_pairs::Integer=1, dist::Distribution=Normal()) =
    hcat([vcat(map(ϕ -> [ϕ[1] ϕ[2]], correlated_pair(ρ, dist) for _ in 1:M)...) for _ in 1:n_pairs]...)

# Normaliza time series matrix
@inline normalize_ts_matrix(M_ts::AbstractMatrix) = hcat(
    map(eachcol(M_ts)) do xⱼ
        xⱼ_avg = mean(xⱼ)
        (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg, corrected=false)
    end...)
@inline function normalize_ts_matrix!(M_ts::AbstractMatrix)
    for xⱼ ∈ eachcol(M_ts)
        xⱼ_avg = mean(xⱼ)
        xⱼ .= (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg)
    end
end

@inline function shuffle_cols!(M_ts::AbstractMatrix)
    permutecols!!(M_ts, randperm(size(M_ts, 2)))
end

# Create cross correlation matrix
@inline function cross_correlation_matrix(M_ts::AbstractMatrix)
    # Number of steps = number of rows
    n_steps = size(M_ts, 1)
    return (1 / n_steps) .* Symmetric(transpose(M_ts) * M_ts)
end

function eigvals_stats(ρ, ts_length, n_pairs, n_samples)
    matrices = [correlated_ts_matrix(ρ, ts_length, n_pairs) for _ in 1:n_samples]
    foreach(matrices) do M
        normalize_ts_matrix!(M)
        shuffle_cols!(M)
    end

    λs = map(matrices) do M
        eigvals(cross_correlation_matrix(M))
    end

    λ_mean = mean(λs)
    λ_var = varm(λs, λ_mean)
    return (λ_mean, sqrt.(λ_var))
end

const n_pairs = 3
const M = 10000
const n_samples = 1000

df = DataFrame(ρ=Float64[], λ1_mean=Float64[], λ2_mean=Float64[], λ3_mean=Float64[], λ4_mean=Float64[], λ5_mean=Float64[], λ6_mean=Float64[])
for ρ ∈ range(-1.0, 1.0, length=21)
    @show ρ
    (λ_mean, λ_std) = eigvals_stats(ρ, M, n_pairs, n_samples)
    push!(df, Dict(:ρ => ρ,
        :λ1_mean => λ_mean[1], :λ2_mean => λ_mean[2], :λ3_mean => λ_mean[3], :λ4_mean => λ_mean[4], :λ5_mean => λ_mean[5], :λ6_mean => λ_mean[6]))
end

plt_lambda1mean = layer(df, x=:ρ, y=:λ1_mean,
    Geom.line())
plt_lambda2mean = layer(df, x=:ρ, y=:λ2_mean,
    Geom.line())
plt_lambda3mean = layer(df, x=:ρ, y=:λ3_mean,
    Geom.line())
plt_lambda4mean = layer(df, x=:ρ, y=:λ4_mean,
    Geom.line())
plt_lambda5mean = layer(df, x=:ρ, y=:λ5_mean,
    Geom.line())
plt_lambda6mean = layer(df, x=:ρ, y=:λ6_mean,
    Geom.line())

plt = plot(plt_lambda1mean, plt_lambda2mean, plt_lambda3mean, plt_lambda4mean, plt_lambda5mean, plt_lambda6mean,
    Guide.title("Mean of eigenvalues of cross correlation matrix for two time series created from pairs of correlation ρ"),
    Guide.xlabel("ρ"), Guide.ylabel("⟨λ⟩"))
draw(PNG(plotsdir("test.png"), 25cm, 15cm), plt)
