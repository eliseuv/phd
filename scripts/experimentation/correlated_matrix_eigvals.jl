@doc raw"""
    Calculation of the eigenvalues of time
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Base: permutecols!!, copymutable
using Logging, Random, Statistics, Distributions, LinearAlgebra, DataFrames, Gadfly, Cairo

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.Matrices

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

# const n_pairs = 3
# const M = 10000
# const n_samples = 1000

# df = DataFrame(ρ=Float64[], λ1_mean=Float64[], λ2_mean=Float64[], λ3_mean=Float64[], λ4_mean=Float64[], λ5_mean=Float64[], λ6_mean=Float64[])
# for ρ ∈ range(-1.0, 1.0, length=21)
#     @show ρ
#     (λ_mean, λ_std) = eigvals_stats(ρ, M, n_pairs, n_samples)
#     push!(df, Dict(:ρ => ρ,
#         :λ1_mean => λ_mean[1], :λ2_mean => λ_mean[2], :λ3_mean => λ_mean[3], :λ4_mean => λ_mean[4], :λ5_mean => λ_mean[5], :λ6_mean => λ_mean[6]))
# end

# plt_lambda1mean = layer(df, x=:ρ, y=:λ1_mean,
#     Geom.line())
# plt_lambda2mean = layer(df, x=:ρ, y=:λ2_mean,
#     Geom.line())
# plt_lambda3mean = layer(df, x=:ρ, y=:λ3_mean,
#     Geom.line())
# plt_lambda4mean = layer(df, x=:ρ, y=:λ4_mean,
#     Geom.line())
# plt_lambda5mean = layer(df, x=:ρ, y=:λ5_mean,
#     Geom.line())
# plt_lambda6mean = layer(df, x=:ρ, y=:λ6_mean,
#     Geom.line())

# plt = plot(plt_lambda1mean, plt_lambda2mean, plt_lambda3mean, plt_lambda4mean, plt_lambda5mean, plt_lambda6mean,
#     Guide.title("Mean of eigenvalues of cross correlation matrix for two time series created from pairs of correlation ρ"),
#     Guide.xlabel("ρ"), Guide.ylabel("⟨λ⟩"))
# draw(PNG(plotsdir("test.png"), 25cm, 15cm), plt)
