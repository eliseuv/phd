@doc raw"""
    Calculation of the eigenvalues of time
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Random, LinearAlgebra, CairoMakie

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.CorrelatedPairs
using .Thesis.RandomMatrices

# Fake command line arguments
push!(ARGS, "1")

const ρ = parse(Float64, ARGS[1])
const t_max = 512
const n_pairs = 256
const n_samples = 512

# Sampler
spl = CorrelatedTimeSeriesMatrixSampler(ρ, t_max, n_pairs)

# Calculate eigenvalues
λs = reduce(vcat,
    map(eigvals ∘ cross_correlation_matrix ∘ normalize_ts_matrix!,
        rand(spl, n_samples))) |> sort!

hist(λs, bins=64, normalization=:pdf)
current_figure()
