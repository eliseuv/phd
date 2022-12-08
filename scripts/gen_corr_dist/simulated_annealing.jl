@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, LinearAlgebra, Statistics, StatsBase, Distributions, Distances, DataFrames, CSV

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.TimeSeries

# Cost calculation
@inline function cost(M_ts, ρ_dist::UnivariateDistribution)
    hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=128), mode=:pdf)
    x_target = map(x -> pdf(ρ_dist, x), hist.edges[begin][begin:end-1])
    x_emp = hist.weights
    chisq_dist(x_emp, x_target)
end

@inline function cost_to_uniform(M_ts)
    n_bins = 128
    hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=n_bins), mode=:pdf)
    x_target = fill(0.5, n_bins)
    x_emp = hist.weights
    chisq_dist(x_emp, x_target)
end

@inline function stats_and_cost_to_uniform(M_ts, n_bins)
    corr_vals = cross_correlation_values(M_ts)
    μ = mean(corr_vals)
    σ² = varm(corr_vals, μ)
    hist = normalize(fit(Histogram, corr_vals, range(minimum(corr_vals), stop=maximum(corr_vals), length=n_bins + 1)), mode=:pdf)
    x_target = fill(0.5, n_bins)
    x_emp = hist.weights
    cost = chisq_dist(x_emp, x_target)
    return (μ, σ², cost)
end

@inline function perturbate!(M_ts, k, σ)
    @views M_ts[:, k] = M_ts[:, k] + rand(Normal(0.0, σ), size(M_ts, 1))
end

# Metropolis sampling applied
function metropolis!(M_ts, β, n_iter)
    n_series = size(M_ts, 2)
    n_bins = 128
    M_ts′ = deepcopy(M_ts)
    # Vector to store measurements at each iteration
    costs = Vector{Float64}(undef, n_iter + 1)
    means = Vector{Float64}(undef, n_iter + 1)
    variances = Vector{Float64}(undef, n_iter + 1)
    # Calculate initial cost
    μ, σ², cost = stats_and_cost_to_uniform(M_ts, n_bins)
    costs[1] = cost
    means[1] = μ
    variances[1] = σ²
    for it ∈ 2:n_iter+1
        for k ∈ rand(1:n_series, n_series)
            perturbate!(M_ts′, k, 0.1)
            μ, σ², cost = stats_and_cost_to_uniform(M_ts′, n_bins)
            Δcost = cost - costs[it-1]
            # Metropolis prescription
            @views if Δcost <= 0 || exp(-β * Δcost) > rand()
                # Accept
                M_ts[:, k] .= M_ts′[:, k]
                means[it] = μ
                variances[it] = σ²
                costs[it] = cost
            else
                # Reject
                M_ts′[:, k] .= M_ts[:, k]
                means[it] = means[it-1]
                variances[it] = variances[it-1]
                costs[it] = costs[it-1]
            end
        end
    end
    return (means, variances, costs)
end

function simulated_annealing!(M_ts, β₀, α, n_steps, n_iter)
    betas = Vector{Float64}()
    means = Vector{Float64}()
    variances = Vector{Float64}()
    costs = Vector{Float64}()
    β = β₀
    for sp ∈ 1:n_steps
        (means_st, variances_st, costs_st) = metropolis!(M_ts, β, n_iter)
        append!(betas, fill(β, n_iter + 1))
        append!(means, means_st)
        append!(variances, variances_st)
        append!(costs, costs_st)
        β *= α
    end
    return (betas, means, variances, costs)
end

# Parameters
# Time series matrix
const n_series = 128
const t_max = 256
# Target correlation distribution
# const ρ_dist = Uniform(-1.0, 1.0)
# Simulated annealing parameters
const β₀ = 0.5
const α = 1.1
const β_F = 2.5
const n_steps = ceil(Int64, log(β_F / β₀) / log(α))
const n_iter = 1024

# Generate time series matrix
M_ts = rand(Normal(), t_max, n_series)

betas, means, variances, costs = simulated_annealing!(M_ts, β₀, α, n_steps, n_iter)

CSV.write(datadir("test.csv"),
    DataFrame(betas=betas, means=means, variances=variances, costs=costs))
