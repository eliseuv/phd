@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, LinearAlgebra, Statistics, StatsBase, Distributions, Distances, DataFrames, JLD2

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.TimeSeries

# Cost calculation
# @inline function cost(M_ts, ρ_dist::UnivariateDistribution)
#     hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=128), mode=:pdf)
#     x_target = map(x -> pdf(ρ_dist, x), hist.edges[begin][begin:end-1])
#     x_emp = hist.weights
#     chisq_dist(x_emp, x_target)
# end

# @inline function cost_to_uniform(M_ts)
#     n_bins = 128
#     hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=n_bins), mode=:pdf)
#     x_target = fill(0.5, n_bins)
#     x_emp = hist.weights
#     chisq_dist(x_emp, x_target)
# end

@inline function distance_to_uniform(values::AbstractVector{<:Real}, n_bins::Integer, dist::F) where {F}
    hist = normalize(fit(Histogram, values, range(extrema(values)..., length=n_bins + 1)), mode=:pdf)
    x_target = fill(0.5, n_bins)
    x_emp = hist.weights
    return dist(x_emp, x_target)
end

# @inline function perturbate!(M_ts::AbstractMatrix{<:Real}, k::Integer, σ::Real)
#     @views M_ts[:, k] = M_ts[:, k] + rand(Normal(0.0, σ), size(M_ts, 1))
# end

@inline function perturbate!(M_ts′::AbstractMatrix{<:Real}, M_ts::AbstractMatrix{<:Real}, g::Integer, σ::Real)
    t_max, n_series = size(M_ts′)
    for j ∈ 1:n_series
        for i ∈ sample(1:t_max, g, replace=false)
            M_ts′[i, j] = M_ts[i, j] + randn() * σ
        end
    end
end

# Metropolis sampling applied
function metropolis!(M_ts::AbstractMatrix{<:Real}, β::Real, n_iter::Integer;
    γ::Real=0.1, σ::Real=0.3, n_bins::Integer=128, dist::F=chisq_dist) where {F}
    t_max, n_series = size(M_ts)
    g = ceil(Int64, γ * t_max)
    # Vector to store measurements at each iteration
    costs = Vector{Float64}(undef, n_iter + 1)
    means = Vector{Float64}(undef, n_iter + 1)
    variances = Vector{Float64}(undef, n_iter + 1)
    # Calculate initial distribution
    corr_vals = cross_correlation_values(M_ts)
    # Calculate initial values
    costs[1] = distance_to_uniform(corr_vals, n_bins, dist)
    means[1] = mean(corr_vals)
    variances[1] = varm(corr_vals, means[1])
    # Iterations loop
    M_ts_prev = deepcopy(M_ts)
    for it ∈ 2:n_iter+1
        perturbate!(M_ts, M_ts_prev, g, σ)
        corr_vals = cross_correlation_values(M_ts)
        cost = distance_to_uniform(corr_vals, n_bins, dist)
        Δcost = cost - costs[it-1]
        # Metropolis prescription
        @views if Δcost <= 0 || exp(-β * Δcost) > rand()
            # Accept
            copy!(M_ts_prev, M_ts)
            means[it] = mean(corr_vals)
            variances[it] = varm(corr_vals, means[it])
            costs[it] = cost
        else
            # Reject
            copy!(M_ts, M_ts_prev)
            means[it] = means[it-1]
            variances[it] = variances[it-1]
            costs[it] = costs[it-1]
        end
    end
    return (means, variances, costs)
end

function simulated_annealing!(M_ts::AbstractMatrix{<:Real}, β₀::Real, α::Real, n_steps::Integer, n_iter::Integer;
    γ::Real=0.1, σ::Real=0.3, n_bins::Integer=128, dist::F=chisq_dist) where {F}
    betas = Vector{Float64}()
    means = Vector{Float64}()
    variances = Vector{Float64}()
    costs = Vector{Float64}()
    β = β₀
    for sp ∈ 1:n_steps
        @show sp
        (means_st, variances_st, costs_st) = metropolis!(M_ts, β, n_iter, γ=γ, σ=σ, n_bins=n_bins, dist=dist)
        append!(betas, fill(β, n_iter + 1))
        append!(means, means_st)
        append!(variances, variances_st)
        append!(costs, costs_st)
        β *= α
    end
    return (betas, means, variances, costs)
end

@inline perturbate_whole!(M_ts′::AbstractMatrix{<:Real}, M_ts::AbstractMatrix{<:Real}, σ::Real) =
    _normalize_ts_matrix!(M_ts′, M_ts + randn(size(M_ts)) .* σ)

function metropolis_whole!(M_ts::AbstractMatrix{<:Real}, β::Real, n_iter::Integer;
    σ::Real=0.3, n_bins::Integer=128, dist::F=chisq_dist) where {F}
    t_max, n_series = size(M_ts)
    # Vector to store measurements at each iteration
    costs = Vector{Float64}(undef, n_iter + 1)
    means = Vector{Float64}(undef, n_iter + 1)
    variances = Vector{Float64}(undef, n_iter + 1)
    # Calculate initial distribution
    corr_vals = cross_correlation_values_norm(M_ts)
    # Calculate initial values
    costs[1] = distance_to_uniform(corr_vals, n_bins, dist)
    means[1] = mean(corr_vals)
    variances[1] = varm(corr_vals, means[1])
    # Iterations loop
    M_ts_prev = deepcopy(M_ts)
    for it ∈ 2:n_iter+1
        perturbate_whole!(M_ts, M_ts_prev, σ)
        corr_vals = cross_correlation_values_norm(M_ts)
        cost = distance_to_uniform(corr_vals, n_bins, dist)
        Δcost = cost - costs[it-1]
        # Metropolis prescription
        @views if Δcost <= 0 || exp(-β * Δcost) > rand()
            # Accept
            copy!(M_ts_prev, M_ts)
            means[it] = mean(corr_vals)
            variances[it] = varm(corr_vals, means[it])
            costs[it] = cost
        else
            # Reject
            copy!(M_ts, M_ts_prev)
            means[it] = means[it-1]
            variances[it] = variances[it-1]
            costs[it] = costs[it-1]
        end
    end
    return (means, variances, costs)
end

function simulated_annealing_whole!(M_ts::AbstractMatrix{<:Real},
    β₀::Real, α::Real, n_steps::Integer, n_iter::Integer;
    σ::Real=1.0, n_bins::Integer=128, dist::F=chisq_dist) where {F}
    betas = Vector{Float64}()
    means = Vector{Float64}()
    variances = Vector{Float64}()
    costs = Vector{Float64}()
    β = β₀
    for sp ∈ 1:n_steps
        @show sp
        (means_st, variances_st, costs_st) = metropolis_whole!(M_ts, β, n_iter, σ=σ, n_bins=n_bins, dist=dist)
        append!(betas, fill(β, n_iter + 1))
        append!(means, means_st)
        append!(variances, variances_st)
        append!(costs, costs_st)
        β *= α
    end
    return (betas, means, variances, costs)
end

# Target correlation distribution
# const ρ_dist = Uniform(-1.0, 1.0)
# Time series matrix
const n_series = 128
const t_max = 256
# Perturbation
# const γ = parse(Float64, ARGS[1])
# const run = parse(Int64, ARGS[1])
const σ = parse(Float64, ARGS[1])
const n_bins = 128
const dist_str = ARGS[2]
const dist = eval(Meta.parse(dist_str))
# Simulated annealing parameters
const β₀ = 0.01
const α = 1.1
const β_F = 1000.0
const n_steps = ceil(Int64, log(β_F / β₀) / log(α))
const n_iter = 8192

const output_datafile = datadir(filename("GenUniformCorrDistSA",
    "gamma" => 1, "sigma" => σ, "dist" => dist_str,
    ext="jld2"))
println(output_datafile)

# Generate time series matrix
M_ts = rand(Normal(), t_max, n_series)

println("Starting simulated annealing...")
betas, means, variances, costs = simulated_annealing_whole!(M_ts, β₀, α, n_steps, n_iter, σ=σ, n_bins=n_bins, dist=dist)

jldsave(output_datafile;
    M_ts,
    df=DataFrame(betas=betas, means=means, variances=variances, costs=costs))
