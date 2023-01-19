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

@inline function hist_distance(values::AbstractVector{<:Real}, hist_target::AbstractVector{<:Real}, distance::F) where {F}
    n_edges = length(hist_target) + 1
    hist_fit = normalize(fit(Histogram, values, range(-1, +1, length=n_edges)), mode=:pdf).weights
    return distance(hist_fit, hist_target) / length(hist_target)
end

# Target Histograms
hist_uniform(n_bins::Integer) = fill(0.5, n_bins)
hist_V(n_bins::Integer) = abs.(range(-1, +1, length=n_bins))
hist_ramp(n_bins::Integer) =
    map(range(-1, +1, length=n_bins)) do x
        if x < 0
            return 0
        else
            return 2 * x
        end
    end

@inline perturbate_normalize!(M_ts′::AbstractMatrix{<:Real}, M_ts::AbstractMatrix{<:Real}, σ::Real) =
    _normalize_ts_matrix!(M_ts′, M_ts + randn(size(M_ts)) .* σ)

function metropolis!(M_ts::AbstractMatrix{<:Real},
    hist_target::AbstractVector{<:Real},
    β::Real, n_iter::Integer;
    σ::Real=1.0, distance::F=Distances.sqeuclidean) where {F}
    # Calculate initial cost
    cost = hist_distance(cross_correlation_values_norm(M_ts), hist_target, distance)
    # Iterations loop
    M_ts_prev = deepcopy(M_ts)
    for _ ∈ 1:n_iter
        perturbate_normalize!(M_ts, M_ts_prev, σ)
        cost_new = hist_distance(cross_correlation_values_norm(M_ts), hist_target, distance)
        # Metropolis prescription
        Δcost = cost_new - cost
        if Δcost <= 0 || exp(-β * Δcost) > rand()
            # Accept
            M_ts_prev, M_ts = M_ts, M_ts_prev
            cost = cost_new
        end
    end
    return cost
end

function simulated_annealing!(M_ts::AbstractMatrix{<:Real},
    hist_target::AbstractVector{<:Real},
    β₀::Real, α::Real, n_steps::Integer, n_iter::Integer;
    σ::Real=1.0, distance::F=Distances.sqeuclidean) where {F}
    betas = Vector{Float64}(undef, n_steps + 1)
    costs = Vector{Float64}(undef, n_steps + 1)
    betas[1] = β₀
    for step ∈ 1:n_steps
        costs[step] = metropolis!(M_ts, hist_target, betas[step], n_iter, σ=σ, distance=distance)
        betas[step+1] = α * betas[step]
    end
    return betas, costs
end

# Time series matrix
const n_series = 128
const t_max = 256
# const run = parse(Int64, ARGS[1])
const σ = parse(Float64, ARGS[1])
const n_bins = parse(Int64, ARGS[2])
const hist_target = hist_V(n_bins)
const distance_str = ARGS[3]
const distance = eval(Meta.parse(distance_str))
# Simulated annealing parameters
const β₀ = 1e-5
const α = 1.1
const β_F = 1e5
const n_steps = ceil(Int64, log(β_F / β₀) / log(α))
const n_iter = 8192
const n_samples = 64

const output_datafile = datadir(filename("GenCorrDistSA",
    "hist_target" => "V", "n_bins" => n_bins, "sigma" => σ, "distance" => distance_str, "n_samples" => n_samples,
    ext="jld2"))
println(output_datafile)

# Single sample
M_ts = rand(Normal(), t_max, n_series)
betas, costs = simulated_annealing!(M_ts, hist_target, β₀, α, n_steps, n_iter, σ=σ, distance=distance)
jldsave(output_datafile;
    M_ts,
    df=DataFrame(betas=betas, costs=costs))

# # Multiple samples
# M_ts_samples = [rand(Normal(), t_max, n_series) for _ ∈ 1:n_samples]
# foreach(M_ts_samples) do M_ts
#     simulated_annealing!(M_ts, hist_target, β₀, α, n_steps, n_iter, σ=σ, distance=distance)
# end
# jldsave(output_datafile;
#     M_ts_samples)
